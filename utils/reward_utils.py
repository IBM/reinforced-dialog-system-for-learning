import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_score import rouge_scorer
from utils.self_play_model_utils import *
from torch.distributions import Categorical
import numpy as np
from datetime import datetime, timedelta


class LanguageModelScorer:
    def __init__(self, model_name_or_path='microsoft/DialoGPT-medium', device=None):
        if not device:
            if not torch.cuda.is_available():
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
        # DialoGPT: The model is trained on 147M multi-turn dialogue from Reddit discussion thread.
        # You may choose Small, Medium or Large
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        # return (5 + sequence_lengths) ** 0.9 / (5 + 1) ** 0.9
        # return torch.sqrt(sequence_lengths)
        return sequence_lengths

    def score_one_instance(self, history):
        # What is the sep token in DialoGPT: It is "<|endoftext|>"
        assert type(history) == list
        history_ = "<|endoftext|>".join(history)
        history_ids = self.tokenizer.encode(history_, return_tensors="pt").to(self.device)
        golden_out = history_ids[:, 1:].unsqueeze(dim=2)
        with torch.no_grad():
            outputs = self.model(history_ids)
            # lm labels should mask the source sentence language model
            shift_logits = outputs[0][..., :-1, :].contiguous()
            # lm_labels = tgt_seq.clone()[..., 1:].contiguous()
            # predict answers
            scores = F.log_softmax(shift_logits, dim=2).gather(2, golden_out)
            # nonzero = golden_out.ne(self.tokenizer.pad_token_id).float()
            scores = (scores).sum(1)
            seq_lengths = golden_out.shape[1] - 1
            # Feature: calculate length penalty
            scores /= self._length_penalty(seq_lengths)
            # scores = scores.sum(dim=1)
        return scores.item()

    def score_batch_instances(self, batch_inputs):
        pass


class CoverageScorer:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        # self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    def score_one_instance(self, history, document):
        assert type(history) is list
        history_ = ' '.join(history)
        scores = self.scorer.score(document, history_)
        return scores['rouge2'].fmeasure


class RLTrainer:
    def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
        self.args = args
        self.wiz = wiz
        self.app = app
        self.sel = sel
        assert len(scorers) == len(alphas)
        assert sum(alphas) == 1.0
        self.scorers = scorers
        self.alphas = alphas
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.wiz.generator.eval()
        self.app.generator.eval()
        self.scorer_c = CoverageScorer()
        if not args.log_dir.endswith('.txt'):
            self.log_dir = args.log_dir + '/log.txt'
        else:
            self.log_dir = args.log_dir
        with open(self.log_dir, 'w') as f:
            f.write('***** Running RL Fine-tuning *****\n')
        self.start_time = datetime.now()
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)

    def get_reward_score(self, history, document):
        score = 0
        for scorer, alpha in zip(self.scorers, self.alphas):
            if type(scorer) is CoverageScorer:
                score_c = scorer.score_one_instance(history, document)
                score += alpha * score_c
            elif type(scorer) is LanguageModelScorer:
                score_lm = scorer.score_one_instance(history)
                score += alpha * score_lm
            else:
                raise NotImplementedError
        return score

    def log(self, content):
        with open(self.log_dir, 'a') as f:
            f.write(content.strip() + '\n')
        print(content.strip() + '\n')

    def make_conversation(self, topic, document, reverse=True, greedy=False):
        history = ""
        log_probs = []
        for i in range(self.args.num_turns):
            wiz_say_candidates = doha_generate(self.wiz, topic, history, document,
                                               num_return_sequences=self.args.num_candicates)
            probs = self.sel.select(wiz_say_candidates, history)
            if not greedy:
                multi_dist = Categorical(probs)
                idx_candidate = multi_dist.sample()
                log_prob = multi_dist.log_prob(idx_candidate)
                log_probs.append(log_prob)
            else:
                idx_candidate = torch.argmax(probs)
            wiz_say = wiz_say_candidates[idx_candidate]
            history = update_history(history, wiz_say, reverse=reverse)
            app_say = bart_generate(self.app, history, num_return_sequences=self.args.num_candicates)
            history = update_history(history, app_say, reverse=reverse)
        history = parse_conversation_history(history, reverse=reverse)
        if not greedy:
            log_probs = torch.stack(log_probs)
            log_probs = torch.sum(log_probs)
        return history, log_probs

    def train_self_play_rl(self, train_dataset, eval_dataset):
        # Only show the progress bar once on each machine.
        progress_bar_train = tqdm(range(self.args.max_train_steps))
        completed_steps = 0
        eval_epoch = 0
        self.sel.model.train()
        for i, instance in enumerate(train_dataset):
            topic, document = instance['topic'], instance['document']
            try:
                ins_start_time = datetime.now()
                # sampling
                sample_history, RL_log_probs = self.make_conversation(topic, document, reverse=self.args.reverse, greedy=False)
                with torch.autograd.no_grad():
                    # greedy baseline
                    greedy_history, _ = self.make_conversation(topic, document, reverse=self.args.reverse, greedy=True)
                sample_reward = self.get_reward_score(sample_history, document)
                baseline_reward = self.get_reward_score(greedy_history, document)
                rl_loss = -(sample_reward - baseline_reward) * RL_log_probs
                self.log('\n### Training step %s ###' % i +
                         '\nSample conversation:\t%s\nSample reward:\t%f' % (
                             '</s>'.join(sample_history), sample_reward) +
                         '\nBaseline conversation\t%s\nBaseline reward:\t%f' % (
                             '</s>'.join(greedy_history), baseline_reward))
                ins_end_time = datetime.now()
                self.log('Per instance time: %s\tTotal time consumed: %s' % (str(ins_end_time-ins_start_time), str(ins_end_time-self.start_time)))
                # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
                # rl_loss = torch.mean(rl_loss)
                # batch_reward = torch.mean(sample_reward).item()
                self.optimizer.zero_grad()
                # rl_loss.backward()
                self.accelerator.backward(rl_loss)
                self.optimizer.step()
            except:
                self.log('Failed training case, idx %s' % i)
                self.log('Topic:\t%s' % topic)
                self.log('Document:\t%s' % document)
            progress_bar_train.update(1)
            completed_steps += 1
            if completed_steps % self.args.eval_steps == 0:
                self.log('--- Eval epoch %s starts ---' % eval_epoch)
                eval_epoch += 1
                self.eval_self_play_rl(eval_dataset)
            if completed_steps % self.args.save_steps == 0:
                self.sel.save_model(self.args.output_dir, 'step_%s' % i)
            if completed_steps >= self.args.max_train_steps:
                break

    def eval_self_play_rl(self, dataset):
        eval_start_time = datetime.now()
        progress_bar_eval = tqdm(range(len(dataset)))
        self.sel.model.eval()
        all_rewards = []
        for j, instance in enumerate(dataset):
            topic, document = instance['topic'], instance['document']
            try:
                with torch.autograd.no_grad():
                    history, _ = self.make_conversation(topic, document, reverse=self.args.reverse, greedy=True)
                reward = self.get_reward_score(history, document)
                self.log('\n--- Eval step %s ---' % j +
                         '\nConversation:\t%s\nReward:\t%f' % (
                             '</s>'.join(history), reward))
                all_rewards.append(reward)
            except:
                self.log('Failed eval case, idx %s' % j)
                self.log('Topic:\t%s' % topic)
                self.log('Document:\t%s' % document)
            progress_bar_eval.update(1)
        self.log('--- Eval average reward: %f ---' % (np.mean(all_rewards)))
        eval_end_time = datetime.now()
        self.log('--- Eval time consumed: %s ---' % str(eval_end_time - eval_start_time))
        self.sel.model.train()

