import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer
from utils.self_play_model_utils import *
import math


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


class CoherenceScorer:
    def __init__(self, args, device=None):
        base_model = 'bert-base-cased'
        self.args = args
        config = AutoConfig.from_pretrained(base_model, num_labels=3, finetuning_task='mnli')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=not args.use_slow_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        self.model.load_state_dict(torch.load(args.model_name_or_path))
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

    def predict(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)[0].cpu()
        return torch.argmax(outputs, dim=1).item()

    def score_one_instance(self, history, full_score=0.3):
        idxs_app = [2*i+1 for i in range(math.ceil(len(history)/2-1))]
        score_per_turn = full_score / len(idxs_app)
        coh_score = 0
        for idx in idxs_app:
            premise = history[idx]
            hypothesis = history[idx + 1]
            pred = self.predict(premise.lower(), hypothesis.lower())
            if pred == 0:
                coh_score += score_per_turn
            elif pred == 1:
                coh_score += score_per_turn * 0.2
            else:
                assert pred == 2
                # do nothing
        return coh_score


# class RLTrainer:
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         self.args = args
#         self.wiz = wiz
#         self.app = app
#         self.sel = sel
#         assert len(scorers) == len(alphas)
#         assert sum(alphas) == 1.0
#         self.scorers = scorers
#         self.alphas = alphas
#         self.optimizer = optimizer
#         self.accelerator = accelerator
#         self.wiz.generator.eval()
#         self.app.generator.eval()
#         self.scorer_c = CoverageScorer()
#         if not args.log_dir.endswith('.txt'):
#             self.log_dir = args.log_dir + '/log.txt'
#         else:
#             self.log_dir = args.log_dir
#         with open(self.log_dir, 'w') as f:
#             f.write('***** Running RL Fine-tuning *****\n')
#         self.start_time = datetime.now()
#         if not os.path.exists(self.args.output_dir):
#             os.mkdir(self.args.output_dir)
#
#     def get_reward_score(self, history, document):
#         scores = {}
#         for scorer, alpha in zip(self.scorers, self.alphas):
#             if type(scorer) is CoverageScorer:
#                 score_cov = scorer.score_one_instance(history, document)
#                 scores['cov'] = alpha * score_cov
#             elif type(scorer) is CoherenceScorer:
#                 score_coh = scorer.score_one_instance(history)
#                 scores['coh'] = alpha * score_coh
#             else:
#                 raise NotImplementedError
#         return sum(scores.values()), scores
#
#     def log(self, content):
#         with open(self.log_dir, 'a') as f:
#             f.write(content.strip() + '\n')
#         print(content.strip() + '\n')
#
#     def make_conversation(self, topic, document, reverse=True, greedy=False):
#         raise NotImplementedError
#
#     def train_self_play_rl(self, train_dataset, eval_dataset):
#         # Only show the progress bar once on each machine.
#         progress_bar_train = tqdm(range(self.args.max_train_steps))
#         completed_steps = 0
#         eval_epoch = 0
#         self.sel.selector.train()
#         for i, instance in enumerate(train_dataset):
#             topic, document = instance['topic'], instance['document']
#             try:
#                 ins_start_time = datetime.now()
#                 # sampling
#                 sample_history, RL_log_probs = self.make_conversation(topic, document, reverse=self.args.reverse, greedy=False)
#                 with torch.autograd.no_grad():
#                     # greedy baseline
#                     greedy_history, _ = self.make_conversation(topic, document, reverse=self.args.reverse, greedy=True)
#                 sample_reward, sample_scores = self.get_reward_score(sample_history, document)
#                 baseline_reward, baseline_scores = self.get_reward_score(greedy_history, document)
#                 rl_loss = -(sample_reward - baseline_reward) * RL_log_probs
#                 self.log('\n### Training step %s ###' % i +
#                          '\nSample conversation:\t%s\nSample reward:\tCoverage - %f\tCoherence - %f' % (
#                              '</s>'.join(sample_history), sample_scores['cov'], sample_scores['coh']) +
#                          '\nBaseline conversation\t%s\nBaseline reward:\tCoverage - %f\tCoherence - %f' % (
#                              '</s>'.join(greedy_history), baseline_scores['cov'], baseline_scores['coh']) +
#                          '\nCoverage difference: %f\tCoherence difference: %f' %
#                          (sample_scores['cov']-baseline_scores['cov'], sample_scores['coh']-baseline_scores['coh'])
#                          )
#                 ins_end_time = datetime.now()
#                 self.log('Per instance time: %s\tTotal time consumed: %s' % (str(ins_end_time-ins_start_time), str(ins_end_time-self.start_time)))
#                 # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
#                 # rl_loss = torch.mean(rl_loss)
#                 # batch_reward = torch.mean(sample_reward).item()
#                 self.optimizer.zero_grad()
#                 # rl_loss.backward()
#                 self.accelerator.backward(rl_loss)
#                 self.optimizer.step()
#             except:
#                 self.log('Failed training case, idx %s' % i)
#                 self.log('Topic:\t%s' % topic)
#                 self.log('Document:\t%s' % document)
#             progress_bar_train.update(1)
#             completed_steps += 1
#             if completed_steps % self.args.eval_steps == 0:
#                 self.log('--- Eval epoch %s starts ---' % eval_epoch)
#                 eval_epoch += 1
#                 self.eval_self_play_rl(eval_dataset)
#             if completed_steps % self.args.save_steps == 0:
#                 self.sel.save_model(self.args.output_dir, 'step_%s' % i)
#             if completed_steps >= self.args.max_train_steps:
#                 break
#
#     def eval_self_play_rl(self, dataset):
#         eval_start_time = datetime.now()
#         progress_bar_eval = tqdm(range(len(dataset)))
#         self.sel.selector.eval()
#         all_scores_cov = []
#         all_scores_coh = []
#         for j, instance in enumerate(dataset):
#             topic, document = instance['topic'], instance['document']
#             try:
#                 with torch.autograd.no_grad():
#                     history, _ = self.make_conversation(topic, document, reverse=self.args.reverse, greedy=True)
#                 reward, scores = self.get_reward_score(history, document)
#                 self.log('\n--- Eval step %s ---' % j +
#                          '\nConversation:\t%s\nReward:\t%f' % (
#                              '</s>'.join(history), reward))
#                 all_scores_cov.append(scores['cov'])
#                 all_scores_coh.append(scores['coh'])
#             except:
#                 self.log('Failed eval case, idx %s' % j)
#                 self.log('Topic:\t%s' % topic)
#                 self.log('Document:\t%s' % document)
#             progress_bar_eval.update(1)
#         self.log('--- Eval average coverage score: %f ---' % (np.mean(all_scores_cov)))
#         self.log('--- Eval average coherence score: %f ---' % (np.mean(all_scores_cov)))
#         eval_end_time = datetime.now()
#         self.log('--- Eval time consumed: %s ---' % str(eval_end_time - eval_start_time))
#         self.sel.selector.train()
#
#
# class RLTrainerForPostSelector(RLTrainer):
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         super().__init__(args, wiz, app, sel, scorers, alphas, optimizer, accelerator)
#
#     def make_conversation(self, topic, document, reverse=True, greedy=False):
#         history = ""
#         log_probs = []
#         for i in range(self.args.num_turns):
#             wiz_say_candidates = doha_generate(self.wiz, topic, history, document,
#                                                num_return_sequences=self.args.num_candicates)
#             probs = self.sel.select(wiz_say_candidates, history, document)
#             if not greedy:
#                 multi_dist = Categorical(probs)
#                 idx_candidate = multi_dist.sample()
#                 log_prob = multi_dist.log_prob(idx_candidate)
#                 log_probs.append(log_prob)
#             else:
#                 idx_candidate = torch.argmax(probs)
#             wiz_say = wiz_say_candidates[idx_candidate]
#             history = update_history(history, wiz_say, reverse=reverse)
#             app_say = bart_generate(self.app, history, num_return_sequences=self.args.num_candicates)
#             history = update_history(history, app_say, reverse=reverse)
#         history = parse_conversation_history(history, reverse=reverse)
#         if not greedy:
#             log_probs = torch.stack(log_probs)
#             log_probs = torch.sum(log_probs)
#         return history, log_probs
#
#
# class RLTrainerForPreSelector(RLTrainer):
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         super().__init__(args, wiz, app, sel, scorers, alphas, optimizer, accelerator)
#
#     def make_conversation(self, topic, document, reverse=True, greedy=False):
#         history = ""
#         log_probs = []
#         sents_doc = sent_tokenize(document)
#         for i in range(self.args.num_turns):
#             probs = self.sel.select(sents_doc, history, document)
#             if not greedy:
#                 multi_dist = Categorical(probs)
#                 idx_sent = multi_dist.sample()
#                 while idx_sent == torch.argmax(probs):
#                     idx_sent = multi_dist.sample()
#                 log_prob = multi_dist.log_prob(idx_sent)
#                 log_probs.append(log_prob)
#             else:
#                 idx_sent = torch.argmax(probs)
#             wiz_say = doha_generate(self.wiz, topic, history, sents_doc[idx_sent], num_return_sequences=1)[0]
#             history = update_history(history, wiz_say, reverse=reverse)
#             if i != self.args.num_turns-1:
#                 app_say = bart_generate(self.app, history, num_return_sequences=self.args.num_candicates)
#                 history = update_history(history, app_say, reverse=reverse)
#         history = parse_conversation_history(history, reverse=reverse)
#         if not greedy:
#             log_probs = torch.stack(log_probs)
#             log_probs = torch.sum(log_probs)
#         return history, log_probs

