from datetime import datetime
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch
import math
import random
from transformers import AutoModelWithLMHead, BartTokenizer, AutoTokenizer
from tqdm import tqdm
import sys
import os
from utils.self_play_infra_utils import get_linear_schedule_with_warmup
from utils.self_play_model_utils import bleu, DataloaderRL


class RLTrainerForGenerator:
    def __init__(self, args, wiz, app, scorers, alphas, accelerator):
        self.args = args
        self.wiz = wiz
        self.app = app
        assert len(scorers) == len(alphas)
        assert sum(alphas) == 1.0
        assert len(scorers) == 2
        self.scorer_cov, self.scorer_coh = scorers
        assert len(alphas) == 2
        self.alpha_cov, self.alpha_coh = alphas
        self.accelerator = accelerator
        self.optimizer = self.accelerator.prepare(self.wiz.get_optimizer())
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=int(self.args.warmup_steps),
                                                    num_training_steps=self.args.max_train_steps)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        if not args.log_dir.endswith('.txt'):
            self.log_dir = args.log_dir + '/log.txt'
        else:
            self.log_dir = args.log_dir
        if self.args.write_to_log:
            with open(self.log_dir, 'w') as f:
                f.write('***** Running RL Fine-tuning *****\n')
        self.start_time = datetime.now()
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        if hasattr(args, 'do_post_process'):
            if args.do_post_process:
                self.post_processor = PostProcessor()
        else:
            self.post_processor = None

    def generate_wiz_response(self, topics, histories, documents, do_post_process=False, do_resampling=False):
        source = ['%s. %s' % (topic, history) for topic, history in zip(topics, histories)]
        source_ = self.wiz.tokenizer(source, return_tensors='pt', truncation=True, padding=True)
        source_ids, source_mask = source_['input_ids'].to(self.accelerator.device), source_['attention_mask'].to(
            self.accelerator.device)
        documents_ = self.convert_documents(source, documents)
        documents_ = self.wiz.tokenizer(documents_, return_tensors='pt', truncation=True, padding=True)
        doc_ids, doc_mask = documents_['input_ids'].to(self.accelerator.device), documents_['attention_mask'].to(
            self.accelerator.device)
        with torch.no_grad():
            source_reps, doc_reps = self.wiz.encode(source_ids, source_mask, doc_ids, doc_mask)
            predicted_ids = self.wiz.generator.generate(input_ids=source_mask,
                                                        attention_mask=(source_mask, doc_mask),
                                                        encoder_outputs=(source_reps, doc_reps),
                                                        num_beams=3,
                                                        min_length=9,
                                                        max_length=self.args.max_target_length,
                                                        early_stopping=True,
                                                        do_sample=True,
                                                        temperature=1.0,
                                                        top_k=50,
                                                        top_p=0.9,
                                                        num_return_sequences=1,
                                                        decoder_start_token_id=0,
                                                        repetition_penalty=1.2,
                                                        )
        if not do_post_process:
            responses_all = self.wiz.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        else:
            predicted_ids_ = torch.cat((predicted_ids[:, 0].reshape(1, -1), predicted_ids[:, 2:]), dim=1)
            responses_all = self.wiz.tokenizer.batch_decode(predicted_ids_, skip_special_tokens=True)
            responses_all_ = []
            for response, document in zip(responses_all, documents):
                responses_all_.append(self.post_processor.post_process(response, document))
            responses_all = responses_all_
        return responses_all

    def generate_wiz_response_finetune(self, source_ids, doc_ids, source_mask, doc_mask, do_sample=False, config=None):
        # This generate function uses num_beams=1
        if config is None:
            config = self.wiz.generator.config
        batch_size, _ = source_ids.shape
        predict_ids = torch.full(
            (batch_size, 1),
            0,
            # config.decoder_start_token_id,
            dtype=torch.long,
            device=next(self.wiz.generator.parameters()).device,
        )
        unfinished_sents = predict_ids.new(batch_size).fill_(1)
        sent_lengths = predict_ids.new(batch_size).fill_(self.args.max_target_length)
        past = None
        cur_len = predict_ids.shape[-1]
        log_probs = []
        decoder_padding_mask = []
        source_reps, doc_reps = self.wiz.encode(source_ids, source_mask, doc_ids, doc_mask)
        while cur_len < self.args.max_target_length:
            model_inputs = self.wiz.generator.prepare_inputs_for_generation(
                predict_ids,
                past=past,
                attention_mask=(source_mask, doc_mask),
                encoder_outputs=(source_reps, doc_reps),
                use_cache=config.use_cache
            )
            outputs = self.wiz.generator(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            scores = self.wiz.generator.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=predict_ids,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                bad_words_ids=config.bad_words_ids,
                cur_len=cur_len,
                min_length=config.min_length,
                max_length=self.args.max_target_length,
                eos_token_id=config.eos_token_id,
                repetition_penalty=config.repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems
            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if config.temperature != 1.0:
                    scores = scores / config.temperature
                # Top-p/top-k filtering
                # next_token_logscores = top_k_top_p_filtering(scores, top_k=config.top_k, top_p=config.top_p)
                next_token_logscores = scores
                # Sample
                prob = F.softmax(next_token_logscores, dim=-1)
                multi_dist = Categorical(prob)
                # next_token = torch.multinomial(prob, num_samples=1).squeeze(1)
                next_token = multi_dist.sample()
                log_prob = multi_dist.log_prob(next_token)
                log_probs.append(log_prob)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
            # update generations and finished sentences
            if config.eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (config.pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            # add token and increase length by one
            predict_ids = torch.cat([predict_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            if config.eos_token_id is not None:
                eos_in_sents = tokens_to_add == config.eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                _ = unfinished_sents.mul_((~eos_in_sents).long())
                decoder_padding_mask.append(unfinished_sents.clone())
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
            # # extend attention_mask for new generated input if only decoder
            # if config.is_encoder_decoder is False:
            #     attention_mask = torch.cat(
            #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            #     )
        if do_sample:
            log_probs = torch.stack(log_probs, dim=1)
            decoder_padding_mask = torch.stack(decoder_padding_mask, dim=1)
            log_probs = log_probs * decoder_padding_mask
            # lens = torch.sum(decoder_padding_mask, dim=1)
            lens_ = torch.unsqueeze(torch.sum(decoder_padding_mask, dim=1), dim=0)
            lens = torch.max(torch.cat((lens_, torch.ones_like(lens_, dtype=lens_.dtype)), dim=0), dim=0).values
            log_probs = torch.sum(log_probs, dim=1) / lens
        responses = self.wiz.tokenizer.batch_decode(predict_ids, skip_special_tokens=True)
        return responses, log_probs

    def select_app_response(self, responses_all, histories, thresh=0.5):
        assert len(responses_all) == len(histories) * self.args.num_candidates
        responses_selected = []
        for i, history in enumerate(histories):
            responses_ = responses_all[i * self.args.num_candidates: (i + 1) * self.args.num_candidates]
            previous_responses = [rec for i, rec in enumerate(history.split(' / ')) if i % 2 == 1]
            if len(self.app.response_cache) > self.args.num_cached_responses:
                cached_responses = random.choices(self.app.response_cache, k=self.args.num_cached_responses)
            else:
                cached_responses = self.app.response_cache
            reference_responses = previous_responses + cached_responses
            if len(reference_responses) == 0:
                responses_selected.append(random.choice(responses_))
                continue
            cand2score = {}
            for j, response in enumerate(responses_):
                refs = {j: reference_responses}
                hyps = {j: [response]}
                (b1, b2, b3), _ = bleu.compute_score(refs, hyps)
                cand2score[response] = b2
            try:
                selection_pool = [cand for cand, score in cand2score.items() if score < thresh]
                assert len(selection_pool) > 0
            except:
                cand_ranked = sorted(cand2score.items(), key=lambda kv: kv[1])
                selection_pool = [cand_ranked[0][0]]
                print("Warning: Apprentice's responses may have low diversity")
            responses_selected.append(random.choice(selection_pool))
        return responses_selected

    def generate_app_response(self, histories):
        source = self.convert_documents(histories, None)
        source_ = self.app.tokenizer(source, return_tensors='pt', truncation=True, padding=True)
        source_ids, source_mask = source_['input_ids'].to(self.accelerator.device), source_['attention_mask'].to(
            self.accelerator.device)
        with torch.no_grad():
            predicted_ids = self.app.generator.generate(input_ids=source_ids,
                                                        attention_mask=source_mask,
                                                        num_beams=10,
                                                        do_sample=True,
                                                        max_length=self.args.max_target_length,
                                                        early_stopping=True,
                                                        top_p=0.9,
                                                        top_k=50,
                                                        num_return_sequences=self.args.num_candidates
                                                        )
        responses_all = self.app.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        responses_selected = self.select_app_response(responses_all, histories)
        self.app.update_response_history(random.choices(responses_selected, k=2))
        return responses_selected

    def get_lengthy_penalty_scale(self, responses):
        # penalize responses that are too long or too short
        penalty_scales = []
        for response in responses:
            response_length = len(self.tokenizer.tokenize(response))
            if response_length < 3:
                penalty_scales.append(0.0)
            elif response_length < 10:
                penalty_scales.append(0.1 * response_length)
            elif response_length <= 45:
                penalty_scales.append(1.0)
            elif 45 < response_length <= 55:
                penalty_scales.append(0.7)
            elif 55 < response_length <= 65:
                penalty_scales.append(0.5)
            elif 65 < response_length <= 75:
                penalty_scales.append(0.3)
            else:
                penalty_scales.append(0.1)
        return np.array(penalty_scales)

    def get_reward_score(self, responses, histories, documents):
        cov_scores = self.scorer_cov.score_utterance_for_generator(responses, histories, documents)
        # if there is no previous response, do not calculate coherence score
        if self.alpha_coh != 0:
            if histories[0] != '':
                coh_scores = self.scorer_coh.score_utterance_for_generator(responses, histories)
            else:
                coh_scores = np.zeros(len(responses))
        else:
            coh_scores = np.zeros(len(responses))
        scores = (self.alpha_cov * cov_scores + self.alpha_coh * coh_scores) * self.get_lengthy_penalty_scale(responses)
        return scores, (cov_scores, coh_scores)

    def finetune_batch_rl(self, batch):
        topics, documents = batch['topic'], batch['document']
        histories = ['' for _ in topics]
        for j in range(self.args.num_turns):
            try:
                self.log('Turn %s' % j)
                # wiz say
                wiz_says = self.finetune_step_rl(topics, documents, histories)
                histories = self.update_histories(wiz_says, histories)
                # app say
                if not j == self.args.num_turns - 1:
                    # The app does not need to respond in the last turn
                    app_says = self.generate_app_response(histories)
                    histories = self.update_histories(app_says, histories)
            except:
                self.log('Error occured\ntopics: %s' % topics)
                break
        return histories

    def finetune_step_rl(self, topics, documents, histories):
        source = ['%s. %s' % (topic, history) for topic, history in zip(topics, histories)]
        source_ = self.wiz.tokenizer(source, return_tensors='pt', truncation=True, padding=True)
        source_ids, source_mask = source_['input_ids'].to(self.accelerator.device), source_['attention_mask'].to(
            self.accelerator.device)
        documents_ = self.convert_documents(source, documents)
        documents_ = self.wiz.tokenizer(documents_, return_tensors='pt', truncation=True, padding=True)
        doc_ids, doc_mask = documents_['input_ids'].to(self.accelerator.device), documents_['attention_mask'].to(
            self.accelerator.device)
        with torch.no_grad():
            greedy_responses, _ = self.generate_wiz_response_finetune(source_ids, doc_ids, source_mask, doc_mask,
                                                                      do_sample=False)
        sample_responses, RL_log_probs = self.generate_wiz_response_finetune(source_ids, doc_ids, source_mask, doc_mask,
                                                                             do_sample=True)

        sample_scores, (sample_cov_scores, sample_coh_scores) = self.get_reward_score(sample_responses, histories,
                                                                                      documents)
        greedy_scores, (greedy_cov_scores, greedy_coh_scores) = self.get_reward_score(greedy_responses, histories,
                                                                                      documents)
        rl_loss = torch.from_numpy(-(sample_scores - greedy_scores)).to(RL_log_probs.device) * RL_log_probs
        rl_loss = torch.mean(rl_loss)
        # check NaN
        # if torch.isnan(rl_loss).sum() > 0:
        #     info = {
        #         'rl_loss': rl_loss,
        #         'RL_log_probs': RL_log_probs,
        #         'sample_scores': sample_scores,
        #         'greedy_scores': greedy_scores,
        #         'topics': topics,
        #         'documents': documents,
        #         'histories': histories,
        #     }
        #     torch.save(info, '../Talk_/za/infox.pt')
        #     exit()
        self.log("Sample scores | final:%s\tcov: %s\tcoh: %s\tseq_len: %s\nGreedy scores | final:%s\tcov: %s\tcoh: %s\tseq_len: %s" % (
            round(sample_scores[0], 3), round(sample_cov_scores[0], 3), round(sample_coh_scores[0], 3), len(sample_responses[0].split(' ')),
            round(greedy_scores[0], 3), round(greedy_cov_scores[0], 3), round(greedy_coh_scores[0], 3), len(greedy_responses[0].split(' '))))
        # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
        self.optimizer.zero_grad()
        # rl_loss.backward()
        self.accelerator.backward(rl_loss)
        self.log('RL loss %s' % round(rl_loss.cpu().item(), 3))
        torch.nn.utils.clip_grad_norm_(self.wiz.generator.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        return sample_responses

    def finetune_batch_mle(self, batch):
        source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens = self.wiz.get_train_batch_data(
            batch)
        source_reps, doc_reps = self.wiz.encode(source_ids, source_mask, doc_ids, doc_mask)
        outputs = self.wiz.generator(input_ids=None,
                                     attention_mask=(source_mask, doc_mask),
                                     encoder_outputs=(source_reps, doc_reps),
                                     decoder_input_ids=target_ids,
                                     lm_labels=target_labels,
                                     labels=target_labels)
        mle_loss = outputs[0]
        # if self.args.gradient_accumulation_steps > 1:
        #     loss = loss / self.args.gradient_accumulation_steps
        self.optimizer.zero_grad()
        self.accelerator.backward(mle_loss)
        torch.nn.utils.clip_grad_norm_(self.wiz.generator.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        # train_stat_curr = {
        #     'loss': loss.item(),
        #     'train_ppl': math.exp(min(loss / batch_total_tokens, 100))
        # }
        # print(str(train_stat_curr))
        self.log("MLE loss %s | Perplexity: %s" % (mle_loss.item(), math.exp(min(mle_loss / batch_total_tokens, 100))))
        sys.stdout.flush()

    def finetune(self, train_dataset_rl, eval_dataset_rl, train_dataset_mle):
        eval_epoch = 0
        self.wiz.generator.train()
        self.app.generator.eval()
        # max_train_steps_possible = int(self.args.max_train_steps / self.args.batch_size_rl)
        # assert max_train_steps_possible >= self.args.max_train_steps
        progress_bar_train = tqdm(range(self.args.max_train_steps))
        train_dataloader_rl = DataloaderRL(train_dataset_rl, self.args.batch_size_rl, shuffle=True)
        eval_dataloader_rl = DataloaderRL(eval_dataset_rl, self.args.batch_size_rl, shuffle=False)
        train_dataloader_mle = iter(self.wiz.get_train_dataloader(train_dataset_mle, self.args.batch_size_mle))
        for step in range(self.args.max_train_steps):
            ins_start_time = datetime.now()
            if self.args.finetune_mle is True:
                for _ in range(self.args.num_mle_per_rl):
                    batch_mle = next(train_dataloader_mle)
                    self.finetune_batch_mle(batch_mle)
            if self.args.finetune_rl is True:
                batch_rl = train_dataloader_rl.get_next_batch()
                _ = self.finetune_batch_rl(batch_rl)
            ins_end_time = datetime.now()
            self.log('Step %s | Step time: %s\tTotal time: %s' % (
                step, str(ins_end_time - ins_start_time).split('.')[0],
                str(ins_end_time - self.start_time).split('.')[0]))
            self.scheduler.step()
            progress_bar_train.update(1)
            if step % self.args.eval_steps == 0 and step > 0:
                self.log('--- Eval epoch %s starts ---' % eval_epoch)
                self.evaluate(eval_dataloader_rl)
                self.log('--- Eval epoch %s ends ---' % eval_epoch)
                eval_epoch += 1
            if step % self.args.save_steps == 0 and step > 0:
                self.wiz.generator.save_pretrained(self.args.output_dir + 'step_%s' % step)
            if step >= self.args.max_train_steps:
                break
            if self.args.finetune_mle:
                if (step + 2) * self.args.num_mle_per_rl >= len(train_dataloader_mle):
                    train_dataloader_mle = iter(
                        self.wiz.get_train_dataloader(train_dataset_mle, self.args.batch_size_mle))

    def evaluate(self, eval_dataloader):
        eval_start_time = datetime.now()
        progress_bar_eval = tqdm(range(eval_dataloader.num_steps))
        self.wiz.generator.eval()
        all_cov_scores = []
        all_coh_scores = []
        for step in range(eval_dataloader.num_steps):
            batch = eval_dataloader.get_next_batch()
            topics, documents = batch['topic'], batch['document']
            histories = ['' for _ in topics]
            try:
                ins_start_time = datetime.now()
                for j in range(self.args.num_turns):
                    wiz_says = self.generate_wiz_response(topics, histories, documents)
                    histories = self.update_histories(wiz_says, histories, reverse=self.args.reverse)
                    if not j == self.args.num_turns - 1:
                        # The app does not need to respond in the last turn
                        app_says = self.generate_app_response(histories)
                        histories = self.update_histories(app_says, histories, reverse=self.args.reverse)
                cov_scores = self.scorer_cov.score_conversation_for_generator(histories, documents)
                coh_scores = self.scorer_coh.score_conversation_for_generator(histories)
                all_cov_scores += cov_scores
                all_coh_scores += coh_scores
                ins_end_time = datetime.now()
                for k, history in enumerate(histories):
                    self.log('Step %s instance %s | Cov score: %s | Coh score: %s | Time: %s\nHistory: %s' % (
                        step, k, round(cov_scores[k], 3), round(coh_scores[k], 3),
                        str(ins_end_time - ins_start_time).split('.')[0],
                        self.reformat_history(history, reverse=self.args.reverse)))
            except:
                self.log('Failed | Step  %s\nTopics:\t%s' % (step, topics))
            progress_bar_eval.update(1)
        self.log('--- Eval average coverage score: %f ---' % (np.mean(all_cov_scores)))
        self.log('--- Eval average coherence score: %f ---' % (np.mean(all_coh_scores)))
        eval_end_time = datetime.now()
        self.log('--- Eval time consumed: %s ---' % str(eval_end_time - eval_start_time))
        self.wiz.generator.train()

    def log(self, content):
        if self.args.write_to_log:
            with open(self.log_dir, 'a') as f:
                f.write(content.strip() + '\n')
        print(content.strip())

    def update_histories(self, responses, histories, reverse=True):
        assert len(responses) == len(histories)
        for i, (response, history) in enumerate(zip(responses, histories)):
            if history != '':
                if reverse:
                    histories[i] = response + ' / ' + history
                else:
                    histories[i] = history + ' / ' + response
            else:
                histories[i] = response
        return histories

    def reformat_history(self, history, reverse=True):
        uttrs = history.split(' / ')
        if reverse:
            uttrs.reverse()
        return ' / '.join(uttrs)

    def convert_documents(self, source, documents):
        # The original model put the source and document together
        if documents is None:
            documents = ['None' for _ in source]
        documents_ = ['chat: %s document: %s' % (s, d) for s, d in zip(source, documents)]
        return documents_


class PostProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("flexudy/t5-base-multi-sentence-doctor")
        self.model = AutoModelWithLMHead.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

    def post_process(self, response, document):
        sents_doc = document.split('. ')
        sents_doc = '{' + '}{'.join(sents_doc) + '}'
        input_text = "repair_sentence: %s context: %s </s>" % (response, sents_doc)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=128, num_beams=1)
        sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return sentence


# class RLTrainerForSelector:
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         self.args = args
#         self.wiz = wiz
#         self.app = app
#         self.sel = sel
#         assert len(scorers) == len(alphas)
#         # assert sum(alphas) == 1.0
#         self.scorers = scorers
#         self.alphas = alphas
#         self.optimizer = optimizer
#         self.accelerator = accelerator
#         self.wiz.generator.eval()
#         self.app.generator.eval()
#         self.scorer_c = CoverageScorer(max_cov_score=args.max_cov_score)
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
#     def log(self, content):
#         with open(self.log_dir, 'a') as f:
#             f.write(content.strip() + '\n')
#         print(content.strip() + '\n')
#
#     def tokenize_doc(self, documents):
#         if type(documents) == str:
#             references = sent_tokenize(documents)
#         else:
#             assert type(documents) == list
#             references = [sent_tokenize(document) for document in documents]
#         return references
#
#
# class RLTrainerForSelectorConv(RLTrainerForSelector):
#     '''
#     The reward is back propagated after each conversation is finished
#     '''
#
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         super().__init__(args, wiz, app, sel, scorers, alphas, optimizer, accelerator)
#
#     def get_reward_score(self, history, document):
#         scores = {}
#         for scorer, alpha in zip(self.scorers, self.alphas):
#             if type(scorer) is CoverageScorer:
#                 score_cov = scorer.score_conversation(history, document)
#                 scores['cov'] = alpha * score_cov
#             elif type(scorer) is CoherenceScorer:
#                 score_coh = scorer.score_conversation(history)
#                 scores['coh'] = alpha * score_coh
#             else:
#                 raise NotImplementedError
#         return sum(scores.values()), scores
#
#     def make_conversation(self, topic, document, reverse=True, greedy=False):
#         raise NotImplementedError
#
#     def train_self_play_rl_step(self, idx, topics, documents):
#         ins_start_time = datetime.now()
#         # sampling
#         sample_histories, RL_log_probs = self.make_conversation(topics, documents, reverse=self.args.reverse,
#                                                                 greedy=False)
#         with torch.autograd.no_grad():
#             # greedy baseline
#             greedy_histories, _ = self.make_conversation(topics, documents, reverse=self.args.reverse, greedy=True)
#         sample_rewards_batch, sample_scores_batch = self.get_reward_score(sample_histories, documents)
#         baseline_rewards_batch, greedy_scores_batch = self.get_reward_score(greedy_histories, documents)
#         rl_loss = -(sample_rewards_batch - baseline_rewards_batch) * RL_log_probs
#         self.log('\n### Training step %s ###' % idx +
#                  '\nSample conversation:\t%s\nSample reward:\tCoverage - %f\tCoherence - %f' % (
#                      '</s>'.join(sample_histories), sample_scores_batch['cov'], sample_scores_batch['coh']) +
#                  '\nBaseline conversation\t%s\nBaseline reward:\tCoverage - %f\tCoherence - %f' % (
#                      '</s>'.join(greedy_histories), greedy_scores_batch['cov'], greedy_scores_batch['coh']) +
#                  '\nCoverage difference: %f\tCoherence difference: %f' %
#                  (sample_scores_batch['cov'] - greedy_scores_batch['cov'],
#                   sample_scores_batch['coh'] - greedy_scores_batch['coh'])
#                  )
#         ins_end_time = datetime.now()
#         self.log('Per instance time: %s\tTotal time consumed: %s' % (
#             str(ins_end_time - ins_start_time), str(ins_end_time - self.start_time)))
#         # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
#         # rl_loss = torch.mean(rl_loss)
#         # batch_reward = torch.mean(sample_reward).item()
#         self.optimizer.zero_grad()
#         # rl_loss.backward()
#         self.accelerator.backward(rl_loss)
#         self.optimizer.step()
#
#     def train_self_play_rl(self, train_dataset, eval_dataset):
#         # Only show the progress bar once on each machine.
#         progress_bar_train = tqdm(range(int(self.args.max_train_steps / self.args.batch_size)))
#         completed_steps = 0
#         eval_epoch = 0
#         self.sel.selector.train()
#         num_steps = math.floor(len(train_dataset) / self.args.batch_size)
#         indexes = list(range(len(train_dataset)))
#         if self.args.shuffle:
#             random.shuffle(indexes)
#         for i in range(num_steps):
#             indexes_batch = indexes[i * self.args.batch_size: (i + 1) * self.args.batch_size]
#             instances = train_dataset[indexes_batch]
#             topics, documents = instances['topic'], instances['document']
#             try:
#                 self.train_self_play_rl_step(i, topics, documents)
#             except:
#                 self.log('Failed training case, idx %s' % i)
#                 self.log('Topic:\t%s' % topics)
#                 self.log('Document:\t%s' % documents)
#             progress_bar_train.update(1)
#             completed_steps += 1
#             if completed_steps % self.args.eval_steps == 0:
#                 self.log('--- Eval epoch %s starts ---' % eval_epoch)
#                 eval_epoch += 1
#                 self.eval_self_play_rl(eval_dataset)
#             if completed_steps % self.args.save_steps == 0:
#                 self.sel.save_model(self.args.output_dir, '/step_%s' % i)
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
#         self.log('--- Eval average coherence score: %f ---' % (np.mean(all_scores_coh)))
#         eval_end_time = datetime.now()
#         self.log('--- Eval time consumed: %s ---' % str(eval_end_time - eval_start_time))
#         self.sel.selector.train()
#
#
# class RLTrainerForPostSelectorConv(RLTrainerForSelectorConv):
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
#             if i != self.args.num_turns - 1:  # the app do not need to response in the last turn
#                 app_say = bart_generate(self.app, history, num_return_sequences=self.args.num_candicates)
#                 history = update_history(history, app_say, reverse=reverse)
#         history = parse_conversation_history(history, reverse=reverse)
#         if not greedy:
#             log_probs = torch.stack(log_probs)
#             log_probs = torch.sum(log_probs)
#         return history, log_probs
#
#
# class RLTrainerForPreSelectorConv(RLTrainerForSelectorConv):
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         super().__init__(args, wiz, app, sel, scorers, alphas, optimizer, accelerator)
#
#     def make_conversation_old(self, topic, document, reverse=True, greedy=False):
#         history = ""
#         log_probs = []
#         sents_doc = sent_tokenize(document)
#         for i in range(self.args.num_turns):
#             probs = self.sel.select(sents_doc, history, document)
#             if not greedy:
#                 multi_dist = Categorical(probs)
#                 idx_sent = multi_dist.sample()
#                 while idx_sent == torch.argmax(probs) and len(sents_doc) > 1:
#                     idx_sent = multi_dist.sample()
#                 log_prob = multi_dist.log_prob(idx_sent)
#                 log_probs.append(log_prob)
#             else:
#                 idx_sent = torch.argmax(probs)
#             wiz_say = doha_generate(self.wiz, topic, history, sents_doc[idx_sent], num_return_sequences=1)[0]
#             history = update_history(history, wiz_say, reverse=reverse)
#             if i != self.args.num_turns - 1:  # the app do not need to response in the last turn
#                 app_say = bart_generate(self.app, history, num_return_sequences=self.args.num_candicates)
#                 history = update_history(history, app_say, reverse=reverse)
#         history = parse_conversation_history(history, reverse=reverse)
#         if not greedy:
#             log_probs = torch.stack(log_probs)
#             log_probs = torch.sum(log_probs)
#         return history, log_probs
#
#     def make_conversation(self, topics, documents, reverse=True, greedy=False):
#         histories = ['' for _ in range(len(topics))]
#         references = self.tokenize_doc(documents)
#         log_probs = None
#         for i in range(self.args.num_turns):
#             probs_batch = self.sel.select(references, histories, documents)
#             if not greedy:
#                 multi_dist = Categorical(probs_batch)
#                 idx_sent = multi_dist.sample()
#                 # force the sampler to select sth different from the baseline
#                 # while idx_sent == torch.argmax(probs_batch) and len(references) > 1:
#                 #     idx_sent = multi_dist.sample()
#                 log_prob = multi_dist.log_prob(idx_sent)
#                 log_probs.append(log_prob)
#             else:
#                 idx_sent = torch.argmax(probs_batch)
#             sents = [references[i][idx_sent[i]] for i in range(len(idx_sent))]
#
#
# class RLTrainerForSelectorUttr(RLTrainerForSelector):
#     '''
#     The reward is back propagated after each conversation is finished
#     '''
#
#     def __init__(self, args, wiz, app, sel, scorers, alphas, optimizer, accelerator):
#         super().__init__(args, wiz, app, sel, scorers, alphas, optimizer, accelerator)
#
#     def make_utterance(self, topics, documents, histories, greedy):
#         references = self.tokenize_doc(documents)
#         probs_batch = self.sel.select(references, histories, documents)
#         log_probs = None
#         if not greedy:
#             multi_dist = Categorical(probs_batch)
#             idx_sent = multi_dist.sample()
#             # force the sampler to select sth different from the baseline
#             # while idx_sent == torch.argmax(probs_batch) and len(references) > 1:
#             #     idx_sent = multi_dist.sample()
#             log_probs = multi_dist.log_prob(idx_sent).to(self.sel.device)
#         else:
#             idx_sent = torch.argmax(probs_batch, dim=1)
#         sents = [references[i][idx_sent[i]] for i in range(len(idx_sent))]
#         wiz_say = doha_generate(self.wiz, topics, histories, sents, num_return_sequences=1)
#         return wiz_say, log_probs
#
#     def get_reward_score(self, documents, utterances, histories, reverse, get_coh):
#         if type(documents) == str:
#             # input is a single instance
#             scores = {}
#             for scorer, alpha in zip(self.scorers, self.alphas):
#                 if type(scorer) is CoverageScorer:
#                     old_cov = scorer.score_utterance(histories, documents)
#                     now_cov = scorer.score_utterance(update_history(histories, utterances, reverse=reverse), documents)
#                     score_cov = now_cov - old_cov
#                     scores['cov'] = alpha * score_cov
#                 elif type(scorer) is CoherenceScorer:
#                     if get_coh:
#                         score_coh = scorer.score_utterance(utterances, histories)
#                         scores['coh'] = alpha * score_coh
#                     else:
#                         scores['coh'] = 0.0
#                 else:
#                     raise NotImplementedError
#             return sum(scores.values()), scores
#         else:
#             assert type(documents) == list
#             # input is a batch
#             rewards_batch, scores_batch = [], []
#             for document, utterance, history in zip(documents, utterances, histories):
#                 reward, score = self.get_reward_score(document, utterance, history, reverse, get_coh)
#                 rewards_batch.append(reward)
#                 scores_batch.append(score)
#             return torch.from_numpy(np.array(rewards_batch)).to(self.sel.device), scores_batch
#
#     def log_step(self, histories, sample_utterances_batch, greedy_utterances_batch, sample_scores_batch,
#                  greedy_scores_batch):
#         for i, history in enumerate(histories):
#             log_str = 'Conversation %s\n' % i
#             history_ = parse_conversation_history(history, reverse=self.args.reverse)
#             wiz_positions = [2 * i for i in range(math.ceil(len(history_) / 2))]
#             app_positions = [2 * i + 1 for i in range(math.floor(len(history_) / 2))]
#             for j, idx in enumerate(wiz_positions):
#                 log_str += '\tSample: %s\n' % sample_utterances_batch[j][i]
#                 log_str += '\tGreedy: %s\n' % greedy_utterances_batch[j][i]
#                 log_str += '\tSample Coverage Score: %s\tSample Coherence Score: %s\n' % (
#                     round(sample_scores_batch[j][i]['cov'], 3), sample_scores_batch[j][i]['coh'])
#                 log_str += '\tGreedy Coverage Score: %s\tGreedy Coherence Score: %s\n' % (
#                     round(greedy_scores_batch[j][i]['cov'], 3), greedy_scores_batch[j][i]['coh'])
#                 if j != len(app_positions):
#                     log_str += '\tApp Response: %s\n' % history_[idx + 1]
#             self.log(log_str)
#
#     def train_self_play_rl_step(self, topics, documents, histories, reverse, turn):
#         # sampling
#         sample_utterances, RL_log_probs = self.make_utterance(topics, documents, histories, greedy=False)
#         # greedy baseline
#         with torch.autograd.no_grad():
#             greedy_utterances, _ = self.make_utterance(topics, documents, histories, greedy=True)
#         sample_rewards, sample_scores = self.get_reward_score(documents, sample_utterances, histories, reverse=reverse,
#                                                               get_coh=(turn != 0))
#         greedy_rewards, greedy_scores = self.get_reward_score(documents, greedy_utterances, histories, reverse=reverse,
#                                                               get_coh=(turn != 0))
#         rl_loss = torch.sum(-(sample_rewards - greedy_rewards) * RL_log_probs)
#         # Self-critic policy gradient training (eq 15 in https://arxiv.org/pdf/1705.04304.pdf)
#         self.optimizer.zero_grad()
#         # rl_loss.backward()
#         self.accelerator.backward(rl_loss)
#         self.optimizer.step()
#         histories = update_history(histories, greedy_utterances, reverse=reverse)
#         if (turn != self.args.num_turns - 1):
#             # app should not respond when it's the last turn
#             app_say = bart_generate(self.app, histories, num_return_sequences=self.args.num_candicates)
#             histories = update_history(histories, app_say, reverse=reverse)
#         return histories, sample_utterances, greedy_utterances, sample_scores, greedy_scores
#
#     def train_self_play_rl(self, train_dataset, eval_dataset):
#         # Only show the progress bar once on each machine.
#         progress_bar_train = tqdm(range(int(self.args.max_train_steps / self.args.batch_size)))
#         completed_steps = 0
#         eval_epoch = 0
#         self.sel.selector.train()
#         num_steps = math.floor(len(train_dataset) / self.args.batch_size)
#         indexes = list(range(len(train_dataset)))
#         if self.args.shuffle:
#             random.shuffle(indexes)
#         for i in range(num_steps):
#             indexes_batch = indexes[i * self.args.batch_size: (i + 1) * self.args.batch_size]
#             instances = train_dataset[indexes_batch]
#             topics, documents = instances['topic'], instances['document']
#             histories = ['' for _ in range(len(topics))]
#             sample_utterances_batch, greedy_utterances_batch, sample_scores_batch, greedy_scores_batch = [], [], [], []
#             try:
#                 ins_start_time = datetime.now()
#                 self.log('\n### Training step %s ###' % i)
#                 for j in range(self.args.num_turns):
#                     histories, sample_utterances, greedy_utterances, sample_scores, greedy_scores = self.train_self_play_rl_step(
#                         topics, documents, histories, reverse=self.args.reverse, turn=j)
#                     sample_utterances_batch.append(sample_utterances)
#                     greedy_utterances_batch.append(greedy_utterances)
#                     sample_scores_batch.append(sample_scores)
#                     greedy_scores_batch.append(greedy_scores)
#                 ins_end_time = datetime.now()
#                 self.log_step(histories, sample_utterances_batch, greedy_utterances_batch, sample_scores_batch,
#                               greedy_scores_batch)
#                 self.log('Per batch time: %s\tTotal time consumed: %s' % (
#                     str(ins_end_time - ins_start_time), str(ins_end_time - self.start_time)))
#             except:
#                 self.log('Failed training case, idx %s' % i)
#                 self.log('Topic:\t%s' % topics)
#                 self.log('Document:\t%s' % documents)
#             progress_bar_train.update(1)
#             completed_steps += 1
#             if completed_steps % self.args.eval_steps == 0:
#                 self.log('--- Eval epoch %s starts ---' % eval_epoch)
#                 eval_epoch += 1
#                 self.eval_self_play_rl(eval_dataset)
#             if completed_steps % self.args.save_steps == 0:
#                 self.sel.save_model(self.args.output_dir, '/step_%s' % i)
#             if completed_steps >= self.args.max_train_steps:
#                 break
#
#     def eval_self_play_rl(self, eval_dataset):
#         eval_start_time = datetime.now()
#         progress_bar_eval = tqdm(range(int(len(eval_dataset) / self.args.batch_size)))
#         indexes = list(range(len(eval_dataset)))
#         num_steps = math.ceil(len(eval_dataset) / self.args.batch_size)
#         self.sel.selector.eval()
#         scores_cov_all = []
#         scores_coh_all = []
#         for i in range(num_steps):
#             indexes_batch = indexes[i * self.args.batch_size: (i + 1) * self.args.batch_size]
#             instances = eval_dataset[indexes_batch]
#             topics, documents = instances['topic'], instances['document']
#             histories = ['' for _ in range(len(topics))]
#             try:
#                 for j in range(self.args.num_turns):
#                     with torch.autograd.no_grad():
#                         utterances, _ = self.make_utterance(topics, documents, histories, greedy=True)
#                     rewards, scores_batch = self.get_reward_score(documents, utterances, histories,
#                                                                   reverse=self.args.reverse, get_coh=(j != 0))
#                     for scores in scores_batch:
#                         scores_cov_all.append(scores['cov'])
#                         scores_coh_all.append(scores['coh'])
#             except:
#                 self.log('Failed eval case, idx %s' % j)
#                 self.log('Topic:\t%s' % topics)
#                 self.log('Document:\t%s' % documents)
#             progress_bar_eval.update(1)
#         self.log('--- Eval average coverage score: %f ---' % (np.mean(scores_cov_all)))
#         self.log('--- Eval average coherence score: %f ---' % (np.mean(scores_coh_all)))
#         eval_end_time = datetime.now()
#         self.log('--- Eval time consumed: %s ---' % str(eval_end_time - eval_start_time))
#         self.sel.selector.train()

