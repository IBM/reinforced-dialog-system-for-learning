import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoModel
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
    def __init__(self, max_cov_score=1.0):
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        self.max_cov_score = max_cov_score
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    def score_conversation(self, history, document):
        if type(history) is list:
            history_ = ' '.join(history)
        else:
            assert type(history) is str
            history_ = history
        scores = self.scorer.score(document, history_)
        return scores['rouge1'].fmeasure

    def score_utterance(self, histories, documents):
        if type(histories) is str:
            # score one instance
            scores = self.scorer.score(documents, histories)
            return scores['rouge1'].fmeasure
        else:
            # score a batch
            assert type(histories) == list
            assert len(documents) == len(histories)
            scores_batch = []
            for history, document in zip(histories, documents):
                scores_batch.append(self.score_utterance(history, document))
            return scores_batch

    def get_lengthy_penalty_scale(self, response):
        response_length = len(self.tokenizer.tokenize(response))
        if response_length < 10:
            return 0.1 * response_length
        elif response_length <= 50:
            return 1.0
        elif 50 < response_length <= 60:
            return 0.9
        elif 60 < response_length <= 70:
            return 0.7
        elif 70 < response_length <= 80:
            return 0.5
        else:
            return 0.3

    def score_utterance_for_generator(self, responses, histories, documents):
        assert type(histories) == list
        assert len(responses) == len(histories) == len(documents)
        cov_score = []
        for (response, history, document) in zip(responses, histories, documents):
            cov_old = self.scorer.score(document, history)
            cov_new = self.scorer.score(document, history + response)
            score_diff = min(max(cov_new['rouge1'].fmeasure - cov_old['rouge1'].fmeasure, 0), self.max_cov_score)
            cov_score.append(score_diff)
        return np.array(cov_score)

    def score_conversation_for_generator(self, histories, documents):
        cov_scores = []
        uttrs = [history.split(' / ') for history in histories]
        uttrs_wiz = [' / '.join([uttr for j, uttr in enumerate(uttrs[i]) if j % 2 == 0]) for i in range(len(uttrs))]
        for uttr_wiz, document in zip(uttrs_wiz, documents):
            cov_score = self.scorer.score(document, uttr_wiz)
            cov_scores.append(cov_score['rouge1'].fmeasure)
        return cov_scores


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
        self.model.load_state_dict(torch.load(args.model_name_or_path, map_location=device))
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

    def score_conversation(self, history, full_score=0.3):
        idxs_app = [2 * i + 1 for i in range(math.ceil(len(history) / 2 - 1))]
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

    def score_utterance(self, premises, hypothesises, full_score=0.3):
        if type(premises) == str:
            # score one instance
            with torch.no_grad():
                pred = self.predict(premises.lower(), hypothesises.lower())
            if pred == 0:
                coh_score = full_score
            elif pred == 1:
                coh_score = full_score * 0.2
            else:
                assert pred == 2
                coh_score = 0.0
            return coh_score
        else:
            # score a batch
            assert type(premises) == list
            assert type(hypothesises) == list
            coh_scores = []
            for premise, hypothesis in zip(premises, hypothesises, full_score):
                coh_scores.append(self.score_utterance(premise, hypothesis))
            return coh_scores

    def score_utterance_for_generator(self, responses, histories):
        coh_scores = []
        assert len(responses) == len(histories)
        last_utterances = [history.split(' / ')[0] for history in histories]
        for premise, hypothesis in zip(last_utterances, responses):
            coh_scores.append(self.score_utterance(premise, hypothesis, 1.0))
        return np.array(coh_scores)

    def score_conversation_for_generator(self, histories):
        coh_scores = []
        histories_ = [history.split(' / ') for history in histories]
        for history in histories_:
            history.reverse()
            coh_scores_conv = []
            for i in range(int(len(history) / 2)):
                uttr_app = history[i+1]
                uttr_wiz = history[i+2]
                coh_scores_conv.append(self.score_utterance(uttr_app, uttr_wiz, 1.0))
            coh_scores.append(np.mean(coh_scores_conv))
        return coh_scores


class CoherenceScorerWoW:
    def __init__(self, args, device=None):
        base_model = 'bert-base-cased'
        self.args = args
        config = AutoConfig.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=not args.use_slow_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            from_tf=False,
            config=config,
        )
        self.model.load_state_dict(torch.load(args.model_name_or_path, map_location=device))

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
        return torch.softmax(outputs[0], dim=0).numpy()[1]

    # def score_conversation(self, history, full_score=0.3):
    #     idxs_app = [2 * i + 1 for i in range(math.ceil(len(history) / 2 - 1))]
    #     score_per_turn = full_score / len(idxs_app)
    #     coh_score = 0
    #     for idx in idxs_app:
    #         premise = history[idx]
    #         hypothesis = history[idx + 1]
    #         pred = self.predict(premise.lower(), hypothesis.lower())
    #         if pred == 0:
    #             coh_score += score_per_turn
    #         elif pred == 1:
    #             coh_score += score_per_turn * 0.2
    #         else:
    #             assert pred == 2
    #             # do nothing
    #     return coh_score

    def score_utterance(self, premises, hypothesises):
        if type(premises) == str:
            # score one instance
            with torch.no_grad():
                coh_score = self.predict(premises.lower(), hypothesises.lower())
            return coh_score
        else:
            # score a batch
            assert type(premises) == list
            assert type(hypothesises) == list
            coh_scores = []
            for premise, hypothesis in zip(premises, hypothesises):
                coh_scores.append(self.score_utterance(premise, hypothesis))
            return coh_scores

    def score_utterance_for_generator(self, responses, histories):
        coh_scores = []
        assert len(responses) == len(histories)
        last_utterances = [history.split(' / ')[0] for history in histories]
        for premise, hypothesis in zip(last_utterances, responses):
            coh_scores.append(self.score_utterance(premise, hypothesis))
        return np.array(coh_scores)

    def score_conversation_for_generator(self, histories):
        coh_scores = []
        histories_ = [history.split(' / ') for history in histories]
        for history in histories_:
            history.reverse()
            coh_scores_conv = []
            for i in range(int(len(history) / 2)):
                uttr_app = history[i+1]
                uttr_wiz = history[i+2]
                coh_scores_conv.append(self.score_utterance(uttr_app, uttr_wiz))
            coh_scores.append(np.mean(coh_scores_conv))
        return coh_scores



