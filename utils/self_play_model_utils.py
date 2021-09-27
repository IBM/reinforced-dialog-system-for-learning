from tqdm.auto import tqdm, trange
import json
import csv
from tqdm import tqdm, trange
import argparse
import random
import math
import numpy as np
import os
import ast
import copy
import codecs
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertModel,
    BertPreTrainedModel,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    RobertaForMultipleChoice
)
from transformers.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    SequenceClassifierOutput,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_callable,
)
from metrics import evaluate_nq
from bart_decoder import MultiHeadBartForConditionalGeneration
from utils.self_play_infra_utils import DataCollatorForResponseSelectorEval
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from datasets import Dataset
from consts import *


bleu = Bleu(3)


class InputFeaturesMultiBart:
    def __init__(self,
                 example_index,
                 source_ids,
                 source_mask,
                 source_len,
                 target_ids,
                 target_labels,
                 target_len,
                 doc_ids,
                 doc_mask,
                 doc_len):
        self.example_index = example_index

        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_len = source_len

        self.target_ids = target_ids
        self.target_labels = target_labels
        self.target_len = target_len

        self.doc_ids = doc_ids
        self.doc_mask = doc_mask
        self.doc_len = doc_len


class InputFeaturesBart:
    def __init__(self,
                 example_index,
                 source_ids,
                 source_mask,
                 source_len,
                 target_ids,
                 target_labels,
                 target_len):
        self.example_index = example_index

        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_len = source_len

        self.target_ids = target_ids
        self.target_labels = target_labels
        self.target_len = target_len


class GenerationInputExample(object):

    def __init__(self, guid, source, target, context=None):
        self.guid = guid
        self.source = source
        self.target = target
        self.context = context

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BartQA:
    def __init__(self, args, device=None):
        self.args = args
        if device is not None:
            self.device=device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        if os.path.exists(self.args.model_file_path):
            print('Loading exisitng model at ' + str(self.args.model_file_path))
            sys.stdout.flush()
            self.generator = BartForConditionalGeneration.from_pretrained(self.args.model_name, state_dict=
            torch.load(self.args.model_file_path)['model'])
        else:
            print('Loading default pre-trained BART weights')
            sys.stdout.flush()
            self.generator = BartForConditionalGeneration.from_pretrained(self.args.model_name)
        self.generator.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(
            self.args.model_name)  # Need to add base to "tokenization_bart.py" when using transformers==2.11.0

    def save(self, num_updates):
        model_to_save = (
            self.generator.module if hasattr(self.generator, "module") else self.generator
        )
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': self.get_optimizer(),
            'args': self.args
        }
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{num_updates}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(output_dir, 'model.pt'))

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed",
                            default=42,
                            type=int,
                            help="Random seed")
        parser.add_argument("--model_name",
                            default='facebook/bart-large',
                            type=str,
                            help="BART model")
        parser.add_argument('--data_dir',
                            type=str,
                            default='cmu_data/',
                            help='path to data_dir')
        parser.add_argument('--output_dir',
                            type=str,
                            default='trained_models/',
                            help='path to save the model')
        parser.add_argument('--log_file_path',
                            type=str,
                            default='./log.txt',
                            help='Log file')
        parser.add_argument('--model_file_path',
                            type=str,
                            default='./model.bin',
                            help='Model file')
        parser.add_argument("--source_max_len",
                            default=512,
                            type=int,
                            help="Max len of source")
        parser.add_argument("--target_max_len",
                            default=128,
                            type=int,
                            help="Max len of target")
        parser.add_argument("--train_batch_size",
                            default=2,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--validation_timing",
                            default=1000,
                            type=int,
                            help="Check dev score after every N updates")
        parser.add_argument("--eval_batch_size",
                            default=2,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=25.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=8,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False,
                            default=1.0,
                            type=float)
        parser.add_argument("--do_train",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument("--do_eval",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument("--do_generate",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument('--experiment_type',
                            type=str,
                            default='chat_context',
                            help='Type of input to be fed. Options are '
                                 '[chat_only | chat_document | chat_wizard]')

        return parser.parse_args()

    def load_examples(self, data_dir, filename):
        examples = []

        with codecs.open(data_dir + filename, 'r', 'utf-8') as inp:
            spam = csv.reader(inp, delimiter='\t')
            for row in spam:
                guid = row[0]
                source = row[1]
                target = row[2]
                if self.args.experiment_type == 'chat_only':
                    context = ''
                else:
                    context = row[3]
                examples.append(GenerationInputExample(
                    guid=guid,
                    source=source,
                    target=target,
                    context=context
                ))

        return examples

    def convert_examples_to_features(self, examples):
        config = self.generator.model.config
        features = []
        index = 0

        for e in tqdm(examples, desc='Examples'):
            # Process source information
            chat_history = e.source

            if self.args.experiment_type == 'chat_only':
                source = chat_history
            elif self.args.experiment_type == 'chat_document':
                context = e.context
                source = 'chat: ' + chat_history + ' document: ' + context
            elif self.args.experiment_type == 'chat_wizard':
                context = e.context
                context = ast.literal_eval(context)
                all_docs = ''
                for doc in context:
                    title = list(doc.keys())[0]
                    passage = ' '.join(doc[title])
                    all_docs = all_docs + ' title: ' + title + ' text: ' + passage
                source = 'chat: ' + chat_history + ' document: ' + all_docs
            else:
                print('Unrecongnized argument for experiment type')

            ### No SEP token between context and chat_history
            source_tokens = self.tokenizer.tokenize(source)[:self.args.source_max_len - 2]
            if len(source_tokens) == 0:
                print('Empty Source: ', e.source, e.context, e.target)
                continue
            source_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(source_tokens) + [
                config.eos_token_id]  # <s> ... </s>
            source_len = len(source_ids)
            source_mask = [1] * source_len

            padding_len = self.args.source_max_len - source_len
            source_ids += ([config.pad_token_id] * padding_len)
            source_mask += ([0] * padding_len)

            assert len(source_ids) == self.args.source_max_len
            assert len(source_mask) == self.args.source_max_len

            # Process target information
            response = e.target
            answer_tokens = self.tokenizer.tokenize(response)[:self.args.target_max_len - 1]  # -1 for <s> or </s>
            if len(answer_tokens) == 0:
                print('Empty Target: ', e.source, e.context, e.target)
                continue
            target_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(answer_tokens)  # <s> ...
            target_labels = self.tokenizer.convert_tokens_to_ids(answer_tokens) + [config.eos_token_id]  # ... </s>
            target_len = len(target_ids)

            padding_len = self.args.target_max_len - target_len
            target_ids += ([config.pad_token_id] * padding_len)
            target_labels += ([-100] * padding_len)  # -100 is the default index to be ignored

            assert len(target_ids) == self.args.target_max_len
            assert len(target_labels) == self.args.target_max_len

            f = InputFeaturesBart(index, source_ids, source_mask, source_len, target_ids, target_labels, target_len)
            features.append(f)

            index += 1

        return features

    def init_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if torch.cuda.is_available:
            torch.cuda.manual_seed(self.args.seed)

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.generator.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def get_train_dataloader(self, train_features, train_batch_size):
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_source_len = torch.tensor([f.source_len for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in train_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids,
                                   all_source_mask,
                                   all_source_len,
                                   all_target_ids,
                                   all_target_labels,
                                   all_target_len)
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    def get_eval_dataloader(self, dev_features, dev_batch_size):
        all_example_indices = torch.tensor([f.example_index for f in dev_features], dtype=torch.long)
        all_source_ids = torch.tensor([f.source_ids for f in dev_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in dev_features], dtype=torch.long)
        all_source_len = torch.tensor([f.source_len for f in dev_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in dev_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in dev_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in dev_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_example_indices,
            all_source_ids,
            all_source_mask,
            all_source_len,
            all_target_ids,
            all_target_labels,
            all_target_len
        )
        eval_sampler = SequentialSampler(eval_data)
        return DataLoader(eval_data, sampler=eval_sampler, batch_size=dev_batch_size)

    def get_train_batch_data(self,
                             batch):

        batch_source_max_len = batch[2].max().item()
        batch_target_max_len = batch[5].max().item()
        batch_total_tokens = batch[5].sum().item()

        batch = tuple(t.to(self.device) for t in batch)
        source_ids, source_mask, _, target_ids, target_labels, __ = batch
        source_ids = source_ids[:, :batch_source_max_len]
        source_mask = source_mask[:, :batch_source_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()

        return source_ids, source_mask, target_ids, target_labels

    def get_eval_batch_data(self, batch):

        example_indices = batch[0].tolist()
        batch_source_max_len = batch[3].max().item()
        batch_target_max_len = batch[6].max().item()
        batch_total_tokens = batch[6].sum().item()

        batch = tuple(t.to(self.device) for t in batch)
        _, source_ids, source_mask, __, target_ids, target_labels, _ = batch
        source_ids = source_ids[:, :batch_source_max_len]
        source_mask = source_mask[:, :batch_source_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()

        return example_indices, source_ids, source_mask, target_ids, target_labels, batch_total_tokens

    def train(self):

        self.init_seed()

        cached_features_devfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_task_{}_dev_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )
        dev_examples = self.load_examples(self.args.data_dir, 'dev.tsv')
        if os.path.exists(cached_features_devfile):
            dev_features = torch.load(cached_features_devfile)
        else:
            dev_features = self.convert_examples_to_features(dev_examples)
            torch.save(dev_features, cached_features_devfile)
        dev_data = (dev_examples, dev_features)

        cached_features_trainfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_task_{}_train_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )

        if os.path.exists(cached_features_trainfile):
            train_features = torch.load(cached_features_trainfile)
        else:
            train_examples = self.load_examples(self.args.data_dir, 'train.tsv')
            train_features = self.convert_examples_to_features(train_examples)
            torch.save(train_features, cached_features_trainfile)

        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)
        num_train_steps = int(
            len(train_features) / train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        optimizer = self.get_optimizer()
        t_total = num_train_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * self.args.warmup_proportion),
                                                    num_training_steps=t_total)

        # logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_features))
        # logger.info("  Batch size = %d", train_batch_size)
        # logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = self.get_train_dataloader(train_features, train_batch_size)

        self.generator.zero_grad()
        self.generator.train()

        num_updates = 0

        if self.args.log_file_path is not None:
            f_log = open(self.args.log_file_path, 'w')
        else:
            f_log = None

        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                source_ids, source_mask, target_ids, target_labels = self.get_train_batch_data(batch)

                outputs = self.generator(input_ids=source_ids,
                                         attention_mask=source_mask,
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels)

                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.generator.zero_grad()
                    num_updates += 1

                    if num_updates % self.args.validation_timing == 0:
                        results = self.evaluate(dev_data)

                        results["steps"] = step
                        results["num_updates"] = num_updates
                        if f_log is not None:
                            f_log.write(str(results))
                            f_log.write('\n')
                            f_log.flush()

                        self.save(num_updates)

        if f_log is not None:
            f_log.close()

    def predict(self,
                dev_data):

        dev_examples, dev_features = dev_data
        eval_dataloader = self.get_eval_dataloader(dev_features, self.args.eval_batch_size)

        self.generator.eval()

        pred = [None] * len(dev_examples)
        total_eval_loss, total_words = 0, 0

        for batch in tqdm(eval_dataloader, desc="Generating"):
            example_indices, source_ids, source_mask, target_ids, \
            target_labels, batch_total_tokens = self.get_eval_batch_data(batch)

            with torch.no_grad():
                outputs = self.generator(input_ids=source_ids,
                                         attention_mask=source_mask,
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels)

                loss = outputs[0]
                total_eval_loss += (loss.item() * batch_total_tokens)
                total_words += batch_total_tokens
                predicted_ids = self.generator.generate(input_ids=source_ids, attention_mask=source_mask,
                                                        num_beams=1,
                                                        max_length=self.args.target_max_len,
                                                        early_stopping=True)

            predicted_ids = predicted_ids.to(self.cpu)
            for i in range(len(example_indices)):
                if pred[example_indices[i]] is not None:
                    continue
                answer = self.tokenizer.decode(
                    predicted_ids[i].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                pred[example_indices[i]] = answer

        self.generator.train()
        return pred, total_eval_loss, total_words

    def evaluate(self, dev_data=None, save_file=False):

        if dev_data is None:
            cached_features_devfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_task_{}_dev_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
            )
            dev_examples = self.load_examples(self.args.data_dir, 'dev.tsv')
            if os.path.exists(cached_features_devfile):
                dev_features = torch.load(cached_features_devfile)
            else:
                dev_features = self.convert_examples_to_features(dev_examples)
                torch.save(dev_features, cached_features_devfile)
        else:
            dev_examples, dev_features = dev_data

        pred, total_eval_loss, total_words = self.predict((dev_examples, dev_features))
        results = evaluate_nq(dev_examples, pred, total_eval_loss, total_words)
        if save_file:
            with codecs.open(self.args.output_dir + 'dev_predictions.txt', 'w', 'utf-8') as out:
                for p in pred:
                    p = self.clean_text(p)
                    out.write(p + '\n')

            with codecs.open(self.args.output_dir + 'dev_reference.txt', 'w', 'utf-8') as out:
                for example in dev_examples:
                    target = self.clean_text(example.target)
                    out.write(target + '\n')
        return results

    def clean_text(self, text):
        text = ' '.join(text.split('\n'))
        text = ' '.join(text.split('\t'))
        text = ' '.join(text.split())
        return text

    def generate(self):

        if self.args.experiment_type == 'chat_wizard':
            self.generate_wizard()
        else:

            cached_features_testfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_task_{}_test_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
            )

            test_examples = self.load_examples(self.args.data_dir, 'test.tsv')
            if os.path.exists(cached_features_testfile):
                test_features = torch.load(cached_features_testfile)
            else:
                test_features = self.convert_examples_to_features(test_examples)
                torch.save(test_features, cached_features_testfile)

            pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
            with codecs.open(self.args.output_dir + 'predictions.txt', 'w', 'utf-8') as out:
                for p in pred:
                    p = self.clean_text(p)
                    out.write(p + '\n')

            with codecs.open(self.args.output_dir + 'reference.txt', 'w', 'utf-8') as out:
                for example in test_examples:
                    target = self.clean_text(example.target)
                    out.write(target + '\n')

            with codecs.open(self.args.output_dir + 'all_results.csv', 'w', 'utf-8') as out:
                writer_ = csv.writer(out, delimiter=',')
                for i in range(len(pred)):
                    writer_.writerow([i, test_examples[i].target, pred[i]])

            results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
            print(str(results))

    def generate_wizard(self):

        cached_features_testfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_task_{}_test_seen_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )

        test_examples = self.load_examples(self.args.data_dir, 'test_seen.tsv')
        if os.path.exists(cached_features_testfile):
            test_features = torch.load(cached_features_testfile)
        else:
            test_features = self.convert_examples_to_features(test_examples)
            torch.save(test_features, cached_features_testfile)

        pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
        with codecs.open(self.args.output_dir + 'predictions_seen.txt', 'w', 'utf-8') as out:
            for p in pred:
                p = self.clean_text(p)
                out.write(p + '\n')

        with codecs.open(self.args.output_dir + 'reference_seen.txt', 'w', 'utf-8') as out:
            for example in test_examples:
                target = self.clean_text(example.target)
                out.write(target + '\n')

        with codecs.open(self.args.output_dir + 'all_results_seen.csv', 'w', 'utf-8') as out:
            writer_ = csv.writer(out, delimiter=',')
            for i in range(len(pred)):
                writer_.writerow([i, test_examples[i].target, pred[i]])

        results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
        print(str(results))

        cached_features_testfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_task_{}_test_unseen_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )

        test_examples = self.load_examples(self.args.data_dir, 'test_unseen.tsv')
        if os.path.exists(cached_features_testfile):
            test_features = torch.load(cached_features_testfile)
        else:
            test_features = self.convert_examples_to_features(test_examples)
            torch.save(test_features, cached_features_testfile)

        pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
        with codecs.open(self.args.output_dir + 'predictions_unseen.txt', 'w', 'utf-8') as out:
            for p in pred:
                p = self.clean_text(p)
                out.write(p + '\n')

        with codecs.open(self.args.output_dir + 'reference_unseen.txt', 'w', 'utf-8') as out:
            for example in test_examples:
                target = self.clean_text(example.target)
                out.write(target + '\n')

        with codecs.open(self.args.output_dir + 'all_results_unseen.csv', 'w', 'utf-8') as out:
            writer_ = csv.writer(out, delimiter=',')
            for i in range(len(pred)):
                writer_.writerow([i, test_examples[i].target, pred[i]])

        results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
        print(str(results))


class MultiBartQA:
    def __init__(self, args, device):
        # self.args = self.parse_args()
        self.args = args
        if device is not None:
            self.device=device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.generator = MultiHeadBartForConditionalGeneration.from_pretrained_multi(self.args,
                                                                                     self.args.model_file_path)
        self.generator.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(
            self.args.model_name)  # Need to add base to "tokenization_bart.py" when using transformers==2.11.0

    def save(self, num_updates):
        model_to_save = (
            self.generator.module if hasattr(self.generator, "module") else self.generator
        )
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': self.get_optimizer(),
            'args': self.args
        }
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{num_updates}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(output_dir, 'model.pt'))

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed",
                            default=42,
                            type=int,
                            help="Random seed")
        parser.add_argument("--model_name",
                            default='facebook/bart-large',
                            type=str,
                            help="BART model")
        parser.add_argument('--data_dir',
                            type=str,
                            default='cmu_data/',
                            help='path to data_dir')
        parser.add_argument('--output_dir',
                            type=str,
                            default='trained_models/',
                            help='path to save the model')
        parser.add_argument('--log_file_path',
                            type=str,
                            default='./log.txt',
                            help='Log file')
        parser.add_argument('--model_file_path',
                            type=str,
                            default='./pytorch_model.bin',
                            help='Model file')
        parser.add_argument("--source_max_len",
                            default=512,
                            type=int,
                            help="Max len of source")
        parser.add_argument("--target_max_len",
                            default=128,
                            type=int,
                            help="Max len of target")
        parser.add_argument("--train_batch_size",
                            default=2,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--validation_timing",
                            default=1000,
                            type=int,
                            help="Check dev score after every N updates")
        parser.add_argument("--eval_batch_size",
                            default=16,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=25.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=8,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False,
                            default=1.0,
                            type=float)
        parser.add_argument("--do_train",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument("--do_eval",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument("--do_generate",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument('--experiment_type',
                            type=str,
                            default='chat_context',
                            help='Type of input to be fed. Options are '
                                 '[doc_only | chat_document | chat_wizard]')

        return parser.parse_args()

    def load_examples(self, data_dir, filename):
        examples = []

        with codecs.open(data_dir + filename, 'r', 'utf-8') as inp:
            spam = csv.reader(inp, delimiter='\t')
            for row in spam:
                guid = row[0]
                source = row[1]
                target = row[2]
                context = row[3]
                examples.append(GenerationInputExample(
                    guid=guid,
                    source=source,
                    target=target,
                    context=context
                ))

        return examples

    def convert_examples_to_features(self, examples):
        config = self.generator.model.config
        features = []
        index = 0

        for e in tqdm(examples, desc='Examples'):
            # Process source information

            source = 'chat: ' + e.source

            source_tokens = self.tokenizer.tokenize(source)[:self.args.source_max_len - 2]
            source_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(source_tokens) + [
                config.eos_token_id]  # <s> ... </s>
            source_len = len(source_ids)
            source_mask = [1] * source_len

            padding_len = self.args.source_max_len - source_len
            source_ids += ([config.pad_token_id] * padding_len)
            source_mask += ([0] * padding_len)

            assert len(source_ids) == self.args.source_max_len
            assert len(source_mask) == self.args.source_max_len

            if self.args.experiment_type == 'doc_only':
                document = 'document: ' + e.context
            elif self.args.experiment_type == 'chat_document':
                document = 'chat: ' + e.source + ' document: ' + e.context
            elif self.args.experiment_type == 'chat_wizard':
                context = e.context
                context = ast.literal_eval(context)
                all_docs = ''
                for doc in context:
                    title = list(doc.keys())[0]
                    passage = ' '.join(doc[title])
                    all_docs = all_docs + ' title: ' + title + ' text: ' + passage
                document = 'chat: ' + e.source + ' document: ' + all_docs
            else:
                print('Unrecongnized argument for experiment type')

            doc_tokens = self.tokenizer.tokenize(document)[:self.args.source_max_len - 2]
            doc_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(doc_tokens) + [
                config.eos_token_id]  # <s> ... </s>
            doc_len = len(doc_ids)
            doc_mask = [1] * doc_len

            padding_len = self.args.source_max_len - doc_len
            doc_ids += ([config.pad_token_id] * padding_len)
            doc_mask += ([0] * padding_len)

            assert len(doc_ids) == self.args.source_max_len
            assert len(doc_mask) == self.args.source_max_len

            # Process target information

            answer = e.target
            answer_tokens = self.tokenizer.tokenize(answer)[:self.args.target_max_len - 1]  # -1 for <s> or </s>
            if len(answer_tokens) == 0:
                print(e.source, e.context, e.target)
                continue
            target_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(answer_tokens)  # <s> ...
            target_labels = self.tokenizer.convert_tokens_to_ids(answer_tokens) + [config.eos_token_id]  # ... </s>
            target_len = len(target_ids)

            padding_len = self.args.target_max_len - target_len
            target_ids += ([config.pad_token_id] * padding_len)
            target_labels += ([-100] * padding_len)  # -100 is the default index to be ignored

            assert len(target_ids) == self.args.target_max_len
            assert len(target_labels) == self.args.target_max_len

            f = InputFeaturesMultiBart(
                index,
                source_ids,
                source_mask,
                source_len,
                target_ids,
                target_labels,
                target_len,
                doc_ids,
                doc_mask,
                doc_len
            )
            features.append(f)

            index += 1

        return features

    def init_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if torch.cuda.is_available:
            torch.cuda.manual_seed(self.args.seed)

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.generator.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def get_train_dataloader(self,
                             train_features,
                             train_batch_size):
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_source_len = torch.tensor([f.source_len for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in train_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in train_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.doc_ids for f in train_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in train_features], dtype=torch.long)
        all_doc_len = torch.tensor([f.doc_len for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_source_ids,
            all_source_mask,
            all_source_len,
            all_target_ids,
            all_target_labels,
            all_target_len,
            all_doc_ids,
            all_doc_mask,
            all_doc_len
        )
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    def get_eval_dataloader(self, dev_features, dev_batch_size):
        all_example_indices = torch.tensor([f.example_index for f in dev_features], dtype=torch.long)
        all_source_ids = torch.tensor([f.source_ids for f in dev_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in dev_features], dtype=torch.long)
        all_source_len = torch.tensor([f.source_len for f in dev_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in dev_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in dev_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in dev_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.doc_ids for f in dev_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in dev_features], dtype=torch.long)
        all_doc_len = torch.tensor([f.doc_len for f in dev_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_example_indices,
            all_source_ids,
            all_source_mask,
            all_source_len,
            all_target_ids,
            all_target_labels,
            all_target_len,
            all_doc_ids,
            all_doc_mask,
            all_doc_len
        )
        eval_sampler = SequentialSampler(eval_data)
        return DataLoader(eval_data, sampler=eval_sampler, batch_size=dev_batch_size)

    def get_train_batch_data(self, batch):

        batch_source_max_len = batch[2].max().item()
        batch_target_max_len = batch[5].max().item()
        batch_doc_max_len = batch[8].max().item()
        batch_total_tokens = batch[5].sum().item()

        batch = tuple(t.to(self.device) for t in batch)
        source_ids, source_mask, _, target_ids, target_labels, _, doc_ids, doc_mask, _ = batch
        source_ids = source_ids[:, :batch_source_max_len]
        source_mask = source_mask[:, :batch_source_max_len]
        doc_ids = doc_ids[:, :batch_doc_max_len]
        doc_mask = doc_mask[:, :batch_doc_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()

        return source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens

    def get_eval_batch_data(self, batch):

        example_indices = batch[0].tolist()
        batch_source_max_len = batch[3].max().item()
        batch_target_max_len = batch[6].max().item()
        batch_total_tokens = batch[6].sum().item()
        batch_doc_max_len = batch[9].max().item()

        batch = tuple(t.to(self.device) for t in batch)
        _, source_ids, source_mask, __, target_ids, target_labels, _, doc_ids, doc_mask, _ = batch
        source_ids = source_ids[:, :batch_source_max_len]
        source_mask = source_mask[:, :batch_source_max_len]
        doc_ids = doc_ids[:, :batch_doc_max_len]
        doc_mask = doc_mask[:, :batch_doc_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()

        return example_indices, source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens

    def encode(self, source_ids, source_mask, doc_ids, doc_mask):

        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D) --> (B, N, L, D)
        # [(B, L1), (B, L2)] --> [(B, L1, D), (B, L2, D)]
        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D) --> Aggregate[(B, N*L, V) + (B, L, V)] --> (B, L, V)
        # (B, N, L) -> (B*N, L) -> (B, N*L)

        source_reps = self.generator.model.encoder(
            input_ids=source_ids,
            attention_mask=source_mask
        )
        source_reps = source_reps[0]

        doc_reps = self.generator.model.encoder(
            input_ids=doc_ids,
            attention_mask=doc_mask
        )
        doc_reps = doc_reps[0]

        return source_reps, doc_reps

    def train(self):

        self.init_seed()

        cached_features_devfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_DoHA_task_{}_dev_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )
        dev_examples = self.load_examples(self.args.data_dir, 'dev.tsv')
        if os.path.exists(cached_features_devfile):
            dev_features = torch.load(cached_features_devfile)
        else:
            dev_features = self.convert_examples_to_features(dev_examples)
            torch.save(dev_features, cached_features_devfile)
        dev_data = (dev_examples, dev_features)

        cached_features_trainfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_DoHA_task_{}_train_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )
        if os.path.exists(cached_features_trainfile):
            train_features = torch.load(cached_features_trainfile)
        else:
            train_examples = self.load_examples(self.args.data_dir, 'train.tsv')
            train_features = self.convert_examples_to_features(train_examples)
            torch.save(train_features, cached_features_trainfile)

        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)
        num_train_steps = int(
            len(train_features) / train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        optimizer = self.get_optimizer()
        t_total = num_train_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * self.args.warmup_proportion),
                                                    num_training_steps=t_total)

        # logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_features))
        # logger.info("  Batch size = %d", train_batch_size)
        # logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = self.get_train_dataloader(train_features, train_batch_size)

        self.generator.zero_grad()
        self.generator.train()

        num_updates = 0
        curr_loss, curr_total_words = 0, 0

        if self.args.log_file_path is not None:
            f_log = open(self.args.log_file_path, 'w')
        else:
            f_log = None

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens = self.get_train_batch_data(
                    batch)
                source_reps, doc_reps = self.encode(source_ids, source_mask, doc_ids, doc_mask)
                outputs = self.generator(input_ids=None,
                                         attention_mask=(source_mask, doc_mask),
                                         encoder_outputs=(source_reps, doc_reps),
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels,
                                         labels=target_labels)

                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                curr_loss += (loss.item() * batch_total_tokens)
                curr_total_words += batch_total_tokens

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.generator.zero_grad()
                    num_updates += 1

                    if (num_updates + 1) % 10 == 0:
                        train_stat_curr = {
                            'step': step,
                            'num_updates': num_updates,
                            'epoch': epoch,
                            'loss': loss.item(),
                            'train_ppl': math.exp(min(curr_loss / curr_total_words, 100))
                        }
                        print(str(train_stat_curr))
                        sys.stdout.flush()
                        curr_loss, curr_total_words = 0, 0

                    if num_updates % self.args.validation_timing == 0:
                        results = self.evaluate(dev_data)

                        results["steps"] = step
                        results["num_updates"] = num_updates
                        if f_log is not None:
                            f_log.write(str(results))
                            f_log.write('\n')
                            f_log.flush()
                        self.save(num_updates)

        if f_log is not None:
            f_log.close()

    def predict(self, dev_data):

        dev_examples, dev_features = dev_data
        eval_dataloader = self.get_eval_dataloader(dev_features, self.args.eval_batch_size)

        self.generator.eval()

        pred = [None] * len(dev_examples)
        total_eval_loss, total_words = 0, 0

        for batch in tqdm(eval_dataloader, desc="Generating"):
            example_indices, source_ids, source_mask, target_ids, \
            target_labels, doc_ids, doc_mask, batch_total_tokens = self.get_eval_batch_data(batch)
            with torch.no_grad():
                source_reps, doc_reps = self.encode(source_ids, source_mask, doc_ids, doc_mask)
                outputs = self.generator(input_ids=None,
                                         attention_mask=(source_mask, doc_mask),
                                         encoder_outputs=(source_reps, doc_reps),
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels,
                                         labels=target_labels)
                loss = outputs[0]
                total_eval_loss += (loss.item() * batch_total_tokens)
                total_words += batch_total_tokens
                predicted_ids = self.generator.generate(
                    input_ids=source_mask,
                    attention_mask=(source_mask, doc_mask),
                    encoder_outputs=(source_reps, doc_reps),
                    num_beams=1,
                    max_length=self.args.target_max_len,
                    early_stopping=True,
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=0.9,
                )

            predicted_ids = predicted_ids.to(self.cpu)
            for i in range(len(example_indices)):
                if pred[example_indices[i]] is not None:
                    continue
                answer = self.tokenizer.decode(
                    predicted_ids[i].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                pred[example_indices[i]] = answer

        self.generator.train()
        return pred, total_eval_loss, total_words

    def evaluate(self, dev_data=None, save_file=False):
        if dev_data is None:
            cached_features_devfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DoHA_task_{}_dev_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
            )
            dev_examples = self.load_examples(self.args.data_dir, 'dev.tsv')
            if os.path.exists(cached_features_devfile):
                dev_features = torch.load(cached_features_devfile)
            else:
                dev_features = self.convert_examples_to_features(dev_examples)
                torch.save(dev_features, cached_features_devfile)
        else:
            dev_examples, dev_features = dev_data

        pred, total_eval_loss, total_words = self.predict((dev_examples, dev_features))
        results = evaluate_nq(dev_examples, pred, total_eval_loss, total_words)
        if save_file:
            with codecs.open(self.args.output_dir + 'dev_predictions.txt', 'w', 'utf-8') as out:
                for p in pred:
                    p = self.clean_text(p)
                    out.write(p + '\n')

            with codecs.open(self.args.output_dir + 'dev_reference.txt', 'w', 'utf-8') as out:
                for example in dev_examples:
                    target = self.clean_text(example.target)
                    out.write(target + '\n')

        return results

    def clean_text(self, text):
        text = ' '.join(text.split('\n'))
        text = ' '.join(text.split('\t'))
        text = ' '.join(text.split())
        return text

    def generate(self):

        if self.args.experiment_type == 'chat_wizard':
            self.generate_wizard()
        else:

            cached_features_testfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DoHA_task_{}_test_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
            )
            test_examples = self.load_examples(self.args.data_dir, 'test.tsv')
            if os.path.exists(cached_features_testfile):
                test_features = torch.load(cached_features_testfile)
            else:
                test_features = self.convert_examples_to_features(test_examples)
                torch.save(test_features, cached_features_testfile)

            pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
            with codecs.open(self.args.output_dir + 'predictions.txt', 'w', 'utf-8') as out:
                for p in pred:
                    p = self.clean_text(p)
                    out.write(p + '\n')

            with codecs.open(self.args.output_dir + 'reference.txt', 'w', 'utf-8') as out:
                for example in test_examples:
                    target = self.clean_text(example.target)
                    out.write(target + '\n')
            results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
            print(str(results))

    def generate_wizard(self):

        cached_features_testfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_DoHA_task_{}_test_seen_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )

        test_examples = self.load_examples(self.args.data_dir, 'test_seen.tsv')
        if os.path.exists(cached_features_testfile):
            test_features = torch.load(cached_features_testfile)
        else:
            test_features = self.convert_examples_to_features(test_examples)
            torch.save(test_features, cached_features_testfile)

        pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
        with codecs.open(self.args.output_dir + 'predictions_seen.txt', 'w', 'utf-8') as out:
            for p in pred:
                p = self.clean_text(p)
                out.write(p + '\n')

        with codecs.open(self.args.output_dir + 'reference_seen.txt', 'w', 'utf-8') as out:
            for example in test_examples:
                target = self.clean_text(example.target)
                out.write(target + '\n')

        with codecs.open(self.args.output_dir + 'all_results_seen.csv', 'w', 'utf-8') as out:
            writer_ = csv.writer(out, delimiter=',')
            for i in range(len(pred)):
                writer_.writerow([i, test_examples[i].target, pred[i]])

        results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
        print(str(results))

        cached_features_testfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_DoHA_task_{}_test_unseen_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )

        test_examples = self.load_examples(self.args.data_dir, 'test_unseen.tsv')
        if os.path.exists(cached_features_testfile):
            test_features = torch.load(cached_features_testfile)
        else:
            test_features = self.convert_examples_to_features(test_examples)
            torch.save(test_features, cached_features_testfile)

        pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
        with codecs.open(self.args.output_dir + 'predictions_unseen.txt', 'w', 'utf-8') as out:
            for p in pred:
                p = self.clean_text(p)
                out.write(p + '\n')

        with codecs.open(self.args.output_dir + 'reference_unseen.txt', 'w', 'utf-8') as out:
            for example in test_examples:
                target = self.clean_text(example.target)
                out.write(target + '\n')

        with codecs.open(self.args.output_dir + 'all_results_unseen.csv', 'w', 'utf-8') as out:
            writer_ = csv.writer(out, delimiter=',')
            for i in range(len(pred)):
                writer_.writerow([i, test_examples[i].target, pred[i]])

        results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
        print(str(results))


class Selector:
    def __init__(self, args, device=None):
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Loading exisitng model at ' + str(self.args.model_name_or_path))
        sys.stdout.flush()
        self.selector = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            config=self.config,
        )
        sys.stdout.flush()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=not args.use_slow_tokenizer)
        # num_added_toks = self.tokenizer.add_tokens(DOC_TOKEN)
        # print('We have added', num_added_toks, 'tokens')
        self.selector.resize_token_embeddings(len(self.tokenizer))
        self.selector.to(self.device)

    def select(self, sents_batch, histories_batch, documents_batch=None):
        # sents could either be utterance (post-sel) or reference (pre-sel)
        if type(histories_batch) == str:
            history = histories_batch.replace('</s>', '\\')
            if documents_batch:
                context = history + '</s>' + documents_batch.replace('\\', '')
            else:
                context = history
            inputs = self.tokenizer(
                sents_batch,
                [context] * len(sents_batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.args.max_length
            ).to(self.device)
            for key in inputs:
                inputs[key] = torch.unsqueeze(inputs[key], dim=0)
            outputs = self.selector(**inputs)
            probs = softmax(outputs[0], dim=1)
            return probs
        else:
            assert type(histories_batch) == list
            histories_batch_ = [history.replace('</s>', '\\') for history in histories_batch]
            if documents_batch:
                assert len(histories_batch_) == len(documents_batch)
                contexts_batch = [histories_batch_[k] + '</s>' + documents_batch[k].replace('\\', '') for k in range(len(histories_batch_))]
            else:
                contexts_batch = histories_batch_
            indexes_references = []
            sents_batch_ = []
            contexts_batch_ = []
            index_now = 0
            # squeeze the references (batch_size, num_reference) into references_ (batch_size * num_reference)
            for context, sents in zip(contexts_batch, sents_batch):
                sents_batch_ += sents
                contexts_batch_ += [context] * len(sents)
                indexes_references.append((index_now, index_now + len(sents)))
                index_now += len(sents)
            inputs = self.tokenizer(
                sents_batch_,
                contexts_batch_,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.args.max_length
            ).to(self.device)
            for key in inputs:
                inputs[key] = torch.unsqueeze(inputs[key], dim=0)
            outputs = self.selector(**inputs)
            # unsqueeze the output
            probs_batch = torch.zeros((len(sents_batch), max(len(sents) for sents in sents_batch)))
            for k, (index_start, index_end) in enumerate(indexes_references):
                output = outputs[0][0][index_start: index_end]
                # probs_batch.append(softmax(output, dim=0))
                probs_batch[k][:output.shape[0]] = softmax(output)
            return probs_batch

    def save_model(self, output_dir, save_name):
        save_dir = output_dir + save_name + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.selector.save_pretrained(save_dir)


class BertForSequenceClassificationWeighted(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class="BertTokenizer",
        checkpoint="bert-base-uncased",
        output_type=SequenceClassifierOutput,
        config_class="BertConfig",
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        weights=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if weights is None:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss(reduction='none')
                    loss_ = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    loss = torch.dot(loss_, weights) / len(loss_)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def decode_output(predicted_ids, robot):
    if predicted_ids.shape[0] == 1:
        say = robot.tokenizer.decode(predicted_ids.cpu().numpy()[0])
        say = say.replace('</s>', '').replace('<s>', '').replace('<pad>', '')
        return say
    else:
        says = []
        for cand in predicted_ids:
            say = robot.tokenizer.decode(cand.cpu().numpy())
            says.append(say.replace('</s>', '').replace('<s>', '').replace('<pad>', ''))
        return says


def select_response(says, history, thresh=0.6):
    cand2score = {}
    history_ = history.split('</s>')
    previous_says = [rec for i, rec in enumerate(history_) if i % 2 == 1]
    if len(previous_says) == 0:
        return random.choice(says)
    candicate_says = [cand for cand in says]
    for i, cand in enumerate(candicate_says):
        refs = {i: previous_says}
        hyps = {i: [cand]}
        (b1, b2, b3), _ = bleu.compute_score(refs, hyps)
        cand2score[cand] = b2
    try:
        selection_pool = [cand for cand, score in cand2score.items() if score < thresh]
        assert len(selection_pool) > 0
    except:
        cand_ranked = sorted(cand2score.items(), key=lambda kv: kv[1])
        selection_pool = [cand_ranked[0][0]]
        print('Overlapping warning')
    return random.choice(selection_pool)


def doha_generate(robot, topic, history, doc, num_return_sequences):
    assert type(robot) == MultiBartQA
    if type(history) == str:
        # history_ = '</s'.join(history.split('</s>'))
        source_str = '%s%s%s' % (topic, robot.tokenizer.sep_token, history)
        source_ids = robot.tokenizer.encode(source_str, max_length=1024, truncation=True, return_tensors='pt').to(
            robot.device)
        doc_str = source_str + ' </s> ' + doc
        doc_ids = robot.tokenizer.encode(doc_str, max_length=1024, truncation=True, return_tensors='pt').to(
            robot.device)
        source_mask = torch.ones_like(source_ids).to(robot.device)
        doc_mask = torch.ones_like(doc_ids).to(robot.device)
        with torch.no_grad():
            source_reps, doc_reps = robot.encode(source_ids, source_mask, doc_ids, doc_mask)
            predicted_ids = robot.generator.generate(
                input_ids=source_mask,
                attention_mask=(source_mask, doc_mask),
                encoder_outputs=(source_reps, doc_reps),
                num_beams=1,
                max_length=robot.args.target_max_len,
                early_stopping=True,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=0.9,
                num_return_sequences=num_return_sequences
            )
        says = decode_output(predicted_ids, robot)
        if type(says) is str:
            says = [says]
        else:
            assert type(says) is list
        return says
    else:
        assert type(history) == list and type(history[0]) == str
        source_strs = ['%s%s%s' % (rec_topic, robot.tokenizer.sep_token, rec_history_) for (rec_topic, rec_history_) in
                       zip(topic, history)]
        source = robot.tokenizer(source_strs, max_length=1024, padding=True, truncation=True, return_tensors='pt').to(
            robot.device)
        doc_str = [source_str + '</s>' + rec_doc for (source_str, rec_doc) in zip(source_strs, doc)]
        doc = robot.tokenizer(doc_str, max_length=1024, padding=True, truncation=True, return_tensors='pt').to(
            robot.device)
        # source_mask = torch.ones_like(source_ids).to(robot.device)
        # doc_mask = torch.ones_like(doc_ids).to(robot.device)
        with torch.no_grad():
            source_reps, doc_reps = robot.encode(source['input_ids'], source['attention_mask'], doc['input_ids'],
                                                 doc['attention_mask'])
            predicted_ids = robot.generator.generate(
                input_ids=source['attention_mask'],
                attention_mask=(source['attention_mask'], doc['attention_mask']),
                encoder_outputs=(source_reps, doc_reps),
                num_beams=1,
                max_length=robot.args.target_max_len,
                early_stopping=True,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=0.9,
                num_return_sequences=num_return_sequences
            )
        says = decode_output(predicted_ids, robot)
        if type(says) is str:
            says = [says]
        else:
            assert type(says) is list
        return says


def bart_generate(robot, history, num_return_sequences=16):
    assert type(robot) == BartQA
    # source_str = history
    inputs = robot.tokenizer(history, max_length=1024, truncation=True, padding=True, return_tensors='pt').to(
        robot.device)
    # source_mask = torch.ones_like(source_ids).to(robot.device)
    with torch.no_grad():
        predicted_ids = robot.generator.generate(input_ids=inputs['input_ids'],
                                                 attention_mask=inputs['attention_mask'],
                                                 num_beams=3,
                                                 do_sample=True,
                                                 max_length=robot.args.target_max_len,
                                                 early_stopping=True,
                                                 top_p=0.9,
                                                 top_k=15,
                                                 num_return_sequences=num_return_sequences
                                                 )
    says = decode_output(predicted_ids, robot)
    if type(history) == str:
        if type(says) is str:
            say = says
        else:
            assert type(says) is list
            say = select_response(says, history)
        return say
    else:
        assert type(history) == list and type(history[0]) == str
        says_groups = [says[i*num_return_sequences: (i+1)*num_return_sequences] for i in range(len(history))]
        says_selected = []
        for (says_group, rec_history) in zip(says_groups, history):
            say = select_response(says_group, rec_history)
            says_selected.append(say)
        return says_selected


def update_history(history, say, charactor=None, reverse=False):
    if type(history) == str:
        if charactor:
            return history + charactor + ': ' + say
        else:
            if history == '':
                return say
            else:
                if reverse:
                    return say + '</s>' + history
                else:
                    return history + '</s>' + say
    else:
        assert type(history) == list and type(history[0]) == str and type(say) == list
        history_ = []
        for rec_history, rec_say in zip(history, say):
            history_.append(update_history(rec_history, rec_say, charactor, reverse))
        return history_


def parse_conversation_history(history, reverse):
    spans = history.split('</s>')
    if reverse:
        spans.reverse()
    return spans


def make_conversation_no_sel(wiz, app, topic, doc, num_turns, reverse=True):
    history = ""
    for i in range(num_turns):
        wiz_says = doha_generate(wiz, topic, history, doc, num_return_sequences=3)
        wiz_say = random.choice(wiz_says)
        history = update_history(history, wiz_say, reverse=reverse)
        app_say = bart_generate(app, history)
        history = update_history(history, app_say, reverse=reverse)
    return parse_conversation_history(history, reverse=reverse)


class DataloaderRL:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.num_steps = math.floor(len(dataset)/batch_size)
        self.indexes = list(range(len(dataset)))
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.indexes)
        self.step = 0
        self.batch_size = batch_size
        self.dataset = dataset

    def get_next_batch(self):
        start_idx = self.step * self.batch_size
        end_idx = (self.step + 1) * self.batch_size
        if end_idx > len(self.indexes):
            self.step = 0
            if self.shuffle:
                random.shuffle(self.indexes)
            start_idx = self.step * self.batch_size
            end_idx = (self.step + 1) * self.batch_size
        indexes_batch = self.indexes[start_idx: end_idx]
        batch = self.dataset[indexes_batch]
        return batch
