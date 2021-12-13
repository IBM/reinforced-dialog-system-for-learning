import torch
from summaqa import QG_masked
from summaqa import QA_Metric
import re
from datasets import load_dataset
from collections import defaultdict
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from models.sim_models import Unlearned
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


def get_uttrs(conv):
    uttrs = conv.split('/')
    uttrs_wiz = [uttr for i, uttr in enumerate(uttrs) if i % 2 == 0]
    uttrs_app = [uttr for i, uttr in enumerate(uttrs) if i % 2 != 0]
    return uttrs_wiz, uttrs_app, uttrs


class SummaQAEvaluator:
    def __init__(self):
        self.question_generator = QG_masked()
        self.qa_metric = QA_Metric()

    def get_conv_score(self, conv, article):
        masked_questions, answer_spans = self.question_generator.get_questions(article)
        uttrs_wiz, uttrs_app, uttrs = get_uttrs(conv)
        summary = ' '.join(uttrs_wiz)
        scores = self.qa_metric.compute(masked_questions, answer_spans, summary)
        return scores


scorer_cov = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scorer_qa = SummaQAEvaluator()
pat_conv = re.compile('History: ([^\n]+)')
pat_score = re.compile('Step [0-9]+ instance [0-9] \| Cov score: ([\.0-9]+) \| Coh score: ([\.0-9]+) \|')


def eval_rouge(convs, eval_dataset, scores):
    print('Eval Rouge scores')
    progress = tqdm(range(len(convs)))
    for i, conv in enumerate(convs):
        psg = eval_dataset[i]['document']
        uttrs_wiz, uttrs_app, uttrs = get_uttrs(conv)
        history = ''
        assert len(uttrs_wiz) > 1
        for j, uttr in enumerate(uttrs_wiz):
            if j > 2:
                continue
            len_uttr = len(uttr.split(' '))
            scores['len-%s' % j].append(len_uttr)
            score_old = scorer_cov.score(psg, history)
            history += uttr + ' '
            score_new = scorer_cov.score(psg, history)
            for key in score_new.keys():
                gain = score_new[key].fmeasure - score_old[key].fmeasure
                scores['%s-%s_gain' % (key, j)].append(gain)
        for key in score_new.keys():
            scores[key].append(score_new[key].fmeasure)
        _ = progress.update(1)
    return scores


def eval_direct_read(con, scores):
    scores_ = pat_score.findall(con)
    print('Eval average score (cov, coh)')
    for cov, coh in scores_:
        scores['cov'].append(float(cov))
        scores['coh'].append(float(coh))
    return scores


def eval_qa(convs, eval_dataset, scores):
    print('Eval QA')
    progress = tqdm(range(len(convs)))
    for i, conv in enumerate(convs):
        rec = eval_dataset[i]
        article = rec['document']
        scores_qa = scorer_qa.get_conv_score(conv, article)
        for key, value in scores_qa.items():
            scores[key].append(value)
        _ = progress.update(1)
    return scores


def eval_relv_unlearned(convs, scores, device):
    sim_evaluator = Unlearned(device)
    print('Eval utterance relevance using unlearned metrics')
    progress = tqdm(range(len(convs)))
    cnt_errors = 0
    for i, conv in enumerate(convs):
        uttrs_wiz, uttrs_app, uttrs = get_uttrs(conv)
        try:
            assert len(uttrs) >= 5
        except:
            cnt_errors += 1
            continue
        relvs = []
        for j in range(1, 5, 2):
            uttr_app = uttrs[j]
            uttr_wiz = uttrs[j+1]
            relv = sim_evaluator.get_relv(uttr_app, uttr_wiz)
            relvs.append(relv)
        scores['relv_unlearned'].append(np.mean(relvs))
        progress.update(1)
    if cnt_errors > 0:
        print('Num errors:%s' % cnt_errors)
    return scores


def eval_relv_wow(convs, scores, device):
    print('Eval utterance relevance using WoW metrics')
    config = AutoConfig.from_pretrained('bert-base-cased', num_label=2)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-cased',
        from_tf=False,
        config=config,
    ).to(device)
    model.load_state_dict(torch.load('../Talk_/saved_models/coh-1.5.1/pytorch_model.bin', map_location=device))
    model.eval()
    progress = tqdm(range(len(convs)))
    cnt_errors = 0
    for i, conv in enumerate(convs):
        uttrs_wiz, uttrs_app, uttrs = get_uttrs(conv)
        try:
            assert len(uttrs) >= 5
        except:
            cnt_errors += 1
            continue
        probs = []
        for j in range(1, 5, 2):
            uttr_app = uttrs[j]
            uttr_wiz = uttrs[j + 1]
            inputs = tokenizer(uttr_app, uttr_wiz, return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model(**inputs)[0].cpu()
            prob = torch.softmax(logits, dim=1)[0][1].item()
            probs.append(prob)
        scores['relv_wow'].append(np.mean(probs))
        _ = progress.update(1)
    if cnt_errors > 0:
        print('Num errors:%s' % cnt_errors)
    return scores


def eval_objective(path_data, path_result, num_test, device=None, test_goals=None):
    if test_goals is None:
        test_goals = set([])
    if not device:
        device = torch.device('cpu')
    extension = 'json'
    raw_datasets = load_dataset(extension, data_files={'test': path_data}, field='data')
    eval_dataset = raw_datasets['test'].select(range(num_test))
    assert path_result.endswith('.txt')
    with open(path_result) as f:
        con = f.read()
    convs = pat_conv.findall(con)
    scores = defaultdict(list)
    # assert len(scores_) == len(convs)
    # 1.1 Average score (cov, coh)
    if 'read' in test_goals:
        scores = eval_direct_read(con, scores)
    # 1.2 Gain in each turn
    if 'rouge' in test_goals:
        scores = eval_rouge(convs, eval_dataset, scores)
    # 1.3 QA scores
    if 'qa' in test_goals:
        scores = eval_qa(convs, eval_dataset, scores)
    if 'relv_unlearned' in test_goals:
        scores = eval_relv_unlearned(convs, scores, device)
    if 'relv_wow' in test_goals:
        scores = eval_relv_wow(convs, scores, device)
    # Write results
    with open(path_result[:-4] + '.results.txt', 'a') as f_out:
        for key, values in scores.items():
            if '-3' in key:
                continue
            print('Avg score of %s: %s' % (key, np.mean(values)))
            f_out.write('Avg score of %s: %s\n' % (key, np.mean(values)))


# path_data = '../Talk_/data/CNN-DailyMail-processed/test.json'
# path_result = '../Talk_/results/1.7.6.txt'




