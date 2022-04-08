import random
import torch
import re
# from datasets import load_dataset
from collections import defaultdict
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from models.sim_models import Unlearned
from transformers import \
    (AutoConfig,
     AutoTokenizer,
     AutoModelForSequenceClassification,
     DPRQuestionEncoder,
     DPRContextEncoder,
     DPRQuestionEncoderTokenizerFast,
     DPRContextEncoderTokenizerFast,
     )
from nltk.tokenize import sent_tokenize


def get_uttrs(conv):
    uttrs = conv.split('/')
    uttrs_wiz = [uttr for i, uttr in enumerate(uttrs) if i % 2 == 0]
    uttrs_app = [uttr for i, uttr in enumerate(uttrs) if i % 2 != 0]
    return uttrs_wiz, uttrs_app, uttrs


class SummaQAEvaluator:
    def __init__(self):
        from summaqa import QG_masked
        from summaqa import QA_Metric
        self.question_generator = QG_masked()
        self.qa_metric = QA_Metric()

    def generate_masked_questions(self, article):
        sents = sent_tokenize(article)
        qa_pairs = []
        for sent in sents:
            masked_questions, answer_spans = self.question_generator.get_questions(sent)
            try:
                idx = random.sample(range(len(masked_questions)), 1)[0]
                qa_pairs.append((masked_questions[idx], answer_spans[idx]))
            except:
                continue
        return qa_pairs

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
        # scores['coh'].append(float(coh))
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


def eval_relv_is(convs, scores, device):
    print('Eval utterance relevance using InferSent metrics')
    config = AutoConfig.from_pretrained('bert-base-cased', num_labels=3, finetuning_task='mnli')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-cased',
        from_tf=False,
        config=config,
    ).to(device)
    model.load_state_dict(torch.load('../Talk_/saved_models/coh-1.0/pytorch_model.bin', map_location=device))
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
                outputs = model(**inputs)[0].cpu()
            pred = torch.argmax(outputs, dim=1).item()
            if pred == 0:
                prob = 1.0
            elif pred == 1:
                prob = 0.2
            else:
                assert pred == 2
                prob = 0.0
            probs.append(prob)
        scores['relv_is'].append(np.mean(probs))
        _ = progress.update(1)
    if cnt_errors > 0:
        print('Num errors:%s' % cnt_errors)
    return scores


def eval_relv_dpr(convs, scores, device):
    print('Eval utterance relevance using DPR')
    tokenizer_q = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    encoder_q = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(device)
    tokenizer_c = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    encoder_c = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base').to(device)
    progress = tqdm(range(len(convs)))
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
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
            uttr_wiz = uttrs[j + 1]
            ids_app = tokenizer_q(uttr_app, return_tensors='pt').to(device)
            ids_wiz = tokenizer_c(uttr_wiz, return_tensors='pt').to(device)
            with torch.no_grad():
                emb_app = encoder_q(**ids_app)[0]
                emb_wiz = encoder_c(**ids_wiz)[0]
                relv = cos(emb_app, emb_wiz).cpu().item()
                relvs.append(relv)
        scores['relv_dpr'].append(np.mean(relvs))
        _ = progress.update(1)
    if cnt_errors > 0:
        print('Num errors:%s' % cnt_errors)
    return scores


def eval_objective(path_data, path_result, num_test, device=None, test_goals=None):
    from datasets import load_dataset
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
    if 'relv_is' in test_goals:
        scores = eval_relv_is(convs, scores, device)
    if 'relv_dpr' in test_goals:
        scores = eval_relv_dpr(convs, scores, device)
    # Write results
    with open(path_result[:-4] + '.results.txt', 'a') as f_out:
        for key, values in scores.items():
            if '-3' in key:
                continue
            print('Avg score of %s: %s' % (key, np.mean(values)))
            f_out.write('Avg score of %s: %s\n' % (key, np.mean(values)))


def show_history(history, reverse=True):
    uttrs = history.split(' / ')
    if reverse:
        uttrs.reverse()
    for i, uttr in enumerate(uttrs):
        if i % 2 == 0:
            print('Wiz:\t%s' % uttr)
        else:
            print('App:\t%s' % uttr)


# Add a leading question at the end of the conversation
def add_post_prompt(uttr, history, cate='food', idx=0):
    prompts = {
        'food': [
            ('Do you wish to know when it is usually served?', "When do people usually eat it?"),
            ("Would you like to know where it comes from?", "Where does it come from?"),
            ("Do you know what it is made of?", "What is it made of?"),
            ("You know what is special about this food?", "What is special about it?")
        ],
        'city': [
            ("Do you know what the city is famous for?", 'What is the city famous for?'),
            ("You know where it locates?", 'Where is it located?'),
            ("Now shall we talk about the city's population?", "What is the city's population?"),
            ("Would you like to know an interesting fact about the city?", "What is special about it?"),
        ],
        'plane': [
            ("Do you know who invent this plane?", 'Which company invented the plane?'),
            ("Would you like to know about its first flight", 'Tell me about its first flight?'),
            ("I have a fun fact about this plane, would you like to know?", "What are other special things about it?")
        ],
        'film': [
            ("Would you like to know about the story?", "What is it about?"),
            ("Do you wish to know who made the film?", "Who made it?"),
            ("Do you know who starred in the film?", "Who stars in the film ?"),
            ("I have an intersting fact about the film, would you like to listen?", "What is special about it?"),
        ]
    }
    added_prompt, real_question = prompts[cate][idx]
    return uttr + ' ' + added_prompt, real_question


# Add a starting statement at the begin of the conversation
def add_pre_prompt(uttr, topic, dtype):
    prompt_idx = random.choice(list(range(4)))
    prompts = {
        'wow':[
            "Let's talk about %s today. ",
            "We will be discussing %s today. ",
            "The topic today will be %s. ",
            "We will talk about %s. "
        ],
        'cd': [
            "Let's focus on the following news: %s. ",
            "We will be talking about the breaking new %s. "
            "Hi! Let's talk about %s. "
            "I'd like to draw your attention to a breaking news: %s."
        ]
    }
    return prompts[dtype][prompt_idx] % topic + uttr


# Anti-repetition generation
def get_wiz_say(trainer, topics, histories, documents):
    uttrs = histories[0].split(' / ')
    ids_wiz = [2*idx+1 for idx in range(int(len(uttrs)/2))]
    uttrs_wiz = [uttrs[idx] for idx in ids_wiz]
    wiz_say = trainer.generate_wiz_response(topics, histories, documents, do_post_process=True)[0]
    cnt = 0
    num_no_repeat_uttrs = len(uttrs_wiz)
    while cnt < 5:
        for uttr in uttrs_wiz:
            rouge = scorer_cov.score(uttr, wiz_say)
            if rouge['rougeL'].precision < 0.8 and rouge['rougeL'].recall < 0.8:
                num_no_repeat_uttrs -= 1
        if num_no_repeat_uttrs == 0:
            break
        wiz_say = trainer.generate_wiz_response(topics, histories, documents, do_post_process=True)[0]
    return wiz_say


def make_human_eval_conversation(trainer, topic, document, dtype='wow-food'):
    topics = [topic]
    documents = [document]
    histories = ['']
    histories_real = []
    dtype0, dtype1 = dtype.split('-')
    # turn 1
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    wiz_say_prompted = add_pre_prompt(wiz_say, topic, dtype0)
    wiz_say_prompted, app_response_cand = add_post_prompt(wiz_say_prompted, '', dtype1, 0)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append('Teacher say:\t%s' % wiz_say_prompted)
    print('Teacher say:\t%s' % wiz_say_prompted)
    app_say = input("You say:\t")
    if app_say.lower().startswith('yes'):
        app_say_real = app_say
        app_say = app_response_cand
    else:
        app_say_real = app_say
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('You say:\t%s' % app_say_real)
    # turn 2
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    wiz_say_prompted, app_response_cand = add_post_prompt(wiz_say, '', dtype1, -1)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append('Teacher say:\t%s' % wiz_say_prompted)
    print('Teacher say:\t%s' % wiz_say_prompted)
    app_say = input("You say:\t")
    if app_say.lower().startswith('yes'):
        app_say_real = app_say
        app_say = app_response_cand
    else:
        app_say_real = app_say
    histories = trainer.update_histories([app_say], histories, reverse=True)
    histories_real.append('You say:\t%s' % app_say_real)
    # turn 3
    wiz_say = get_wiz_say(trainer, topics, histories, documents)
    histories = trainer.update_histories([wiz_say], histories, reverse=True)
    histories_real.append('Teacher say:\t%s' % wiz_say)
    print('===Conversation history===')
    for uttr in histories_real:
        print(uttr)





