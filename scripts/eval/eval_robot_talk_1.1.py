import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from demos.v1_1.english_corner_utils import *
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from pathlib import Path
from tqdm import tqdm
import numpy as np

'''
This is the primary analysis scripts
This scripts evaluates the conversation when selector is absent
'''

'''
Step 1: Generate robot talk
'''
# load model
version2path = {
    '1.0': {
        'wiz': 'saved_models/wiz-1.0/checkpoint-21000/model.pt',
        'app': 'saved_models/app-1.0/checkpoint-29000/model.pt'
    },
    '1.1': {
        'wiz': 'saved_models/wiz-1.1/checkpoint-26000/model.pt',
        'app': 'saved_models/app-1.1/checkpoint-23000/model.pt'
    },
    '1.2': {
        'wiz': 'saved_models/wiz-1.2/checkpoint-20000/model.pt',
        'app': 'saved_models/app-1.2/checkpoint-30000/model.pt'
    }
}

version = '1.1'

with open('./za/args/args_doha_train.pkl', 'rb') as f:
    args_wiz = pickle.load(f)
    args_wiz.experiment_type = 'chat_document'
    args_wiz.model_file_path = version2path[version]['wiz']

with open('./za/args/args_bart_train.pkl', 'rb') as f:
    args_app = pickle.load(f)
    args_app.experiment_type = 'chat_document'
    args_app.model_file_path = version2path[version]['app']

app = BartQA(args_app)
wiz = MultiBartQA(args_wiz)
num_turns = 5

# load data
with open('./data/WoW-1.1/test_overlap.json') as f:
    test_ovlp = json.load(f)

with open('./data/WoW-1.1/test_nonoverlap.json') as f:
    test_nvlp = json.load(f)

TEST_NUM = 40

results_ovlp = []
for i, (topic, doc) in enumerate(test_ovlp):
    print('Overlap - %s' % i)
    history = make_conversation(wiz, app, topic, doc, num_turns, reverse=True)
    results_ovlp.append((topic, doc, history))
    if i > TEST_NUM:
        break

# 6 overlapping utterance warning
Path('./results/%s/' % version).mkdir(parents=True, exist_ok=True)
with open('./results/%s/ovlp.json' % version, 'w') as f:
    json.dump(results_ovlp, f)

results_nvlp = []
for i, (topic, doc) in enumerate(test_nvlp):
    print('Non-Overlap - %s' % i)
    history = make_conversation(wiz, app, topic, doc, num_turns, reverse=True)
    results_nvlp.append((topic, doc, history))
    if i > TEST_NUM:
        break

# 8 overlapping utterance warning
with open('./results/%s/nvlp.json' % version, 'w') as f:
    json.dump(results_nvlp, f)

'''
Step 2: Evaluate coverage
'''
rouger = Rouge()

# Overlap
doc_ovlp = {}
wiz_ovlp = {}
app_ovlp = {}
all_ovlp = {}
with open('./results/%s/ovlp.json' % version) as f:
    results_ovlp = json.load(f)

for idx, (topic, doc, history) in enumerate(results_ovlp):
    history_wiz = ' '.join([history[i] for i in range(len(history)) if i % 2 == 0])
    history_app = ' '.join([history[i] for i in range(len(history)) if i % 2 == 1])
    history_all = ' '.join(history)
    doc_ovlp[idx] = [doc]
    wiz_ovlp[idx] = [history_wiz]
    app_ovlp[idx] = [history_app]
    all_ovlp[idx] = [history_all]

score_wiz_ovlp, _ = rouger.compute_score(doc_ovlp, wiz_ovlp)
score_app_ovlp, _ = rouger.compute_score(doc_ovlp, app_ovlp)
score_all_ovlp, _ = rouger.compute_score(doc_ovlp, all_ovlp)
print('Overlapped Topics')
print('score_wiz_ovlp:\t%s' % score_wiz_ovlp)
print('score_app_ovlp:\t%s' % score_app_ovlp)
print('score_all_ovlp:\t%s' % score_all_ovlp)

# score_wiz_ovlp: 0.3084824766466013
# score_app_ovlp: 0.1023110976894328
# score_all_ovlp: 0.27116873900272376


# Non-overlap
doc_nvlp = {}
wiz_nvlp = {}
app_nvlp = {}
all_nvlp = {}
with open('./results/%s/nvlp.json' % version) as f:
    results_nvlp = json.load(f)

for idx, (topic, doc, history) in enumerate(results_nvlp):
    history_wiz = ' '.join([history[i] for i in range(len(history)) if i % 2 == 0])
    history_app = ' '.join([history[i] for i in range(len(history)) if i % 2 == 1])
    history_all = ' '.join(history)
    doc_nvlp[idx] = [doc]
    wiz_nvlp[idx] = [history_wiz]
    app_nvlp[idx] = [history_app]
    all_nvlp[idx] = [history_all]

score_wiz_nvlp, _ = rouger.compute_score(doc_nvlp, wiz_nvlp)
score_app_nvlp, _ = rouger.compute_score(doc_nvlp, app_nvlp)
score_all_nvlp, _ = rouger.compute_score(doc_nvlp, all_nvlp)
print('Non-Overlapped Topics')
print('score_wiz_nvlp:\t%s' % score_wiz_nvlp)
print('score_app_nvlp:\t%s' % score_app_nvlp)
print('score_all_nvlp:\t%s' % score_all_nvlp)

# score_wiz_nvlp: 0.2248878555555468
# score_app_nvlp: 0.07457239166468857
# score_all_nvlp: 0.1982674064356882


'''
Step 3: Evaluate coherence by PPL
'''
device = torch.device('cuda')
tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModel.from_pretrained("microsoft/DialoGPT-medium")
model = model.to(device)
model.eval()


def get_response_ppl(response, history):
    history_ = '<|endoftext|>'.join(history)
    history_ids = tokenizer.encode(history_)
    input_ids = tokenizer.encode(history_ + response, return_tensors='pt').to(device)
    target_ids = input_ids.clone()
    target_ids[0, :len(history_ids)] = -100
    with torch.no_grad():
        loss = model(input_ids, labels=target_ids)[0].cpu().item()
    return math.exp(loss)


# Seen
with open('./results/%s/ovlp.json' % version) as f:
    results_ovlp = json.load(f)

ppl_all = []
for idx, (topic, doc, conversation) in tqdm(enumerate(results_ovlp)):
    ppl_temp = []
    for j in range(len(conversation)):
        response = conversation[j]
        history = conversation[:j][:-3]
        ppl = get_response_ppl(response, history)
        ppl_temp.append(ppl)
    ppl_all.append(ppl_temp)

print('Avg PPL:\t%s' % np.mean(np.array(ppl_all)))
# Avg PPL:        503.71279275273054
# Avg PPL:        302.83214171778246 (last three turns)


with open('./results/%s/nvlp.json' % version) as f:
    results_nvlp = json.load(f)

ppl_all = []
for idx, (topic, doc, conversation) in tqdm(enumerate(results_nvlp)):
    ppl_temp = []
    for j in range(len(conversation)):
        response = conversation[j]
        history = conversation[:j][:-3]
        ppl = get_response_ppl(response, history)
        ppl_temp.append(ppl)
    ppl_all.append(ppl_temp)

print('Avg PPL:\t%s' % np.mean(np.array(ppl_all)))
# Avg PPL:        1088.1397292845459
# Avg PPL:        645.31916706343
