import json
import pickle
import torch
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np

with open('../Talk_/za/args/args_coh.pkl', 'rb') as f:
    args = pickle.load(f)

accelerator = Accelerator()
config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=3, finetuning_task=args.task_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
device = torch.device('cuda')
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
)
saved_model_path = '../Talk_/saved_models/coh-1.0/pytorch_model.bin'
model.load_state_dict(torch.load(saved_model_path))
model.to(device)


def predict(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)[0].cpu()
    return torch.argmax(outputs, dim=1).item()


'''
Test the model on the test set
'''
test_dataset_path = '../Talk_/data/InferConvAI-processed/v-1.1/test.json'
test_dataset = load_dataset('json', data_files={"test": test_dataset_path}, field='data')['test']
preds = []
golds = []
for idx in range(5566):
# for idx in range(100):
    premise = test_dataset['premise'][idx]
    hypothesis = test_dataset['hypothesis'][idx]
    gold = test_dataset['label'][idx]
    pred = predict(premise, hypothesis)
    golds.append(gold)
    preds.append(pred)
    if idx % 100 == 0:
        print(idx)
        print('accuracy:\t%s' % accuracy_score(golds, preds))

print('accuracy:\t%s' % accuracy_score(golds, preds))
# accuracy of coh-1.0 on the test set: 88.40
# accuracy of coh-1.1 on the test set: 88.96


'''
Test the model on the WoW test set
'''
with open('../Talk_/data/WoW-processed/pure_dialog_valid.json') as f:
    dialogs = json.load(f)[:100]

preds_all = []
for k, dialog in enumerate(dialogs):
    if k % 10 == 0:
        print(k)
    preds_dialog = []
    for i in range(len(dialog)-1):
        sent1, sent2 = dialog[i], dialog[i+1]
        pred = predict(sent1.lower(), sent2.lower())
        preds_dialog.append(pred)
    preds_all += preds_dialog

num_zero = np.sum(np.array((np.array(preds_all) == 0), dtype=int))
num_one = np.sum(np.array((np.array(preds_all) == 1), dtype=int))
num_two = np.sum(np.array((np.array(preds_all) == 2), dtype=int))
print('Zero rate: %s' % (num_zero/len(preds_all)))
print('One rate: %s' % (num_one/len(preds_all)))
print('Two rate: %s' % (num_two/len(preds_all)))

i, j = 12, 2
print(dialogs[i])
print('Pred label: %s' % preds_all[i][j])
print('Sent1: %s ' % dialogs[i][j])
print('Sent2: %s ' % dialogs[i][j+1])


'''
Test the model on a few generated utterance
'''
# 1
sent1 = 'What is the purpose of gardening?'
sent2 = 'Well gardening is the practice of growing and cultivating plants as part of horticulture.'
pred = predict(sent1.lower(), sent2.lower())
print('# pred result: %s' % pred)
# pred result: 2

# 2
sent1 = "Yes, I'm talking about the New York Knicks"
sent2 = "Agreed, I think I'm starting off as a Knicks fan! Does the league have a pick where they belong? "
pred = predict(sent1.lower(), sent2.lower())
print('# pred result: %s' % pred)
# pred result: 0

# 3
sent1 = "Yes, the NBA has a pick system"
sent2 = "Right? It's sad to think the cholker has made it this far, but, so do I."
pred = predict(sent1.lower(), sent2.lower())
print('# pred result: %s' % pred)
# pred result: 2


'''
Test the model on generated conversation (demo result 1.5 -- conversation over CNN-dailymail)
'''
with open('../Talk_/logs/demo_results/demo-1.5.json', 'r') as f:
    con = json.load(f)

# preds_all = []
# for rec in con:
#     preds_dialog = []
#     for idx in [1, 3, 5, 7]:
#         premise = rec['history'][idx]
#         hypothesis = rec['history'][idx+1]
#         pred = predict(premise.lower(), hypothesis.lower())
#         preds_dialog.append(pred)
#     preds_all.append(preds_dialog)

'''
coh_1.0
>>> preds_all[0]
[1, 1, 1, 0]
>>> preds_all[1]
[0, 0, 0, 2]
>>> preds_all[2]
[0, 2, 0, 0]
>>> preds_all[3]
[0, 0, 2, 1]
>>> preds_all[4]
[2, 1, 0, 2]
>>> preds_all[5]
[0, 0, 1, 1]
'''


def show_dialog(rec):
    print("doc: %s" % rec['doc'])
    if 'topic' in rec:
        print("topic: %s" % rec['topic'])
    preds_dialog = []
    print("wiz: %s" % rec['history'][0])
    for idx in [1, 3, 5, 7]:
        premise = rec['history'][idx]
        hypothesis = rec['history'][idx + 1]
        pred = predict(premise.lower(), hypothesis.lower())
        print('app: %s\nwiz: %s\npred: %s' % (premise, hypothesis, pred))
        preds_dialog.append(pred)
    return preds_dialog



'''
Test the model on generated conversation (demo result 1.6 -- conversation over Wikipedia)
'''

with open('../Talk_/logs/demo_results/demo-1.6.json', 'r') as f:
    con = json.load(f)

idxs = []
import numpy as np
for i in range(100):
    preds = show_dialog(con[i])
    if np.sum(preds) != 0:
        idxs.append(i)

