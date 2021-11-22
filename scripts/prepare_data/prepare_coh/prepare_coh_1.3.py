'''
This script generates additional training data
The additional training data would be added to the previous InferConvAI dataset
Differentiate data from InferConvAI and WoW
'''
import json
import random
import os


PATH_OUT = '../Talk_/data/InferConvAI-processed/v-1.3/'

# Load InferConvAI dataset
PATH_IN_IC = '../Talk_/data/InferConvAI-processed/v-1.0/train.json'
with open(PATH_IN_IC) as f:
    con = json.load(f)
instances_ic = con['data']
for i in range(len(instances_ic)):
    instances_ic[i]['data_src'] = 'ic'
print('Number of instances from InferConvAI: %s' % len(instances_ic))
# Number of instances from InferConvAI: 1,059,599

# Load WoW dataset
base_idx = len(instances_ic)
PATH_IN_WOW = '../Talk_/data/WoW-raw/train.json'
with open(PATH_IN_WOW) as f:
    con = json.load(f)
instances_wow = []
cnt = 0
for i, rec in enumerate(con):
    utts = rec['dialog']
    history = []
    for utt in utts:
        speaker = utt['speaker'][2:]
        text = utt['text'].lower()
        # if speaker == 'Wizard' and len(history) > 0:
        if len(history) > 0:
            instance = {
                "premise": history[-1],
                "hypothesis": text,
                'label': 0,
                'idx': base_idx + cnt,
                'data_src': 'wow'
            }
            instances_wow.append(instance)
            cnt += 1
        history.append(text)
print('Number of instances from WoW: %s' % len(instances_wow))
# Number of instances from WoW: 148,357

# Save the dataset
instances = instances_ic + instances_wow
if not os.path.exists(PATH_OUT):
    os.mkdir(PATH_OUT)
with open(PATH_OUT + 'train.json', 'w') as f:
    con = {
        'version': 1.3,
        'data': instances
    }
    json.dump(con, f)



# Validation and test data
for dtype in ['valid', 'test']:
    PATH_IN_IC = '../Talk_/data/InferConvAI-processed/v-1.0/%s.json' % dtype
    with open(PATH_IN_IC) as f:
        con = json.load(f)
    instances_ic = con['data']
    for i in range(len(instances_ic)):
        instances_ic[i]['data_src'] = 'ic'
    print('Number of instances from InferConvAI %s set: %s' % (dtype, len(instances_ic)))
    with open(PATH_OUT + '%s.json' % dtype, 'w') as f:
        con = {
            'version': 1.3,
            'data': instances_ic
        }
        json.dump(con, f)
