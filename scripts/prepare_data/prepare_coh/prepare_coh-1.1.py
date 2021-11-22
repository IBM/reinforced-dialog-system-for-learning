'''
This script generates additional training data
The additional training data would be added to the previous InferConvAI dataset
'''
import json
import random

# Load InferConvAI dataset
PATH_IN_IC = '../Talk_/data/InferConvAI-processed/v-1.0/train.json'
with open(PATH_IN_IC) as f:
    con = json.load(f)
instances_ic = con['data']
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
                'idx': base_idx + cnt
            }
            instances_wow.append(instance)
            cnt += 1
        history.append(text)
print('Number of instances from WoW: %s' % len(instances_wow))
# Number of instances from WoW: 148,357

# Save the dataset
instances = instances_ic + instances_wow
random.shuffle(instances)

PATH_OUT = '../Talk_/data/InferConvAI-processed/v-1.1/train.json'
with open(PATH_OUT, 'w') as f:
    con = {
        'version': 1.1,
        'data': instances
    }
    json.dump(con, f)










