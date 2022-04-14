'''
This scripts convert WoW dataset to coherence model trainig data
'''
import json
import os

for dtype in ['test']:
    instances = []
    PATH_IN_WOW = '../Talk_/data/WoW-raw/%s.json' % dtype
    with open(PATH_IN_WOW) as f:
        con = json.load(f)
    for i, rec in enumerate(con):
        dialog = rec['dialog']
        history = []
        for j, utterance in enumerate(dialog):
            speaker = utterance['speaker'][2:]
            text = utterance['text']
            history.append((speaker, text))
        if history[0][0] == 'Wizard':
            history = history[1:]
        pairs = []
        for j in range(0, len(history)-1, 2):
            pairs.append((history[j][1], history[j+1][1]))
        for idx1 in range(len(pairs)):
            for idx2 in range(len(pairs)):
                if idx1 == idx2:    # positive example
                    instance = {
                        "premise": pairs[idx1][0],
                        "hypothesis": pairs[idx2][1],
                        'label': 1,
                        'idx': len(instances)
                    }
                else:   # negative example
                    instance = {
                        "premise": pairs[idx1][0],
                        "hypothesis": pairs[idx2][1],
                        'label': 0,
                        'idx': len(instances)
                    }
                instances.append(instance)
    print('Number of instances from WoW: %s' % len(instances))
    PATH_OUT = '../Talk_/data/InferConvAI-processed/v-1.5/'
    if not os.path.exists(PATH_OUT):
        os.mkdir(PATH_OUT)
    with open(PATH_OUT + '%s.json' % dtype, 'w') as f:
        con = {
            'version': 1.5,
            'data': instances
        }
        json.dump(con, f)
