import os
import json

# version 1.5 selector data
# Used to fintune the selector

dtype = 'train'
# ['train', 'dev', test_random_split.json]
PATH_IN = '../Talk_/data/WoW-raw/%s.json' % dtype

with open(PATH_IN) as f:
    con = json.load(f)

topic2document = {}
instances = []

for i, rec in enumerate(con):
    # ['question', 'answers', 'target', 'ctxs']
    dialog = rec['dialog']
    # topic = rec['chosen_topic']
    all_refs = {}
    for _, utterance in enumerate(dialog):
        for rec in utterance['retrieved_passages']:
            assert len(rec) == 1
            topic = list(rec.keys())[0]
            document = ' '.join(list(rec.values())[0])
            topic2document[topic] = document

for topic, document in topic2document.items():
    instances.append({
        'topic': topic,
        'document': document
    })

out_path = '../Talk_/data/WoW-selector-1.5/'
if not os.path.exists(out_path):
    os.mkdir(out_path)


with open(out_path + dtype + '.json', 'w') as f:
    dataset = {
        'version': 1.5,
        'data': instances
    }
    json.dump(dataset, f)



