import os
import json

# version 1.5 selector data
# Used to fintune the selector

# build train/dev data
dtype = 'train'
# ['train', 'dev', 'test_random_split']
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


# build test data
path_topics = '../Talk_/data/WoW-raw/topic_splits.json'
with open(path_topics) as f:
    topics = json.load(f)

cates_train = set(topics['train'])
cates_test = set(topics['test'])

topic2document_seen = {}
topic2document_unseen = {}
instances = []

dtype = 'test'
PATH_IN = '../Talk_/data/WoW-raw/%s.json' % dtype
with open(PATH_IN) as f:
    con = json.load(f)
for i, rec in enumerate(con):
    # ['question', 'answers', 'target', 'ctxs']
    dialog = rec['dialog']
    cate = rec['chosen_topic']
    all_refs = {}
    for _, utterance in enumerate(dialog):
        for rec in utterance['retrieved_passages']:
            assert len(rec) == 1
            topic = list(rec.keys())[0]
            document = ' '.join(list(rec.values())[0])
            # topic2document[topic] = document
            if cate in cates_train:
                topic2document_seen[topic] = document
            else:
                topic2document_unseen[topic] = document

for topic, document in topic2document_unseen.items():
    instances.append({
        'topic': topic,
        'document': document
    })

for i, (topic, document) in enumerate(topic2document_seen.items()):
    if i < 25:
        instances.insert(0, {
            'topic': topic,
            'document': document
        })
    else:
        instances.append({
            'topic': topic,
            'document': document
        })

out_path = '../Talk_/data/WoW-selector-1.5/'

with open(out_path + 'test.json', 'w') as f:
    dataset = {
        'version': 1.5,
        'data': instances
    }
    json.dump(dataset, f)
