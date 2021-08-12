import os
from nltk.tokenize import sent_tokenize
from utils.mysql_utils import get_passage
from collections import defaultdict
import json

# version 1.1 selector data
# conversation history in reverse order
# Remove complex document
# only use history and cur sentence
# only use later sentence in the conversation as negative response

dtype = 'dev'
# ['train', 'dev', test_random_split.json]
PATH_IN = 'data/WoW-raw/%s.json' % dtype

with open(PATH_IN) as f:
    con = json.load(f)

errors = []
instances = []
idx_dialog2_idx_ins = defaultdict(list)

for i, rec in enumerate(con):
    # ['question', 'answers', 'target', 'ctxs']
    dialog = rec['dialog']
    # topic = rec['chosen_topic']
    all_refs = {}
    for _, utterance in enumerate(dialog):
        if 'retrieved_passages' in utterance:
            for retrieved_passage in utterance['retrieved_passages']:
                all_refs.update(retrieved_passage)
    history_all = []
    for j, utterance in enumerate(dialog):
        speaker = utterance['speaker'][2:]
        text = utterance['text']
        if speaker == 'Apprentice':
            pass
        else:
            assert speaker == 'Wizard'
            if 'checked_sentence' in utterance:
                reference = None
                if len(utterance['checked_sentence']) == 1 and \
                        'no_passages_used' not in utterance['checked_sentence']:
                    info = list(utterance['checked_sentence'].keys())[0].split('_')
                    cate = info[0]
                    checked_sent = list(utterance['checked_sentence'].values())[0]
                    reference_title = ' '.join(info[1:-1])
                    # if the reference passage is recorded, use it directly
                    if reference_title in all_refs:
                        reference = all_refs[reference_title]
                    # else search for it in Wikipedia database
                    else:
                        passage = get_passage(reference_title)
                        if passage is None:
                            reference = None
                        else:
                            try:
                                passage_ = sent_tokenize(passage)[:6]
                                reference = passage_
                            except:
                                errors.append(('%s-%s' % (i, j)))
                                reference = None
                if reference is not None:
                    history_now = history_all.copy()
                    history_now.reverse()
                    # context_wiz = reference_title + ' \n ' + ' \n '.join(history_now)
                    context_wiz = '</s>'.join(history_now)
                    idx_dialog2_idx_ins[i].append(len(instances))
                    instance = {
                        'context': context_wiz,
                        'positive_response': text,
                        'negative_response': [],
                        'doc': ' '.join(reference),
                        'labels': 0
                    }
                    instances.append(instance)
        history_all.append(text.replace('\n', '').replace('#', ''))

idx_ins2idxs_ins = {}
for idx_dialog, idxs_ins in idx_dialog2_idx_ins.items():
    for idx_ins in idxs_ins:
        idx_ins2idxs_ins[idx_ins] = idxs_ins

for idx_ins, instance in enumerate(instances):
    instance['negative_response'] = [idx for idx in idx_ins2idxs_ins[idx_ins] if idx > idx_ins]

out_path = './data/WoW-selector-1.1/'
if not os.path.exists(out_path):
    os.mkdir(out_path)
with open(out_path + dtype + '.json', 'w') as f:
    dataset = {
        'version': 1.1,
        'data': instances
    }
    json.dump(dataset, f)



