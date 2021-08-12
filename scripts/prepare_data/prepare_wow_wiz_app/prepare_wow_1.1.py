import csv
import codecs
import json
from rouge import Rouge
from nltk.tokenize import sent_tokenize
from utils.mysql_utils import get_passage
import os
from pathlib import Path

# version 1.1 conversation history in reverse order
# Also remove speaker in the conversation history

'''
Generate train and dev data
'''

dtype = 'dev'
# ['train', 'dev']
PATH_IN = 'data/WoW-raw/%s.json' % dtype
rouge = Rouge()
chat_hist_len = 3

with open(PATH_IN) as f:
    con = json.load(f)


def write_files(out_dir, all_data, split):
    if split.startswith('valid'):
        split = 'dev'
    with codecs.open(out_dir + '/' + split + '.tsv', "w", "utf-8") as out:
        spam = csv.writer(out, delimiter='\t')
        for i in range(len(all_data['source'])):
            spam.writerow([i, all_data['source'][i], all_data['target'][i], all_data['docs'][i]])


# examples_app = []
# examples_wiz = []
errors = []
source_list_app, target_list_app, doc_list_app = [], [], []
source_list_wiz, target_list_wiz, doc_list_wiz = [], [], []

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
            history_last = history_all[-1*chat_hist_len: ]
            history_last.reverse()
            source_app = '</s>'.join(history_last)
            source_list_app.append(source_app)
            target_list_app.append(text)
            doc_list_app.append('None')
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
                                r1 = rouge.get_scores(checked_sent, ' '.join(passage_))
                                assert r1['rouge-1']['p'] > 0.5
                                reference = passage_
                            except:
                                errors.append(('%s-%s' % (i, j)))
                                reference = None
                if reference is not None:
                    history_last = history_all[-1 * chat_hist_len:]
                    history_last.reverse()
                    source_wiz = reference_title + '</s>' + '</s>'.join(history_last)
                    source_list_wiz.append(source_wiz)
                    target_list_wiz.append(text)
                    doc_list_wiz.append(' '.join(reference))
        # history_all.append(text.replace('\n', ''))
        history_all.append(text.replace('\n', ''))


out_path_app = './data/WoW_1.1/app'
if not os.path.exists(out_path_app):
    Path(out_path_app).mkdir(parents=True, exist_ok=True)
data_app = {
    'source': source_list_app,
    'target': target_list_app,
    'docs': doc_list_app
}
write_files(out_path_app, data_app, dtype)

out_path_wiz = './data/WoW_1.1/wiz'
if not os.path.exists(out_path_wiz):
    Path(out_path_wiz).mkdir(parents=True, exist_ok=True)
data_wiz = {
    'source': source_list_wiz,
    'target': target_list_wiz,
    'docs': doc_list_wiz
}
write_files(out_path_wiz, data_wiz, dtype)


'''
Generate test data
'''
PATH_IN = 'data/WoW-raw/train.json'
with open(PATH_IN) as f:
    con_train = json.load(f)
themes_trained = {}
for rec in con_train:
    themes_trained[rec['chosen_topic']] = None

PATH_IN = 'data/WoW-raw/test.json'
with open(PATH_IN) as f:
    con_test = json.load(f)

outputs_themes_overlap = []
outputs_themes_nonoverlap = []
for rec in con_test:
    theme = rec['chosen_topic']
    for dialog in rec['dialog']:
        if 'retrieved_passages' in dialog:
            assert len(dialog['retrieved_passages']) > 0
            topic, paras = list(dialog['retrieved_passages'][0].items())[0]
            passage = ' '.join(paras)
            if topic in themes_trained:
                outputs_themes_overlap.append((topic, passage))
            else:
                outputs_themes_nonoverlap.append((topic, passage))
            break

import json
with open('./data/WoW-1.1/test_overlap.json', 'w') as f:
    json.dump(outputs_themes_overlap, f)
with open('./data/WoW-1.1/test_nonoverlap.json', 'w') as f:
    json.dump(outputs_themes_nonoverlap, f)



