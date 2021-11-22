import os
from nltk.tokenize import sent_tokenize
from utils.mysql_utils import get_passage
from collections import defaultdict
import json
from rouge_score import rouge_scorer
from tqdm import tqdm

# version 1.7 selector pre-train data
# conversation history in reverse order
# Remove complex document
# only use history and cur sentence
# only use later sentence in the conversation as negative response
# only use references in database, select sentences from 6 candicates

dtype = 'train'
# ['train', 'dev', test_random_split.json]
PATH_IN = '../Talk_/data/WoW-raw/%s.json' % dtype

with open(PATH_IN) as f:
    con = json.load(f)

errors = []
instances = []
idx_dialog2_idx_ins = defaultdict(list)
scorer = rouge_scorer.RougeScorer(['rouge2'])
rouge_thresh = 0.6


def select_candidate_references(references, label, num_candidates=4):
    if len(references) == num_candidates:
        return references, label
    else:
        assert len(references) > num_candidates
        if label < num_candidates:
            return references[:num_candidates], label
        else:
            return references[label-num_candidates+1: label+1], num_candidates-1


progress_bar = tqdm(range(len(con)))
for i, rec in enumerate(con):
    progress_bar.update(1)
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
                references = None
                if len(utterance['checked_sentence']) == 1 and \
                        'no_passages_used' not in utterance['checked_sentence']:
                    info = list(utterance['checked_sentence'].keys())[0].split('_')
                    cate = info[0]
                    checked_sent = list(utterance['checked_sentence'].values())[0]
                    reference_title = ' '.join(info[1:-1])
                    passage = get_passage(reference_title)
                    if passage is None:
                        references = None
                    else:
                        try:
                            references = sent_tokenize(passage)[:8]
                        except:
                            errors.append(('%s-%s' % (i, j)))
                            references = None
                if references is not None:
                    # We need to make sure the checked sent is in the reference
                    ref2rouge = {}
                    for k, reference in enumerate(references):
                        ref2rouge[k] = scorer.score(reference, checked_sent)['rouge2'].recall
                    sorted_ref2rouge = sorted(ref2rouge.items(), key=lambda kv: kv[1], reverse=True)
                    idx_ref, score = sorted_ref2rouge[0]
                    # If the checked sent is in the reference
                    if score >= rouge_thresh and len(references) >= 6:
                        history_now = history_all.copy()
                        history_now.reverse()
                        # context_wiz = reference_title + ' \n ' + ' \n '.join(history_now)
                        context_wiz = ' \\ '.join(history_now)
                        idx_dialog2_idx_ins[i].append(len(instances))
                        references_, label = select_candidate_references(references, idx_ref, 6)
                        instance = {
                            'context': context_wiz,
                            'doc': references_,
                            'labels': label,
                            'response': text,
                            'checked_sent': checked_sent
                        }
                        instances.append(instance)
        history_all.append(text.replace('\n', '').replace('#', ''))


out_path = '../Talk_/data/WoW-selector-1.7/'
if not os.path.exists(out_path):
    os.mkdir(out_path)
with open(out_path + dtype + '.json', 'w') as f:
    dataset = {
        'version': 1.7,
        'data': instances
    }
    json.dump(dataset, f)

print('Num instances in %s set - Raw:\t%s' % (dtype, len(con)))
print('Num instances in %s set - Processed:\t%s' % (dtype, len(instances)))

# Num instances in train set - Raw:	18430
# Num instances in train set - Processed:	60784

# Num instances in dev set - Raw:	981
# Num instances in dev set - Processed:	3232
