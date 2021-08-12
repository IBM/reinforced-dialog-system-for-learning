import os
from nltk.tokenize import sent_tokenize
from utils.mysql_utils import get_passage
from collections import defaultdict
import json
from rouge_score import rouge_scorer

# version 1.6 selector data
# conversation history in reverse order
# Remove complex document
# only use history and cur sentence
# only use later sentence in the conversation as negative response

dtype = 'train'
# ['train', 'dev', test_random_split.json]
PATH_IN = '../Talk_/data/WoW-raw/%s.json' % dtype

with open(PATH_IN) as f:
    con = json.load(f)

errors = []
instances = []
idx_dialog2_idx_ins = defaultdict(list)
scorer = rouge_scorer.RougeScorer(['rouge2'])
rouge_thresh = 0.7


def select_candidate_references(references, label, num_candidates=4):
    if len(references) == num_candidates:
        return references, label
    else:
        assert len(references) > num_candidates
        if label < num_candidates:
            return references[:num_candidates], label
        else:
            return references[label-num_candidates+1: label+1], num_candidates-1


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
                references = None
                if len(utterance['checked_sentence']) == 1 and \
                        'no_passages_used' not in utterance['checked_sentence']:
                    info = list(utterance['checked_sentence'].keys())[0].split('_')
                    cate = info[0]
                    checked_sent = list(utterance['checked_sentence'].values())[0]
                    reference_title = ' '.join(info[1:-1])
                    # if the reference passage is recorded, use it directly
                    if reference_title in all_refs:
                        references = all_refs[reference_title]
                    # else search for it in Wikipedia database
                    else:
                        passage = get_passage(reference_title)
                        if passage is None:
                            references = None
                        else:
                            try:
                                passage_ = sent_tokenize(passage)[:6]
                                references = passage_
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
                    if score >= rouge_thresh and len(references) > 3:
                        history_now = history_all.copy()
                        history_now.reverse()
                        # context_wiz = reference_title + ' \n ' + ' \n '.join(history_now)
                        context_wiz = '</s>'.join(history_now)
                        idx_dialog2_idx_ins[i].append(len(instances))
                        references_, label = select_candidate_references(references, idx_ref, 4)
                        instance = {
                            'context': context_wiz,
                            'doc': references_,
                            'labels': label,
                            'response': text
                        }
                        instances.append(instance)
        history_all.append(text.replace('\n', '').replace('#', ''))


out_path = '../Talk_/data/WoW-selector-1.6/'
if not os.path.exists(out_path):
    os.mkdir(out_path)
with open(out_path + dtype + '.json', 'w') as f:
    dataset = {
        'version': 1.6,
        'data': instances
    }
    json.dump(dataset, f)


