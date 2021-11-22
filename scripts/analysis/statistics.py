from consts import *
import csv
import numpy as np
import sys

'''
WoW dataset, report the number for the following metrics
--number of turns
--avg length of a utterance
--avg length of a reference doc
'''
csv.field_size_limit(sys.maxsize)
for dtype in ['train', 'dev']:
    len_utts = []
    len_docs = []
    with open(BASE_PATH + 'data/WoW-processed/WoW-1.1/wiz/%s.tsv' % dtype) as f:
        for i, line in enumerate(csv.reader(f, delimiter="\t")):
            _, context, utt, doc = line
            len_utts.append(len(utt.split(' ')))
            len_docs.append(len(doc.split(' ')))
    with open(BASE_PATH + 'data/WoW-processed/WoW-1.1/app/%s.tsv' % dtype) as f:
        for j, line in enumerate(csv.reader(f, delimiter="\t")):
            _, context, utt, _ = line
            len_utts.append(len(utt.split(' ')))
    print('Avg length utterance in %s set:\t%s' % (dtype, np.mean(len_utts)))
    print('Avg length document in %s set:\t%s' % (dtype, np.mean(len_docs)))
    print('Num instance in %s set for wiz:\t%s' % (dtype, i))
    print('Num instance in %s set for app:\t%s' % (dtype, j))

'''
Avg length utterance in train set:	16.634892040857604
Avg length document in train set:	110.30335619006803
Num instance in train set for wiz:	74667
Num instance in train set for app:	83539

Avg length utterance in dev set:	16.612350926909908
Avg length document in dev set:	109.89410589410589
Num instance in dev set for wiz:	4003
Num instance in dev set for app:	4464
'''
