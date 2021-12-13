import json
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

path_wiki = '../Talk_/data/WoW-selector-1.5/train.json'
with open(path_wiki) as f:
    con = json.load(f)['data']
ls = []
for rec in con:
    ls.append(len(rec['document'].split(' ')))
print('Avg length document in Wikipedia:\t%s' % np.mean(ls))


path_paper = '../Talk_/data/Papers-processed/train.json'
with open(path_paper) as f:
    con = json.load(f)['data']
ls = []
for rec in con:
    ls.append(len(rec['document'].split(' ')))
print('Avg length document in Papers:\t%s' % np.mean(ls))


path_wow_train = '../Talk_/data/WoW-selector-1.5/train.json'
path_wow_test = '../Talk_/data/WoW-selector-1.5/test.json'
path_cd_train = '../Talk_/data/CNN-DailyMail-processed/train.json'
path_cd_test = '../Talk_/data/CNN-DailyMail-processed/test.json'
path_paper_train = '../Talk_/data/Papers-processed/train.json'
path_paper_test = '../Talk_/data/Papers-processed/test.json'


def get_dataset_statistics(path):
    with open(path) as f:
        con = json.load(f)['data']
    print('Num Docs:\t%s' % len(con))
    ls = []
    for rec in con:
        doc = rec['document']
        ls.append(len(doc.split(' ')))
    print('Avg Length:\t%s' % np.mean(ls))


get_dataset_statistics(path_wow_train)
# Num Docs:	165023
# Avg Length:	111.78929603752204
get_dataset_statistics(path_wow_test)
# Num Docs:	25399
# Avg Length:	112.74581676443954
get_dataset_statistics(path_cd_train)
# Num Docs:	287113
# Avg Length:	129.88160410709372
get_dataset_statistics(path_cd_test)
# Num Docs:	11490
# Avg Length:	129.9029590948651
get_dataset_statistics(path_paper_train)

get_dataset_statistics(path_paper_test)












