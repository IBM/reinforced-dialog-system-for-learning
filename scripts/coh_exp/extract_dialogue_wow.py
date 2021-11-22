import json
import random


dtype = 'dev'
PATH_IN = '../Talk_/data/WoW-raw/%s.json' % dtype
PATH_OUT = '../Talk_/data/WoW-processed/pure_dialog_valid.json'

with open(PATH_IN) as f:
    con = json.load(f)

dialogs = []
for i, rec in enumerate(con):
    dialog = []
    for j, utterance in enumerate(rec['dialog']):
        dialog.append(utterance['text'])
    dialogs.append(dialog)

random.shuffle(dialogs)
with open(PATH_OUT, 'w') as f:
    json.dump(dialogs, f)








