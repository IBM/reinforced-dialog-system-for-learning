import matchzoo as mz
import pandas as pd
import json
import pickle
from tqdm import tqdm
'''
Prepare the data for training/validating the matching pyramid
'''

# preprocessor = mz.models.MatchPyramid.get_default_preprocessor()

for dtype in ['train', 'dev']:
    print('Processing %s' % dtype)
    PATH_IN = '../Talk_/data/WoW-raw/%s.json' % dtype
    with open(PATH_IN) as f:
        con = json.load(f)
    progress = tqdm(range(len(con)))
    text_left = []
    text_right = []
    for i, rec in enumerate(con):
        # ['question', 'answers', 'target', 'ctxs']
        dialog = rec['dialog']
        history = []
        for j, utterance in enumerate(dialog):
            speaker = utterance['speaker'][2:]
            text = utterance['text']
            history.append((speaker, text))
        assert len(history) >= 2    # If the conversation has less than two utterance it is deemed as noise
        if history[0][0] == 'Wizard':
            history = history[1:]
        for j in range(0, len(history), 2):
            if j + 1 < len(history):    # There is a wizard response
                text_left.append(history[j][1])
                text_right.append(history[j+1][1])
        _ = progress.update(1)
    data = pd.DataFrame({
        'text_left': text_left,
        'text_right': text_right,
        'label': [1 for _ in text_right]
    })
    pack_raw = mz.pack(data)
    PATH_OUT = '../Talk_/data/WoW-pyramid/%s.pkl' % dtype
    with open(PATH_OUT, 'wb') as f:
        pickle.dump(pack_raw, f)










