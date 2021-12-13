import os
import json
from datasets import load_dataset
from tqdm import tqdm

# version 1.7 dataset (CNN-Dailymail)
# Used for rl fintune=ing the generator

datasets = load_dataset('cnn_dailymail', '3.0.0')
for dtype, dataset in datasets.items():
    progress_bar = tqdm(range(dataset.num_rows))
    instances = []
    for rec in dataset:
        topic = rec['highlights'].split('\n')[0]
        document = rec['article'].replace('\n', ' ').replace('-- ', '').replace('(CNN)', '')
        txt = ' '.join(document.split(' ')[:130])
        instances.append({
            'topic': topic,
            'document': txt
        })
        progress_bar.update(1)
    out_path = '../Talk_/data/CNN-DailyMail-processed/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print('Volumn of %s set:\t %s' % (dtype, len(instances)))
    if dtype == 'validation':
        dtype = 'dev'
    with open(out_path + dtype + '.json', 'w') as f:
        dataset_processed = {
            'version': 1.7,
            'data': instances
        }
        json.dump(dataset_processed, f)

