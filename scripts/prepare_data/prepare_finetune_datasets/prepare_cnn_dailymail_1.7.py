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
    for text in dataset['highlights']:
        topic = ' '
        document = text.replace('\n', ' ')
        instances.append({
            'topic': topic,
            'document': document
        })
        progress_bar.update(1)
    out_path = '../Talk_/data/CNN-DailyMail-processed/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print('Volumn of %s set:\t %s' % (dtype, len(instances)))
    with open(out_path + dtype + '.json', 'w') as f:
        dataset_processed = {
            'version': 1.7,
            'data': instances
        }
        json.dump(dataset_processed, f)

