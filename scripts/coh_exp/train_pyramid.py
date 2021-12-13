import torch
import numpy as np
import matchzoo as mz
import pickle

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=10))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

print('data loading ...')
with open('../Talk_/data/WoW-pyramid/train.pkl', 'rb') as f:
    train_pack_raw = pickle.load(f)

with open('../Talk_/data/WoW-pyramid/dev.pkl', 'rb') as f:
    dev_pack_raw = pickle.load(f)

print('data loaded as `train_pack_raw` `dev_pack_raw`')


preprocessor = mz.models.MatchPyramid.get_default_preprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)

glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=10,
    batch_size=8,
    resample=True,
    sort=False,
    shuffle=True
)
devset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    batch_size=8,
    sort=False,
    shuffle=False
)

padding_callback = mz.models.MatchPyramid.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
devloader = mz.dataloader.DataLoader(
    dataset=devset,
    stage='dev',
    callback=padding_callback
)

model = mz.models.MatchPyramid()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['kernel_count'] = [16, 32]
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['dpool_size'] = [3, 10]
model.params['dropout_rate'] = 0.1

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=devloader,
    validate_interval=None,
    epochs=3
)

trainer.run()






