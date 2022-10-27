"""Main file for interactive training
"""
import os
import sys
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
import pickle
import torch
from torch import nn as nn
from stud import dataset, config, lstm

device = config.DEVICE

with open(config.MODEL / 'glove-emb-vocab.pkl', 'rb') as f:
    pretrained_emb, _ = pickle.load(f)

# get dataloaders
trainset: dataset.NerDatasetWnut = dataset.NerDatasetWnut(
    path=config.DATA / 'wnut17/wnut17train.conll', lower=True, threshold=2)
vocab: dataset.Vocabulary = trainset.vocab
devset: dataset.NerDatasetWnut = dataset.NerDatasetWnut(
    path=config.DATA / 'wnut17/emerging.dev.conll', vocab=vocab)

trainloader, devloader = dataset.get_dataloaders(trainset=trainset,
                                                 devset=devset,
                                                 vocab=vocab,
                                                 batch_size_train=64)

# without char embedding
model: lstm.NerModel = lstm.NerModel(n_classes=vocab.n_labels,
                                     embedding_dim=100,
                                     vocab_size=len(vocab),
                                     padding_idx=vocab.pad,
                                     hidden_size=100,
                                     bidirectional=True,
                                     pretrained_emb=pretrained_emb).to(
                                         config.DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_label_id, reduction='mean')
params: lstm.TrainParams = lstm.TrainParams(optimizer=optimizer,
                                            scheduler=None,
                                            vocab=vocab,
                                            loss_fn=loss_fn,
                                            epochs=400,
                                            log_steps=None,
                                            verbose=True,
                                            device=config.DEVICE,
                                            f1_average='macro',
                                            save_path=config.MODEL)

print(f'training model: {model}')
train_loss, train_f1, eval_loss, eval_f1 = lstm.train(model,
                                                      trainloader,
                                                      devloader,
                                                      params,
                                                      use_crf=True)

print('dumping the history')
with open(config.MODEL / 'wnut-history.pkl', 'wb') as f:
    pickle.dump((train_loss, train_f1, eval_loss, eval_f1), f)
