from pathlib import Path

import torch
from torch import nn

import dataset
import lstm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# dataset
trainset = dataset.NerDataset(path=Path('../../data/conll2003/train.txt'),
                              threshold=2, window_size=100, is_conll=True,
                              lower=False)
vocab: dataset.Vocabulary = trainset.vocab
devset = dataset.NerDataset(path=Path('../../data/conll2003/valid.txt'),
                            vocab=vocab, is_conll=True)
pad_label_id = vocab.pad_label_id

# dataloaders
trainloader, devloader = dataset.get_dataloaders(trainset,
                                                 devset,
                                                 batch_size_train=128,
                                                 batch_size_dev=1024)
# model
model = lstm.NerModel(n_classes=9, embedding_dim=100, vocab_size=len(vocab),
                      padding_idx=vocab.pad, hidden_size=100,
                      bidirectional=True)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# loss
weighted_loss = False
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_id, reduction='sum')

# params
params = lstm.TrainParams(optimizer=optimizer, scheduler=None, vocab=vocab,
                          loss_fn=loss_fn, epochs=400, log_steps=None,
                          verbose=True, device=device, f1_average='macro',
                          save_path=Path('../../model/'))

# train
train: bool = True
if train:
    lstm.train(model, trainloader, devloader, params)
else:
    model.load_state_dict(torch.load('../../model/6033bi.pth'))
