"""Main file for interactive training and testing
"""

import torch
import data
import lstm
import pathlib
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
trainset = data.NerDataset(threshold=2, window_size=100)
vocab: data.Vocabulary = trainset.vocab
devset= data.NerDataset(path='../../data/dev.tsv', vocab=vocab)
pad_label_id = vocab.pad_label_id
o_weight = 0.005
loss_weights = torch.tensor([(1 - o_weight)/12] * 12 + [o_weight]).to(device)

# dataloaders
trainloader, devloader = data.get_dataloaders(
    trainset,
    devset,
    batch_size_train=32,
    batch_size_dev=1024)

# model
model = lstm.NerModel(n_classes=13,
                      embedding_dim=100,
                      vocab_size=len(vocab),
                      padding_idx=(vocab.pad),
                      hidden_size=100,
                      bidirectional=True)

# optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.001,
                            weight_decay=0,
                            momentum=0.9)

# loss
weighted_loss = False
loss_fn = nn.CrossEntropyLoss(
    weight=loss_weights if weighted_loss else None,
    ignore_index=pad_label_id,
    reduction='sum'
)

# params
params = lstm.TrainParams(optimizer=optimizer,
                          vocab=vocab,
                          loss_fn=loss_fn,
                          epochs=400,
                          log_steps=None,
                          verbose=True,
                          device=device,
                          f1_average='macro',
                          save_path=pathlib.Path('../../model/'))

# train
train = True
if train:
    lstm.train(model, trainloader, devloader, params)
else:
    model.load_state_dict(torch.load('../../model/4407.pth'))



# 44.07 with lr=0.01, no weight decay, standard crossentropy, 400 epochs

# 43.72 with lr=0.001, no weight decay, mom=0.9,
# standard crossentropy, 200 epochs

# 59.88 with lr=0.001, no weight decay, m=0.9, 400 ep,
# standard crossentropy, bidirectional 


