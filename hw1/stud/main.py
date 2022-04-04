"""Main file for interactive training and testing
"""

import torch
import data
import lstm
import pathlib
import cbloss
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
trainset = data.NerDataset(threshold=2, window_size=100)
vocab: data.Vocabulary = trainset.vocab
devset= data.NerDataset(path='../../data/dev.tsv', vocab=vocab)
trainloader, devloader = data.get_dataloaders(
    trainset,
    devset,
    batch_size_train=128,
    batch_size_dev=1024)

model = lstm.NerModel(
    n_classes=13,
    embedding_dim=100,
    vocab_size=len(vocab),
    padding_idx=(vocab.pad),
    hidden_size=100,
    bidirectional=True
)


# 44.07 with lr=0.01, no weight decay, standard crossentropy, 400 epochs

# 43.72 with lr=0.001, no weight decay, mom=0.9,
# standard crossentropy, 200 epochs

# 59.88 with lr=0.001, no weight decay, m=0.9, 400 ep,
# standard crossentropy, bidirectional 


# optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    weight_decay=0,
    momentum=0.9
)

# loss
pad_label_id = vocab.get_label_id(vocab.pad_label)
o_weight = 0.005
loss_weights = torch.tensor([(1 - o_weight)/12] * 12 + [o_weight]).to(device)
loss_fn = nn.CrossEntropyLoss(
    # weight=loss_weights,
    ignore_index=pad_label_id,
    reduction='sum'
)

if True:
    model.load_state_dict(torch.load('../../model/5988bi.pth'))
else:
    lstm.train(
        model,
        trainloader,
        devloader,
        optimizer,
        vocab=vocab,
        loss_fn=loss_fn,
        patience=1,
        epochs=400,
        log_steps=None,
        device=device,
        early_stop=False,
        f1_average='macro',
        save_path=pathlib.Path('../../model/')
    )


# samples_per_cls = [
#     2975,
#     3551,
#     3375,
#     4556,
#     5090,
#     2770,
#     2987,
#     5716,
#     6084,
#     2598,
#     5805,
#     1710,
#     192841,
# ]

# loss_fn = cbloss.CB_loss(
#     samples_per_cls=samples_per_cls,
#     no_of_classes=14,
#     loss_type='focal',
#     beta=0.999,
#     gamma=2.0,
#     device=device,
# )