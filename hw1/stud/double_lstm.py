"""This whole idea is kinda thrash, remove this file at some point
"""

import dataset
import lstm
import torch
import pathlib


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
trainset = data.NerDataset(threshold=2, window_size=100)
vocab: data.Vocabulary = trainset.vocab
devset= data.NerDataset(path='../../data/dev.tsv', vocab=vocab)
trainloader, devloader = data.get_dataloaders(
    trainset,
    devset,
    batch_size_train=32,
    batch_size_dev=1024)
pad_label_id = vocab.get_label_id(vocab.pad_label)

# the detector has the purpose of distinguishing named entities from 'O'
detector: lstm.NerModel = lstm.NerModel(
    n_classes=2,
    embedding_dim=100,
    vocab_size=len(vocab),
    padding_idx=vocab.pad,
    hidden_size=100,
    bidirectional=True,
)

# the classifiers classifies the named entities is categories
classifier: lstm.NerModel = lstm.NerModel(
    n_classes=12,
    embedding_dim=100,
    vocab_size=len(vocab),
    padding_idx=(vocab.pad),
    hidden_size=100,
    bidirectional=True
)

# optimizer for the detector
detector_opt = torch.optim.SGD(
    detector.parameters(),
    lr=0.1,
    weight_decay=0,
    # momentum=0.9,
)

# optimizer for the classifier
classifier_opt = torch.optim.SGD(
    classifier.parameters(),
    lr=0.001,
    weight_decay=0,
    momentum=0.9,
)

detector_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=2)
classifier_loss_fn = torch.nn.CrossEntropyLoss()

lstm.train(
    detector,
    trainloader=trainloader,
    devloader=devloader,
    optimizer=detector_opt,
    vocab=vocab,
    loss_fn=detector_loss_fn,
    epochs=200,
    log_steps=None,
    device=device,
    early_stop=False,
    save_path=pathlib.Path('../../model/')
)

