"""
Handles the generation of hyperparameters, and provides easy function to avoid
the repetition of boilerplate code.

Example - entire code to train a model on ConLL

    trainloader, devloader = conll.get_conll_dataloaders()
    model, params = hypers.get_conll_hypers(trainloader.dataset.vocab)
    lstm.train(model, trainloader, devloader, params)
"""
from . import dataset, lstm, config
import torch
from torch import nn


def get_conll_model(vocab):
    model = lstm.NerModel(n_classes=9,
                          embedding_dim=100,
                          vocab_size=len(vocab),
                          padding_idx=vocab.pad,
                          hidden_size=100,
                          bidirectional=True)
    return model


def get_conll_hypers(vocab):
    model: lstm.NerModel = lstm.NerModel(n_classes=9,
                                         embedding_dim=100,
                                         vocab_size=len(vocab),
                                         padding_idx=vocab.pad,
                                         hidden_size=100,
                                         bidirectional=True).to(config.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_label_id,
                                  reduction='sum')
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

    return model, params


def get_default_params(model: lstm.NerModel, vocab: dataset.Vocabulary):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_label_id,
                                  reduction='sum')
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
    return params
