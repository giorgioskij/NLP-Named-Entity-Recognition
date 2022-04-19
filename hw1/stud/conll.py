"""
This file holds utility functions to interact with the Conll-2003 dataset.
It doesn't include any implementation, but rather calls to other functions with
many arguments, so it serves only to reduce boilerplate code in the main files.
"""
from pathlib import Path

import torch
import torch.utils.data

from . import dataset, lstm, config, hypers


def get_conll_dataloaders(vocab: dataset.Vocabulary):
    """
    Generate and return dataloaders for the conll-2003 dataset, if wanted
    with a custom vocabulary (needed for pretrained embeddings)

    Args:
        vocab (Optional[dataset.Vocabulary], optional):
        Custom vocabulary to use. Defaults to None.

    Returns:
        Tuple[torch.utils.data.dataloader, torch.utils.data.dataloader]:
            train dataloader and eval dataloader
    """
    trainset = dataset.NerDataset(path=config.DATA /
                                  Path('conll2003/train.txt'),
                                  vocab=vocab,
                                  threshold=2,
                                  window_size=100,
                                  is_conll=True,
                                  lower=True)
    devset = dataset.NerDataset(path=config.DATA / Path('conll2003/valid.txt'),
                                vocab=trainset.vocab,
                                is_conll=True,
                                lower=True)
    # dataloaders
    trainloader, devloader = dataset.get_dataloaders(trainset=trainset,
                                                     devset=devset,
                                                     batch_size_train=128,
                                                     batch_size_dev=256)

    return trainloader, devloader


def train_on_conll():
    """Generates a model and trains is on the ConLL-2003 dataset

    Returns:
        lstm.NerModel: the trained model
    """

    trainloader, devloader = get_conll_dataloaders()
    model, params = hypers.get_conll_hypers(
        trainloader.dataset.vocab)  # type: ignore
    lstm.train(model, trainloader, devloader, params)

    return model


def test_on_conll(model_name: str):
    """Tests a trained model con the ConLL-2003 dataset

    Args:
        model_name (str): basename of the trained model
    """
    _, devloader = get_conll_dataloaders()
    model, params = hypers.get_conll_hypers(
        devloader.dataset.vocab)  # type: ignore

    model_path: Path = config.MODEL / f'{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
    lstm.test(model, devloader, params)
