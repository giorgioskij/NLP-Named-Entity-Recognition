from pathlib import Path
from typing import Optional

import torch
import torch.utils.data
from torch import nn

from . import dataset, lstm, config, hypers


def get_conll_dataloaders():
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
                                                     batch_size_dev=1024)

    return trainloader, devloader


def train_on_conll():
    """Generates a model and trains is on the ConLL-2003 dataset

    Returns:
        lstm.NerModel: the trained model
    """

    trainloader, devloader = get_conll_dataloaders()
    model, params = hypers.get_conll_hypers(trainloader.dataset.vocab)
    lstm.train(model, trainloader, devloader, params)

    return model


def test_on_conll(model_name: str):
    """Tests a trained model con the ConLL-2003 dataset

    Args:
        model_name (str): basename of the trained model
    """
    _, devloader = get_conll_dataloaders()
    model, params = hypers.get_conll_hypers(devloader.dataset.vocab)

    model_path: Path = config.MODEL / f'{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
    lstm.test(model, devloader, params)
