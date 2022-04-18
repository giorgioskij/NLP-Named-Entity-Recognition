"""Main file for interactive testing
"""
import os
from pathlib import Path
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
import torchcrf
from torch.utils.data import DataLoader
import torch
from stud import dataset
from stud import config
from stud import lstm
from stud import hypers

device = config.DEVICE

vocab = dataset.Vocabulary(path=config.MODEL / 'vocab-glove.pkl')
devset = dataset.NerDataset(path=config.DEV, vocab=vocab)
devloader = DataLoader(devset, batch_size=64)

model: lstm.NerModel = lstm.NerModel(n_classes=13,
                                     embedding_dim=100,
                                     vocab_size=len(vocab),
                                     padding_idx=vocab.pad,
                                     hidden_size=100,
                                     bidirectional=True,
                                     pretrained_emb=None).to(config.DEVICE)
params: lstm.TrainParams = hypers.get_default_params(model, vocab)
crf: torchcrf.CRF = torchcrf.CRF(num_tags=14,
                                 batch_first=True).to(config.DEVICE)

# load models from file
model.load_state_dict(
    torch.load(config.MODEL / '7572-stacked-100h-crf.pth',
               map_location=config.DEVICE))
crf.load_state_dict(
    torch.load(config.MODEL / 'crf-7572.pth', map_location=config.DEVICE))

acc, loss, f1 = lstm.test(model, devloader, params, crf=crf)
