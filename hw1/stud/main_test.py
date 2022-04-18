"""Main file for interactive testing
"""
import os
from pathlib import Path
import sys
from typing import List, Tuple

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
import torchcrf
from torch.utils.data import DataLoader
import torch
from seqeval import metrics
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

acc, loss, f1, true, predicted = lstm.test(model, devloader, params, crf=crf)


def read_dataset(path: str) -> Tuple[List[List[str]], List[List[str]]]:

    tokens_s = []
    labels_s = []

    tokens = []
    labels = []

    with open(path) as f:

        for line in f:

            line = line.strip()

            if line.startswith("#\t"):
                tokens = []
                labels = []
            elif line == "":
                tokens_s.append(tokens)
                labels_s.append(labels)
            else:
                token, label = line.split("\t")
                tokens.append(token)
                labels.append(label)

    assert len(tokens_s) == len(labels_s)

    return tokens_s, labels_s


tokens, labels = read_dataset(str(config.DEV))
model.eval()
crf.eval()
preds = lstm.predict(model, vocab, tokens, device, window_size=100, crf=crf)
score = metrics.f1_score(labels, preds, average='macro')
print(f'{score=}')