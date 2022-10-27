"""Main file for interactive testing
"""
import os
from pathlib import Path
import sys
from typing import List, Tuple

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.append(str(Path(__file__).parent.parent))
# os.system('conda activate nlp2022-hw1')

# pylint: disable=wrong-import-position
import torchcrf
from torch.utils.data import DataLoader
import torch
from sklearn import metrics as skmetrics
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from stud import dataset, config, lstm, hypers, conll

device = config.DEVICE

vocab = dataset.Vocabulary(path=config.MODEL / 'vocab-glove.pkl')
# for our dataset
devset = dataset.NerDataset(path=config.DEV, vocab=vocab)
devloader = DataLoader(devset, batch_size=64)

# for CoNNL-2003
# _, devloader = conll.get_conll_dataloaders(vocab)

n_classes = 9 if devloader.dataset.is_conll else 13
model: lstm.NerModel = lstm.NerModel(n_classes=n_classes,
                                     embedding_dim=100,
                                     vocab_size=len(vocab),
                                     padding_idx=vocab.pad,
                                     hidden_size=100,
                                     bidirectional=True,
                                     pretrained_emb=None).to(config.DEVICE)
params: lstm.TrainParams = hypers.get_default_params(model, vocab)
crf: torchcrf.CRF = torchcrf.CRF(num_tags=n_classes + 1,
                                 batch_first=True).to(config.DEVICE)

# load models from file
model.load_state_dict(
    torch.load(config.MODEL / '7597-stacked-100h-crf.pth',
               map_location=config.DEVICE))
crf.load_state_dict(
    torch.load(config.MODEL / 'crf-7597.pth', map_location=config.DEVICE))

acc, loss, f1 = lstm.test(model, devloader, params, crf=crf)


def read_dataset(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """This function was copied here from evaluate.py just for convenience

    Args:
        path (str): path to read the datset from

    Returns:
        Tuple[List[List[str]], List[List[str]]]: sentences
    """
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

labels_flat = [l for i in labels for l in i]
preds_flat = [p for i in preds for p in i]
cm = skmetrics.confusion_matrix(labels_flat, preds_flat, normalize='true')
target_names = vocab.ltos[:-1]

plt.style.use('bmh')
fig = plt.figure(figsize=(12, 8))
plt.title('Confusion matrix')
sns.heatmap(cm,
            annot=True,
            fmt='.2f',
            xticklabels=target_names,
            yticklabels=target_names)
fig.savefig('notes/img/cm.jpg', dpi=300)
plt.show()