"""Main file for interactive training and testing
"""
#%% imports
import os

from numpy import double

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from curses import window
from pathlib import Path
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from stud import dataset
from stud import config
from stud import lstm
from stud import conll
from stud import hypers

device = config.DEVICE

# pretrained.build_pretrained_embeddings(save_stuff=True,
#                                        freeze=False,
#                                        double_linear=True,
#                                        use_pos=True,
#                                        hidden_size=200)

#%% load and test

# pretrained_emb, vocab_glove = pretrained.get_pretrained_embeddings()
with open(config.MODEL / 'glove-emb-vocab.pkl', 'rb') as f:
    pretrained_emb, vocab_glove = pickle.load(f)

vocab = vocab_glove

# trainset = dataset.NerDatasetChar(vocab=vocab, window_size=50)
# devset = dataset.NerDatasetChar(path=config.DEV,
#                                 vocab=vocab,
#                                 char_vocab=trainset.char_vocab,
#                                 window_size=100)

# trainloader, devloader = dataset.get_dataloaders(trainset=trainset,
#                                                  devset=devset,
#                                                  batch_size_train=128)

# without char
trainloader, devloader = dataset.get_dataloaders(use_pos=False,
                                                 vocab=vocab,
                                                 window_size=50,
                                                 batch_size_train=128)

# with char embedding
# model = lstm.NerModelChar(n_classes=13,
#                           embedding_dim=100,
#                           char_embedding_dim=50,
#                           char_vocab=trainset.char_vocab,
#                           vocab_size=len(vocab),
#                           padding_idx=vocab.pad,
#                           hidden_size=300,
#                           char_out_channels=50,
#                           char_hidden_size=50,
#                           bidirectional=True,
#                           pretrained_emb=pretrained_emb,
#                           freeze_weights=False).to(config.DEVICE)

# without char embedding
model = lstm.NerModel(n_classes=13,
                      embedding_dim=100,
                      vocab_size=len(vocab),
                      padding_idx=vocab.pad,
                      hidden_size=100,
                      bidirectional=True,
                      pretrained_emb=pretrained_emb).to(config.DEVICE)

params = hypers.get_default_params(model, vocab)

#%%
# test
# model.load_state_dict(
#     torch.load(config.MODEL / '9981-charembedding.pth',
#                map_location=config.DEVICE))
# lstm.test(model, devloader, params, logic=False)

# train
# model.load_state_dict(
#     torch.load(config.MODEL / 'emb-100.pth', map_location=config.DEVICE))

print(f'training model: {model}')
lstm.train(model, trainloader, devloader, params, use_crf=True)
