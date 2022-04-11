"""Main file for interactive training and testing
"""
#%% imports
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from curses import window
from pathlib import Path
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from stud.nerdtagger import NerdTagger
from stud import pretrained
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

# vocab
print('Loading data...')
vocab = dataset.Vocabulary(path=config.MODEL / 'vocab-glove.pkl')

# dataset
trainloader, devloader = dataset.get_dataloaders(vocab,
                                                 use_pos=True,
                                                 window_size=50)

# model
model = lstm.NerModel(n_classes=13,
                      embedding_dim=100,
                      vocab_size=len(vocab),
                      padding_idx=vocab.pad_label_id,
                      hidden_size=200,
                      bidirectional=True,
                      double_linear=True,
                      use_pos=True).to(config.DEVICE)
params = hypers.get_default_params(model, vocab)

#%%
# test
# model.load_state_dict(
#     torch.load(config.MODEL / '7071-glove-200h-double.pth',
#                map_location=config.DEVICE))
# lstm.test(model, devloader, params)

# train
model.load_state_dict(
    torch.load(config.MODEL / 'emb-200-double-pos.pth',
               map_location=config.DEVICE))
lstm.train(model, trainloader, devloader, params)
