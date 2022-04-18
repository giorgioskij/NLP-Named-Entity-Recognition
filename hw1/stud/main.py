"""Main file for interactive training
"""
import os
import sys
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
import pickle
from stud import dataset
from stud import config
from stud import lstm
from stud import hypers

device = config.DEVICE

with open(config.MODEL / 'glove-emb-vocab.pkl', 'rb') as f:
    pretrained_emb, vocab = pickle.load(f)

# get dataloaders
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
model: lstm.NerModel = lstm.NerModel(n_classes=13,
                                     embedding_dim=100,
                                     vocab_size=len(vocab),
                                     padding_idx=vocab.pad,
                                     hidden_size=100,
                                     bidirectional=True,
                                     pretrained_emb=pretrained_emb).to(
                                         config.DEVICE)

params: lstm.TrainParams = hypers.get_default_params(model, vocab)

print(f'training model: {model}')
train_loss, train_f1, eval_loss, eval_f1 = lstm.train(model,
                                                      trainloader,
                                                      devloader,
                                                      params,
                                                      use_crf=True)

print('dumping the history')
with open(config.MODEL / 'history.pkl', 'wb') as f:
    pickle.dump((train_loss, train_f1, eval_loss, eval_f1), f)
