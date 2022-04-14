#%%
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

# from stud.nerdtagger import NerdTagger
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

# pretrained_emb, vocab_glove = pretrained.get_pretrained_embeddings()
vocab = dataset.Vocabulary(path=config.MODEL / 'vocab-glove.pkl')
char_vocab = dataset.CharVocabulary(path=config.MODEL / 'vocab-char.pkl')

devset = dataset.NerDatasetChar(path=config.DEV,
                                vocab=vocab,
                                char_vocab=char_vocab)

devloader = DataLoader(devset, batch_size=32)

model = lstm.NerModelChar(n_classes=13,
                          embedding_dim=100,
                          char_embedding_dim=50,
                          char_vocab=char_vocab,
                          vocab_size=len(vocab),
                          padding_idx=vocab.pad,
                          hidden_size=100,
                          char_hidden_size=50,
                          bidirectional=True,
                          pretrained_emb=None).to(config.DEVICE)

params = hypers.get_default_params(model, vocab)

#%%
# test
model.load_state_dict(
    torch.load(config.MODEL / '9646-charembedding-glove.pth',
               map_location=config.DEVICE))
lstm.test(model, devloader, params)

# train
# model.load_state_dict(
#     torch.load(config.MODEL / 'emb-100.pth', map_location=config.DEVICE))

# print(f'training model: {model}')
# lstm.train(model, trainloader, devloader, params)
