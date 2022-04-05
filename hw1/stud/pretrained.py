#%%
import gensim.downloader
from gensim.models.keyedvectors import KeyedVectors
import dataset
import lstm
import numpy as np
import torch
from torch import nn
from pathlib import Path


def build_pretrained_embeddings():
    embeddings: KeyedVectors = gensim.downloader.load(
        'glove-wiki-gigaword-100')  # type: ignore

    d = dataset.NerDataset(threshold=2)
    v = d.vocab

    v_words: set[str] = {v[i] for i in range(len(v))}  # type: ignore
    g_words: set[str] = set(embeddings.key_to_index)

    lenv = len(v_words)
    leng = len(g_words)
    common = len(v_words & g_words)

    print(f'Vocab has {lenv} words, pretrained has {leng}\n'
          f'common words: {common}\n'
          f'words unique in vocab: {lenv - common}\n'
          f'words unique in pretrained: {leng - common}\n'
          f'pretrained is all lowercase? '
          f'{not any(w.lower() != w for w in g_words)}')

    print('Add unk and pad to pretrained embeddings')
    vectors: np.ndarray = embeddings.vectors  # type: ignore
    pad_vector = np.random.rand(1, vectors.shape[1])
    unk_vector = np.mean(vectors, axis=0, keepdims=True)
    vectors = np.concatenate((pad_vector, unk_vector, vectors))
    vectors = torch.FloatTensor(vectors)  # type: ignore
    print(f'{vectors.shape=}')

    print('Building model with pretrained embeddings')
    model = lstm.NerModel(n_classes=13,
                          embedding_dim=vectors.shape[1],
                          vocab_size=vectors.shape[0],
                          padding_idx=0,
                          hidden_size=100,
                          bidirectional=True,
                          pretrained_emb=vectors)  # type: ignore

    print('Saving the model')
    torch.save(model.state_dict(), '../../model/pre_bi.pth')
    return model


#%%
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
trainset = dataset.NerDataset(threshold=2, window_size=100)
vocab: dataset.Vocabulary = trainset.vocab
pad_label_id = vocab.pad_label_id
devset = dataset.NerDataset(path=Path('../../data/dev.tsv'), vocab=vocab)

# dataloaders
trainloader, devloader = dataset.get_dataloaders(trainset,
                                                 devset,
                                                 batch_size_train=128,
                                                 batch_size_dev=1024)

# model
pre_model = lstm.NerModel(n_classes=13,
                          embedding_dim=100,
                          vocab_size=400_002,
                          padding_idx=0,
                          hidden_size=100,
                          bidirectional=True,
                          pretrained_emb=None,
                          freeze_weights=False)
pre_model.load_state_dict(torch.load('../../model/pre_bi.pth'))

# loss, opt, params
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_label_id, reduction='sum')
optimizer = torch.optim.SGD(params=pre_model.parameters(),
                            lr=0.001,
                            momentum=0.9)
params = lstm.TrainParams(optimizer=optimizer,
                          vocab=vocab,
                          loss_fn=loss_fn,
                          epochs=400,
                          log_steps=None,
                          verbose=True,
                          device=device,
                          f1_average='macro',
                          save_path=Path('../../model/'))

# train
lstm.train(pre_model, trainloader, devloader, params)


# 63.22 with lr=0.001 m=0.9 bs=128, pretrained, bidirectional