from pathlib import Path

import gensim.downloader
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import torch
from torch import nn

import dataset
import lstm


def compare_vocabularies():
    print('Loading pretrained glove embeddings')
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


def build_pretrained_embeddings(save_stuff: bool = False, freeze: bool = False):
    print('Loading pretrained glove embeddings')
    embeddings: KeyedVectors = gensim.downloader.load(
        'glove-wiki-gigaword-100')  # type: ignore
    
    # use only pretrained
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
                          pretrained_emb=vectors,  # type: ignore
                          freeze_weights=freeze)
    
    print('Building vocabulary of pretrained embeddings')
    vocab = dataset.Vocabulary(premade=embeddings.index_to_key)
    
    if save_stuff:
        print('Saving the model')
        torch.save(model.state_dict(), 'model/pre_bi.pth')
        print('Saving the vocab')
        vocab.dump_data(Path('model/glove_vocab.pkl'))
    return model, vocab


def fine_tune(vocab: dataset.Vocabulary, model: lstm.NerModel = None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    
    trainset = dataset.NerDataset(path=Path('data/train.tsv'), vocab=vocab)
    devset = dataset.NerDataset(path=Path('data/dev.tsv'), vocab=vocab)
    
    # dataloaders
    trainloader, devloader = dataset.get_dataloaders(trainset,
                                                     devset,
                                                     batch_size_train=128,
                                                     batch_size_dev=1024)
    
    if model is None:
        # model
        model = lstm.NerModel(n_classes=13,
                              embedding_dim=100,
                              vocab_size=400_002,
                              padding_idx=0,
                              hidden_size=100,
                              bidirectional=True,
                              pretrained_emb=None,
                              freeze_weights=True)
        model.load_state_dict(torch.load('model/pre_bi.pth'))
    
    # loss, opt, params
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_label_id,
                                  reduction='sum')
    
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.001,
                                momentum=0.9)
    
    params = lstm.TrainParams(optimizer=optimizer,
                              scheduler=None,
                              vocab=vocab,
                              loss_fn=loss_fn,
                              epochs=400,
                              log_steps=None,
                              verbose=True,
                              device=device,
                              f1_average='macro',
                              save_path=Path('model/'))
    
    lstm.train(model, trainloader, devloader, params)
    
    # as expected, fine-tuning the model while keeping the same embeddings
    # barely works, as the dictionary and the embeddings are completelly
    # misaligned. We have to build a new dictionary from the actual embeddings
    
    # 63.22 with lr=0.001 m=0.9 bs=128, pretrained, bidirectional
