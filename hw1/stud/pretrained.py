"""
Build and train models that use pretrained word embeddings, like the ones from
glove-wiki-gigaword-100

the best results were achieved loading pretrained embeddings and fine-tuning
them, with double_linear = False and hidden_dim of the lstm = 200
"""

from pathlib import Path
from typing import List
import os

import gensim.downloader
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import torch
from torch import nn

from hw1.stud import dataset
from hw1.stud import lstm


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


def merge_pretrained_embeddings(save_stuff: bool = False,
                                freeze: bool = False,
                                double_linear: bool = True,
                                use_pos: bool = False):
    """
        Creates a new vocabulary merging the one from the pretrained embeddings
        and the one from our dataset. Then extends the embedding matrix with
        as many random vectors as there are unique words in our vocabulary, to
        be trained while the others are being fine-tuned.
    """
    print('Loading pretrained glove embeddings')
    embeddings: KeyedVectors = gensim.downloader.load(
        'glove-wiki-gigaword-100')  # type: ignore
    glove_count, emb_size = embeddings.vectors.shape

    # set of words in the pretrained embeddings
    # glove_vocab: dataset.Vocabulary = dataset.Vocabulary(
    #     premade=embeddings.index_to_key)
    glove_words: set[str] = set(embeddings.index_to_key)

    # set of words in our dataset
    trainset: dataset.NerDataset = dataset.NerDataset(
        path=Path('data/train.tsv'), threshold=2)
    our_vocab: dataset.Vocabulary = trainset.vocab
    our_words: set[str] = {our_vocab[i] for i in range(len(our_vocab))}

    # how many words are in our vocabulary and not in the glove embeddings?
    only_our_words: set[str] = our_words - glove_words
    only_our_words_count: int = len(only_our_words)

    # add our words to the glove vocabulary (preserving the order)
    new_wordlist: List[str] = embeddings.index_to_key.copy()
    new_wordlist.extend(only_our_words)

    # make sure that the order has not changed
    assert new_wordlist[:glove_count] == embeddings.index_to_key

    # generate a new vocabulary with all the words
    vocab: dataset.Vocabulary = dataset.Vocabulary(premade=new_wordlist)

    # extend the pretrained embedding matrix with new untrained vectors
    # to match the size of the new vocabulary
    vectors: np.ndarray = embeddings.vectors
    rand_vectors: np.ndarray = np.random.rand(only_our_words_count, emb_size)
    new_vectors: np.ndarray = np.concatenate((vectors, rand_vectors))

    # make sure that the size is correct
    assert new_vectors.shape == (len(glove_words | our_words), emb_size)

    new_vectors: torch.Tensor = torch.FloatTensor(new_vectors)

    print('Building model with pretrained embeddings')
    model = lstm.NerModel(
        n_classes=13,
        embedding_dim=emb_size,
        vocab_size=vectors.shape[0],
        padding_idx=vocab.pad,
        hidden_size=100,
        bidirectional=True,
        pretrained_emb=new_vectors,  # type: ignore
        freeze_weights=freeze,
        double_linear=double_linear,
        use_pos=use_pos)

    if save_stuff:
        print('Saving the model')
        torch.save(model.state_dict(), 'model/pre_bi_merged.pth')
        print('Saving the vocab')
        vocab.dump_data(Path('model/glove_vocab_merged.pkl'))
    return model, vocab


def build_pretrained_embeddings(save_stuff: bool = False,
                                freeze: bool = False,
                                double_linear: bool = True,
                                use_pos: bool = False,
                                hidden_size: int = 100):
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

    print('Building vocabulary of pretrained embeddings')
    vocab = dataset.Vocabulary(premade=embeddings.index_to_key)

    print('Building model with pretrained embeddings')
    model = lstm.NerModel(
        n_classes=13,
        embedding_dim=vectors.shape[1],
        vocab_size=vectors.shape[0],
        padding_idx=vocab.pad,
        hidden_size=hidden_size,
        bidirectional=True,
        pretrained_emb=vectors,  # type: ignore
        freeze_weights=freeze,
        double_linear=double_linear,
        use_pos=use_pos)

    if save_stuff:
        print('Saving the model')
        torch.save(model.state_dict(), 'model/emb-100.pth')
        print('Saving the vocab')
        vocab.dump_data(Path('model/glove-vocab.pkl'))
    return model, vocab


def fine_tune(vocab: dataset.Vocabulary, model: lstm.NerModel):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    trainset = dataset.NerDataset(path=Path('data/train.tsv'), vocab=vocab)
    devset = dataset.NerDataset(path=Path('data/dev.tsv'), vocab=vocab)

    # dataloaders
    trainloader, devloader = dataset.get_dataloaders(trainset,
                                                     devset,
                                                     batch_size_train=128,
                                                     batch_size_dev=1024)

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


model, vocab = build_pretrained_embeddings(save_stuff=False,
                                           freeze=False,
                                           double_linear=False,
                                           use_pos=False,
                                           hidden_size=100)

fine_tune(vocab, model)
