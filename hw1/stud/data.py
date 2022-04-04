"""
Contains classes and functions for the management of the data
"""

import torch
from tqdm import tqdm
from typing import  List, Tuple, Dict
from collections import Counter
from pathlib import Path
import pickle


class Vocabulary():
    """ Implements a vocabulary of both words and labels.
        Automatically adds '<unk>' and '<pad>' word types.
    """

    def __init__(self,
                 sentences: List[Tuple[List[str], List[str]]] = None,
                 threshold: int = 1,
                 path: str = None):
        """Initialize the vocabulary from a dataset

        Args:
            sentences (List[Tuple[List[str], List[str]]]):
                The dataset as a list of tuples.
                Each tuple contains two lists: the words of a sentence
                and the corresponding labels

            threshold (int, optional):
                Number of appearances needed for a word to
                be inserted in the dictionary. Defaults to 1.
        """

        if path is None and sentences is None:
            raise ValueError('To build a vocabulary you must give either a '
                             'list of sentences or a path to a dump.')

        self.threshold: int = threshold

        # load from dump
        if path is not None:
            self.counts, self.lcounts = self.load_counts(path)
        # build from sentences
        else:
            self.counts: Counter = Counter()
            self.lcounts: Counter  = Counter()
            for sentence, labels in sentences:
                for word, label in zip(sentence, labels):
                    self.counts[word] += 1
                    self.lcounts[label] += 1

                    if label == 'id':
                        print(f'{sentence=}')
                        print(f'{labels=}')

        # unk and pad symbols
        self.unk_symbol = '<unk>'
        self.pad_symbol = '<pad>'
        self.pad_label = 'PAD'

        # word vocabularies
        self.itos: List[str] = sorted(list(
            filter(lambda x: self.counts[x] >= threshold, self.counts.keys())) +
            [self.unk_symbol, self.pad_symbol])
        self.stoi: Dict[str, int] = {
            s: i for i, s in enumerate(self.itos)}

        # label vocabularies
        self.ltos: List[str] = sorted(list(self.lcounts.keys()) + ['PAD'])
        self.stol: Dict[str, int] = {s: i for i, s in enumerate(self.ltos)}

        self.unk: int = self.stoi[self.unk_symbol]
        self.pad: int = self.stoi[self.pad_symbol]
        self.pad_label_id: int = self.stol[self.pad_label]
        self.n_labels: int = len(self.ltos)

    def __contains__(self, word: str):
        return word in self.stoi

    def __len__(self):
        return len(self.itos)

    def load_counts(self, path: str):
        """Loads self.counts and self.lcounts from a previous dump

        Args:
            path (str): the path to the pickle dump

        Returns:
            counts, lcounts: Counters of words and labels
        """
        with open(path, 'rb') as f:
            counts, lcounts = pickle.load(f)
        return counts, lcounts

    def dump_counts(self, path: str):
        """Dumps self.counts and self.lcounts as pickle objects

        Args:
            path (str): the path of the dump
        """
        with open(path, 'wb') as f:
            pickle.dump((self.counts, self.lcounts), f)

    def get_word(self, idx: int) -> str:
        """Return the word at a given index

        Args:
            idx (int): the index of a word

        Returns:
            str: the word corresponding to the given index
        """
        return self.itos[idx]

    def get_word_id(self, word: str) -> int:
        """Get the index of a given word

        Args:
            word (str): The word to retrieve the index of

        Returns:
            int: Index of the word if present, otherwise the index of '<unk>'
        """
        return self.stoi[word] if word in self.stoi else self.unk

    def get_label(self, idx: int) -> str:
        """Get a label name from its index

        Args:
            id (int): the index of a label

        Returns:
            str: the correpsonding label name
        """
        return self.ltos[idx]

    def get_label_id(self, label: str) -> int:
        """Get the id of a label

        Args:
            label (str): the name of the label

        Returns:
            int: the corresponding index
        """
        return self.stol[label]

    def __getitem__(self, idx: int or str) -> str or int:
        if isinstance(idx, str):
            return self.get_word_id(idx)
        elif isinstance(idx, int):
            return self.get_word(idx)
        raise NotImplementedError()


class NerDataset(torch.utils.data.Dataset):
    """Represent a Named Entity Recognition Dataset
    """

    def __init__(self,
                 path: Path = Path('../../data/train.tsv'),
                 vocab: Vocabulary = None,
                 threshold: int = 2,
                 window_size: int = 100,
                 window_shift: int = None):
        """
        Build a Named Entity Recognition dataset from a .tsv file,
        which loads data as fixed-size windows

        Args:
            path (Path, optional):
                Path of the .tsv dataset file.
                Defaults to Path('../../data/train.tsv').
            vocab (Vocabulary, optional):
                Vocabulary to index the data. If none, build one.
                Defaults to None.
            threshold (int, optional):
                If vocab is None, threshold for the vocabulary. Defaults to 1.
            window_size (int, optional):
                Size of the windows. Defaults to 5.
            window_shift (int, optional):
                Shift of the windows. Defaults to None.
        """
        super().__init__()
        if vocab is None and 'train' not in str(path):
            raise ValueError('Careful, you are trying to build a vocabulary'
                             'on something that is not the training data')
        self.path: Path = path
        self.sentences: List[Tuple[List[str]]] = self.load_data(self.path)
        self.vocab: Vocabulary = (
            Vocabulary(self.sentences, threshold=threshold)
            if vocab is None
            else vocab)
        self.indexed_data: List[Tuple[List[int], List[int]]] = self.index_data()
        self.window_size: int = window_size
        self.window_shift: int = window_shift or window_size
        if (not 0 <= self.window_shift <= self.window_size or
            self.window_size <= 0):
            raise ValueError('Window shift must be equal or less than window'
                             'size, both must be positive')
        self.windows: List[Tuple[torch.Tensor]] = self.build_windows()

    def load_data(self, path: Path):
        """Loads the dataset from file

        Args:
            path (Path): path of the .tsv dataset

        Returns:
            sentences (List[Tuple[List[str], List[str]]]):
                a list of sentences. Each sentences is a tuple made of:
                - list of words in the sentence
                - list of labels of the words
        """
        words = []
        labels = []
        sentences = []
        with open (path, 'r', encoding='utf-8') as f:
            # strip lines, remove empty ones and the first
            lines = list(filter(None, map(str.strip, f.readlines())))[1:]
            for line in tqdm(lines, desc='Reading data', total=len(lines)):
                line: List[str] = line.split('\t')
                if line[0] == '#':
                    sentences.append((words, labels))
                    words = []
                    labels = []
                else:
                    words.append(line[0])
                    labels.append(line[1])

            # last sentence
            sentences.append((words, labels))
        return sentences

    def index_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        Builds self.indexed_data transforming both
        words and labels in integers

        Args:
            vocab (Vocabulary):
                the vocabulary to use to convert words to indices
        """
        data = list(map(
            lambda sentence: (
                [self.vocab[w] for w in sentence[0]],
                [self.vocab.get_label_id(l) for l in sentence[1]]
            ),
            self.sentences
        ))
        return data

    def build_windows(self) -> List[Tuple[List[int], List[int]]]:
        """Builds fixed-size windows from the indexed data

        Returns:
            List[Tuple[Tensor, Tensor]]: List of fixed-size windows
        """
        windows: List[Tuple[List[int], List[int]]] = []
        for word_ids, label_ids in self.indexed_data:
            start = 0
            while start < len(word_ids):
                # generate window
                word_window = word_ids[start: start+self.window_size]
                label_window = label_ids[start: start+self.window_size]
                # pad
                word_window += (
                    [self.vocab.pad] *
                    (self.window_size - len(word_window)))
                label_window += (
                    [self.vocab.get_label_id(self.vocab.pad_label)] *
                    (self.window_size - len(label_window)))
                # append
                windows.append(
                    (torch.tensor(word_window),
                    torch.tensor(label_window)))
                start += self.window_shift
        return windows

    # def get_window(self, idx):
    #     return self.windows[idx]

    # def get_window_count(self):
    #     return len(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def get_dataloaders(trainset: NerDataset,
                    devset: NerDataset,
                    batch_size_train: int = 128,
                    batch_size_dev: int = 256):
    """Return the dataloaders from the given datasets

    Args:
        trainset (NerDataset): dataset for training
        devset (NerDataset): dataset for evaluation
        batch_size (int, optional): Batch size. Defaults to 64.

    Returns:
        Tuple[Dataloader, Dataloader]: the dataloaders
    """
    return (
        torch.utils.data.DataLoader(trainset,
                                    batch_size=batch_size_train,
                                    shuffle=True),
        torch.utils.data.DataLoader(devset,
                                    batch_size=batch_size_dev),
    )
