"""
Contains classes and functions for the management of the data, such as custom
datasets which extend torch.util.data.Dataset, and custom vocabularies.
"""

from typing import Dict, List, Optional, Set, Tuple, Union, Iterable
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import torch.utils.data
import stanza
from . import config


class PosVocabulary:
    """
        Implements a vocabulary for the POS tags
    """

    def __init__(self):
        weights = [
            -1.5842, -1.3862, -1.1882, -0.9901, -0.7921, -0.5941, -0.3961,
            -0.1980, 0.0000, 0.1980, 0.3961, 0.5941, 0.7921, 0.9901, 1.1882,
            1.3862, 1.5842
        ]

        self.index_to_label: List[str] = [
            'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
            'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
        ]
        self.label_to_index: Dict[str, int] = {
            s: i for i, s in enumerate(self.index_to_label)
        }

        self.label_to_weight: Dict[str, float] = {
            l: weights[i] for i, l in enumerate(self.label_to_index)
        }

    def __len__(self):
        return len(self.index_to_label)

    def __getitem__(self, idx: Union[int, str]) -> Union[str, float]:
        if isinstance(idx, str):
            return self.label_to_weight[idx]
        elif isinstance(idx, int):
            return self.index_to_label[idx]
        raise NotImplementedError()


class CharVocabulary:
    """
        Implements a vocabulary for the characters
    """

    def __init__(self,
                 sentences: Optional[List[Tuple[List[str], List[str]]]] = None,
                 path: Optional[Path] = None,
                 threshold: int = 2):
        """
        Initializes a character vocabulary, either from a list of sentences,
        or by loading it from a path

        Args:
            sentences (Optional[List[Tuple[List[str], List[str]]]], optional):
                A list of tuples in the form <list of words, list of labels>
                to initialize the vocabulary from. Defaults to None.
            path (Optional[Path], optional):
                The path from which to load the vocabulary. Defaults to None.
            threshold (int, optional):
                If sentences is specified, the number of occurrences needed
                for a character to be included. Defaults to 2.

        Raises:
            ValueError:
                Either a list of sentences or a path have to be specified
        """

        self.unk_label: str = '<unk>'
        self.pad_label: str = '<pad>'

        if sentences is None and path is None:
            raise ValueError('You have to provide either sentences '
                             'or a path to load the vocabulary from')

        # build vocabulary from sentences
        if sentences is not None:
            # chars = set()
            chars = Counter()
            for s in sentences:
                for w in s[0]:
                    for c in w:
                        chars[c] += 1
            #         chars = chars | set(w)
            # chars = chars | {self.pad_label, self.unk_label}
            # chars = sorted(list(chars))
            self.itoc: List[str] = sorted(
                list(filter(lambda x: chars[x] >= threshold, chars.keys())) +
                [self.unk_label, self.pad_label])

            # self.itoc: List[str] = chars

        # load from file
        elif path is not None:
            self.itoc: List[str] = self.load_data(path)

        self.ctoi: Dict[str, int] = {c: i for i, c in enumerate(self.itoc)}

        self.unk: int = self.ctoi[self.unk_label]
        self.pad: int = self.ctoi[self.pad_label]

    def __len__(self):
        return len(self.itoc)

    def get_char_id(self, char: str) -> int:
        """Returns the index for the given char

        Args:
            char (str): the character as a string

        Returns:
            int: the corresponding index
        """
        return self.ctoi[char] if char in self.ctoi else self.unk

    def get_char(self, idx: int) -> str:
        """Returns the character corresponding to the given index.

        Args:
            idx (int): the index

        Returns:
            str: the corresponding character
        """
        return self.itoc[idx]

    def __getitem__(self, idx: Union[int, str]) -> Union[str, float]:
        if isinstance(idx, str):
            return self.get_char_id(idx)
        elif isinstance(idx, int):
            return self.get_char(idx)
        raise NotImplementedError()

    def dump_data(self, path: Path):
        """Dumps itself as a binary at the given path

        Args:
            path (Path): the path to dump to
        """
        with open(path, 'wb') as f:
            pickle.dump(self.itoc, f)

    def load_data(self, path: Path) -> List[str]:
        """Loads a vocabulary from the given path

        Args:
            path (Path): the path to load from

        Returns:
            List[str]: the loaded index-to-character list
        """
        with open(path, 'rb') as f:
            itoc = pickle.load(f)
        return itoc


class Vocabulary:
    """ Implements a vocabulary of both words and labels.
        Automatically adds '<unk>' and '<pad>' word types.
    """

    def __init__(self,
                 sentences: Optional[List[Tuple[List[str], List[str]]]] = None,
                 threshold: int = 1,
                 path: Optional[Path] = None,
                 premade: Optional[List[str]] = None):
        """Initialize the vocabulary from a dataset

        Args:
            sentences (Optional[List[Tuple[List[str], List[str]]]]):
                The dataset as a list of tuples.
                Each tuple contains two lists: the words of a sentence
                and the corresponding labels

            threshold (int, optional):
                Number of appearances needed for a word to
                be inserted in the dictionary. Defaults to 1.

            path (Path, optional):
                The path to load the vocabulary from. Defaults to None

            premade: (List[str], optional):
                A list of words to initialize the vocabulary from.
                Defaults to None.

        Raises:
            ValueError:
                either a path, a list of sentences, or a premade list
                of words have to be specified.
        """

        if path is None and sentences is None and premade is None:
            raise ValueError('To build a vocabulary you must give either a '
                             'list of sentences or a path to a dump.')

        self.threshold: int = threshold

        # unk and pad symbols
        self.unk_symbol: str = '<unk>'
        self.pad_symbol: str = '<pad>'
        self.pad_label: str = 'PAD'

        # data containers
        self.counts: Counter
        self.lcounts: Counter
        self.itos: List[str]
        self.stoi: Dict[str, int]
        self.ltos: List[str]
        self.stol: Dict[str, int]

        # build from pretrained embedding matrix
        if premade is not None:
            self.threshold = 1
            self.itos = list(premade.copy())
            if self.pad_symbol not in self.itos:
                self.itos.insert(0, self.pad_symbol)
            if self.unk_symbol not in self.itos:
                self.itos.insert(1, self.unk_symbol)
            self.ltos = [
                'B-CORP', 'B-CW', 'B-GRP', 'B-LOC', 'B-PER', 'B-PROD', 'I-CORP',
                'I-CW', 'I-GRP', 'I-LOC', 'I-PER', 'I-PROD', 'O', 'PAD'
            ]

        # load from dump
        elif path is not None:
            self.itos, self.ltos = self.load_data(path)
            # self.counts, self.lcounts = self.load_counts(path)

        # build from sentences
        elif sentences is not None:
            self.counts = Counter()
            self.lcounts = Counter()
            for sentence, labels in sentences:
                for word, label in zip(sentence, labels):
                    self.counts[word] += 1
                    self.lcounts[label] += 1
                    # if label == 'id':
                    #     print(f'{sentence=}')
                    #     print(f'{labels=}')
            self.itos = sorted(
                list(
                    filter(lambda x: self.counts[x] >= threshold,
                           self.counts.keys())) +
                [self.unk_symbol, self.pad_symbol])
            # label vocabularies
            self.ltos = sorted(list(self.lcounts.keys()) + ['PAD'])

        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.stol = {s: i for i, s in enumerate(self.ltos)}

        self.unk: int = self.stoi[self.unk_symbol]
        self.pad: int = self.stoi[self.pad_symbol]
        self.pad_label_id: int = self.stol[self.pad_label]
        self.n_labels: int = len(self.ltos)

    def replace_labels(self, new_labels: Set[str]):
        self.ltos = sorted(list(new_labels) + [self.pad_label])
        self.stol = {s: i for i, s in enumerate(self.ltos)}
        self.pad_label_id = self.ltos.index(self.pad_label)
        self.n_labels = len(self.ltos)

    def __contains__(self, word: str):
        return word in self.stoi

    def __len__(self):
        return len(self.itos)

    def load_counts(self, path: Path):
        """Loads self.counts and self.lcounts from a previous dump

        Args:
            path (str): the path to the pickle dump

        Returns:
            counts, lcounts: Counters of words and labels
        """
        with open(path, 'rb') as f:
            counts, lcounts = pickle.load(f)
        return counts, lcounts

    def dump_counts(self, path: Path):
        """Dumps self.counts and self.lcounts as pickle objects

        Args:
            path (str): the path of the dump
        """
        with open(path, 'wb') as f:
            pickle.dump((self.counts, self.lcounts), f)

    def dump_data(self, path: Path):
        """Dumps itself as a binary at the given path

        Args:
            path (Path): the path to dump to
        """
        with open(path, 'wb') as f:
            pickle.dump((self.itos, self.ltos), f)

    def load_data(self, path: Path) -> Tuple[List[str], List[str]]:
        """Loads a vocabulary from the given path

        Args:
            path (Path): the path to load from

        Returns:
            Tuple[List[str], List[str]]: lists index-to-word and index-to-label
        """
        with open(path, 'rb') as f:
            itos, ltos = pickle.load(f)
        return itos, ltos

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

    def __getitem__(self, idx: Union[int, str]) -> Union[str, int]:
        if isinstance(idx, str):
            return self.get_word_id(idx)
        elif isinstance(idx, int):
            return self.get_word(idx)
        raise NotImplementedError()

    def decode(self, sentence: Iterable[int]) -> str:
        """Decodes a sentences to human-readable form

        Args:
            sentence (Iterable[int]): the sentence as an iterable of indices

        Returns:
            str: the human-readable sentence
        """
        return ' '.join([self.get_word(i) for i in sentence])

    def encode(self, sentence: str, lower: bool = True) -> List[int]:
        """Encodes to indices a human-readable sentence

        Args:
            sentence (str): the sentence as a string
            lower (bool, optional):
                Whether to make it lowercase before encoding. Defaults to True.

        Returns:
            List[int]: a list of indices that encodes the sentence
        """
        return [
            self.get_word_id(i.lower() if lower else i)
            for i in sentence.strip().split()
        ]


class NerDataset(torch.utils.data.Dataset):
    """Represent a Named Entity Recognition Dataset
    """

    def __init__(self,
                 path: Path = config.TRAIN,
                 vocab: Optional[Vocabulary] = None,
                 threshold: int = 2,
                 window_size: int = 100,
                 window_shift: Optional[int] = None,
                 is_conll: bool = False,
                 lower: bool = False,
                 replace_labels: bool = False):
        """
        Build a Named Entity Recognition dataset from a .tsv file,
        which loads data as fixed-size windows

        Args:
            path (Path, optional):
                Path of the .tsv dataset file.
                Defaults to Path('data/train.tsv').
            vocab (Vocabulary, optional):
                Vocabulary to index the data. If none, build one.
                Defaults to None.
            threshold (int, optional):
                If vocab is None, threshold for the vocabulary. Defaults to 1.
            window_size (int, optional):
                Size of the windows. Defaults to 5.
            window_shift (int, optional):
                Shift of the windows. Defaults to None.
            is_conll (bool, optional):
                True if the dataset is Conll-2003. Defaults to False.
            lower (bool, optional):
                Whether to store every sentence as lowercase. Defaults to False.

        Raises:
            ValueError:
                if specified, window_shift has to be equal or smaller than
                window_size, and both must be positive
            ValueError:
                cannot build a vocabulary from a file that does not contain the
                word "train" in its name.
        """
        super().__init__()
        if vocab is None and 'train' not in str(path):
            raise ValueError('Careful, you are trying to build a vocabulary'
                             'on something that is not the training data')
        self.replace_labels: bool = replace_labels
        self.path: Path = path
        self.is_conll: bool = is_conll
        self.lower: bool = lower
        self.threshold = threshold
        self.window_size: int = window_size
        self.window_shift: int = window_shift or self.window_size
        if (not 0 <= self.window_shift <= self.window_size or
                self.window_size <= 0):
            raise ValueError('Window shift must be equal or less than window'
                             'size, both must be positive')

        self.build_dataset(vocab)

    def build_dataset(self, vocab):
        self.sentences: List[Tuple[List[str], List[str]]] = (self.load_conll(
            self.path, self.lower) if self.is_conll else self.load_data(
                self.path))

        self.vocab: Vocabulary = (Vocabulary(self.sentences,
                                             threshold=self.threshold)
                                  if vocab is None else vocab)
        if self.replace_labels:
            labels: Set[str] = {
                l for sentence in self.sentences for l in sentence[1]
            }
            self.vocab.replace_labels(labels)

        # this should become a subclass ConllNerDatatset
        if self.is_conll:
            labels: set[str] = {
                l for sentence in self.sentences for l in sentence[1]
            }
            self.vocab.ltos = sorted(list(labels) + ['PAD'])
            self.vocab.stol = {s: i for i, s in enumerate(self.vocab.ltos)}
            self.vocab.pad_label_id = 9
            self.vocab.n_labels = 9
        ###
        self.indexed_data: List[Tuple[List[int], List[int]]] = self.index_data()
        self.windows: List[Tuple[torch.Tensor,
                                 torch.Tensor]] = self.build_windows()

    def load_conll(self, path: Path,
                   lower: bool) -> List[Tuple[List[str], List[str]]]:
        words = []
        labels = []
        sentences = []

        with open(path, 'rt', encoding='utf-8') as f:
            lines = list(map(str.strip, f.readlines()))[2:]
            for line_str in tqdm(lines, desc='Reading data', total=len(lines)):

                if not line_str and words:
                    sentences.append((words, labels))
                    words = []
                    labels = []
                else:
                    line: List[str] = line_str.split()
                    words.append(line[0].lower() if lower else line[0])
                    labels.append(line[3])
            if words:
                sentences.append((words, labels))

        return sentences

    def load_data(self, path: Path) -> List[Tuple[List[str], List[str]]]:
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
        with open(path, 'r', encoding='utf-8') as f:
            # strip lines, remove empty ones and the first
            lines = list(filter(None, map(str.strip, f.readlines())))[1:]
            for line_str in tqdm(lines, desc='Reading data', total=len(lines)):
                line: List[str] = line_str.split('\t')
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
        data = list(
            map(
                lambda sentence:
                ([self.vocab.get_word_id(w) for w in sentence[0]],
                 [self.vocab.get_label_id(l) for l in sentence[1]]),
                self.sentences))
        return data

    def build_windows(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Builds fixed-size windows from the indexed data

        Returns:
            List[Tuple[Tensor, Tensor]]: List of fixed-size windows
        """
        windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for word_ids, label_ids in self.indexed_data:
            start = 0
            while start < len(word_ids):
                # generate window
                word_window = word_ids[start:start + self.window_size]
                label_window = label_ids[start:start + self.window_size]
                # pad
                word_window += ([self.vocab.pad] *
                                (self.window_size - len(word_window)))
                label_window += (
                    [self.vocab.get_label_id(self.vocab.pad_label)] *
                    (self.window_size - len(label_window)))
                # append
                windows.append(
                    (torch.tensor(word_window), torch.tensor(label_window)))
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

    def human(self, idx: int):
        return (' '.join(self.sentences[idx][0]),
                ' '.join(self.sentences[idx][1]))


class NerDatasetWnut(NerDataset):
    """Represent a Named Entity Recognition Dataset
    """

    def __init__(self,
                 path: Path = config.TRAIN,
                 vocab: Optional[Vocabulary] = None,
                 lower: bool = True,
                 threshold: int = 2):

        super().__init__(path=path,
                         vocab=vocab,
                         replace_labels=True,
                         lower=lower,
                         threshold=threshold)

    def load_data(self, path: Path) -> List[Tuple[List[str], List[str]]]:
        words = []
        labels = []
        sentences = []
        with open(path, 'r', encoding='utf-8') as f:
            # strip lines, remove empty ones and the first
            lines = list(map(str.strip, f.readlines()))
            for line_str in tqdm(lines, desc='Reading data', total=len(lines)):
                if line_str == '' and words and labels:
                    sentences.append((words, labels))
                    words = []
                    labels = []
                else:
                    line: List[str] = line_str.split('\t')
                    words.append(line[0].lower() if self.lower else line[0])
                    labels.append(line[1])

            if words and labels:
                sentences.append((words, labels))
        return sentences


class NerDatasetChar(NerDataset):
    """
    Extends a NerDatasets, but stores the characters for each word as well
    """

    def __init__(
        self,
        path: Path = config.TRAIN,
        vocab: Optional[Vocabulary] = None,
        char_vocab: Optional[CharVocabulary] = None,
        threshold: int = 2,
        window_size: int = 100,
        window_shift: Optional[int] = None,
    ):
        """Initialize a NerDatsetChar

        Args:
            path (Path, optional):
                The path to load the data from.
                Defaults to Path('data/train.tsv').
            vocab (Vocabulary, optional):
                Vocabulary to index the data. If none, build one.
                Defaults to None.
            char_vocab (Optional[CharVocabulary], optional):
                Same as vocab, but for the charcter vocabulary.
                Defaults to None.
            threshold (int, optional):
                If vocab is None, threshold for the vocabulary. Defaults to 1.
            window_size (int, optional):
                Size of the windows. Defaults to 5.
            window_shift (int, optional):
                Shift of the windows. Defaults to None.

        Raises:
            ValueError:
                if specified, window_shift has to be equal or smaller than
                window_size, and both must be positive
            ValueError:
                cannot build a vocabulary from a file that does not contain the
                word "train" in its name.
        """

        if char_vocab is None and 'train' not in str(path):
            raise ValueError(
                'Careful, you are trying to build a character vocabulary'
                'on something that is not the training data')
        self.indexed_data: List[Tuple[List[int], List[int], List[List[int]]]]
        self.max_word_len: int

        self.char_vocab: Optional[CharVocabulary] = char_vocab  # type: ignore
        super().__init__(path, vocab, threshold, window_size, window_shift)

    def index_data(self) -> List[Tuple[List[int], List[int], List[List[int]]]]:
        """
        Builds self.indexed_data transforming words, labels and characters
        into integers

        Returns: List[Tuple[List[int], List[int], List[List[int]]]]:
            A list of sentences. Each sentence is a tuple of a list of word
            indices, a list of label indices, and a list of lists of character
            indices.
        """
        self.char_vocab: CharVocabulary = CharVocabulary(
            self.sentences) if self.char_vocab is None else self.char_vocab
        data = list(
            map(
                lambda sentence:
                ([self.vocab.get_word_id(w) for w in sentence[0]], [
                    self.vocab.get_label_id(l) for l in sentence[1]
                ], [[self.char_vocab.get_char_id(c)
                     for c in word]
                    for word in sentence[0]]), self.sentences))
        return data

    def _pad_char_sequence(self, chars: List[int], total: int) -> List[int]:
        if len(chars) > total:
            return chars[:total]
        return chars + [self.char_vocab.pad] * (total - len(chars))

    def build_windows(
            self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Builds fixed-size windows from the indexed data

        Returns:
            List[Tuple[Tensor, Tensor, Tensor]]: List of fixed-size windows
        """
        windows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # self.max_word_len = max(len(w[0]) for w in self.sentences)
        self.max_word_len = 15
        for word_ids, label_ids, word_chars in self.indexed_data:
            start = 0
            while start < len(word_ids):
                # generate window
                word_window = word_ids[start:start + self.window_size]
                label_window = label_ids[start:start + self.window_size]
                chars_window = [
                    self._pad_char_sequence(chars, total=self.max_word_len)
                    for chars in word_chars[start:start + self.window_size]
                ]
                # pad
                word_window += ([self.vocab.pad] *
                                (self.window_size - len(word_window)))
                label_window += (
                    [self.vocab.get_label_id(self.vocab.pad_label)] *
                    (self.window_size - len(label_window)))
                chars_window += (
                    [self._pad_char_sequence([], self.max_word_len)] *
                    (self.window_size - len(chars_window)))
                # append
                windows.append(
                    (torch.tensor(word_window), torch.tensor(label_window),
                     torch.tensor(chars_window)))
                start += self.window_shift
        return windows


class NerDatasetPos(NerDataset):
    """
    Extends a NerDatasets, but computes and mantains pos tags for each word,
    using the stanza library
    """

    def __init__(
        self,
        path: Path = config.TRAIN,
        pos_tagger: Optional[stanza.Pipeline] = None,
        vocab: Optional[Vocabulary] = None,
        threshold: int = 2,
        window_size: int = 100,
        window_shift: Optional[int] = None,
    ):
        """Initialize a NerDatsetPos

        Args:
            path (Path, optional):
                The path to load the data from.
                Defaults to Path('data/train.tsv').
            pos_tagger (stanza.Pipeline, optional):
                The pos tagger to use. Defaults to None.
            vocab (Vocabulary, optional):
                Vocabulary to index the data. If none, build one.
                Defaults to None.
            threshold (int, optional):
                If vocab is None, threshold for the vocabulary. Defaults to 1.
            window_size (int, optional):
                Size of the windows. Defaults to 5.
            window_shift (int, optional):
                Shift of the windows. Defaults to None.

        Raises:
            ValueError:
                if specified, window_shift has to be equal or smaller than
                window_size, and both must be positive
            ValueError:
                cannot build a vocabulary from a file that does not contain the
                word "train" in its name.
        """

        if pos_tagger is not None:
            self.pos_tagger: stanza.Pipeline = pos_tagger
        else:
            stanza.download('en', logging_level='FATAL')
            self.pos_tagger: stanza.Pipeline = stanza.Pipeline(
                'en',
                processors='tokenize, pos',
                tokenize_pretokenized=True,
                logging_level='FATAL')

        self.pos_vocab: PosVocabulary = PosVocabulary()

        super().__init__(path, vocab, threshold, window_size, window_shift)

    def build_dataset(self, vocab):
        """Builds a list of sentences and a vocabularyfrom the given file

        Args:
            vocab (_type_): the vocab given in the arguments of __init__
        """
        sentences_nopos: List[Tuple[List[str], List[str]]] = (self.load_conll(
            self.path, self.lower) if self.is_conll else self.load_data(
                self.path))

        self.vocab: Vocabulary = (Vocabulary(sentences_nopos,
                                             threshold=self.threshold)
                                  if vocab is None else vocab)

        print('Getting POS tags...')
        tokens: List[List[str]] = list(map(lambda x: x[0], sentences_nopos))

        pos_tags: List[List[str]] = self.pos_tagger(tokens).get(  # type: ignore
            'pos', as_sentences=True)

        self.sentences: List[Tuple[List[str], List[str], List[str]]] = [
            (sentences_nopos[i][0], sentences_nopos[i][1], pos_tags[i])
            for i in range(len(sentences_nopos))
        ]
        print('Built dataset!')

        self.indexed_data: List[Tuple[List[int], List[int],
                                      List[int]]] = self.index_data()
        self.windows: List[Tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor]] = self.build_windows()

    def index_data(self) -> List[Tuple[List[int], List[int], List[int]]]:
        data = list(
            map(
                lambda sentence:
                ([self.vocab.get_word_id(w) for w in sentence[0]
                 ], [self.vocab.get_label_id(l) for l in sentence[1]],
                 [int(self.pos_vocab.label_to_index[p]) for p in sentence[2]]),
                self.sentences))
        return data

    def build_windows(
            self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Builds fixed-size windows from the indexed data

        Returns:
            List[Tuple[Tensor, Tensor]]: List of fixed-size windows
        """
        windows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for word_ids, label_ids, pos_ids in self.indexed_data:
            start = 0
            while start < len(word_ids):
                # generate window
                word_window = word_ids[start:start + self.window_size]
                label_window = label_ids[start:start + self.window_size]
                pos_window = pos_ids[start:start + self.window_size]
                # pad
                word_window += ([self.vocab.pad] *
                                (self.window_size - len(word_window)))
                label_window += (
                    [self.vocab.get_label_id(self.vocab.pad_label)] *
                    (self.window_size - len(label_window)))
                pos_window += ([int(self.pos_vocab['PUNCT'])] *
                               (self.window_size - len(pos_window)))
                # append
                windows.append(
                    (torch.tensor(word_window), torch.tensor(label_window),
                     torch.tensor(pos_window)))
                start += self.window_shift
        return windows

    def human(self, idx: int) -> Tuple[str, str, str]:
        """Returns the sentence at a given index in a human-readable format

        Args:
            idx (int): the index of the sentence

        Returns:
            Tuple[str, str, str]: the human-readable sentence
        """
        return (' '.join(self.sentences[idx][0]),
                ' '.join(self.sentences[idx][1]),
                ' '.join(self.sentences[idx][2]))


def get_dataloaders(
    vocab: Optional[Vocabulary] = None,
    trainset: Optional[NerDataset] = None,
    devset: Optional[NerDataset] = None,
    use_pos: bool = False,
    window_size: int = 100,
    batch_size_train: int = 128,
    batch_size_dev: int = 256
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders

    Args:
        vocab (Optional[Vocabulary], optional):
            The vocabulary to use to build the datsets. Defaults to None.
        trainset (Optional[NerDataset], optional):
            The dataset for training. Defaults to None.
        devset (Optional[NerDataset], optional):
            The datset for evaluation. Defaults to None.
        use_pos (bool, optional):
            Wheter to use a NerDatsetPos. Defaults to False.
        window_size (int, optional):
            Window size of the vocabulary. Defaults to 100.
        batch_size_train (int, optional):
            Batch size for the dataloader for training. Defaults to 128.
        batch_size_dev (int, optional):
            Batch size for the dataloader for evaluation. Defaults to 256.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            Trainloader and Testloader
    """
    if trainset is None or devset is None:
        trainset, devset = get_datasets(vocab,
                                        use_pos=use_pos,
                                        window_size=window_size)
    return (
        torch.utils.data.DataLoader(trainset,
                                    batch_size=batch_size_train,
                                    shuffle=True),
        torch.utils.data.DataLoader(devset, batch_size=batch_size_dev),
    )


def get_datasets(vocab: Optional[Vocabulary] = None,
                 use_pos: bool = False,
                 window_size: int = 100) -> Tuple[NerDataset, NerDataset]:
    """Returns datsets for training and testing

    Args:
        vocab (Optional[Vocabulary], optional):
            Vocabulary to use to build the datsets. Defaults to None.
        use_pos (bool, optional):
            Whether to build NerDatasetPos. Defaults to False.
        window_size (int, optional):
            Window size to use. Defaults to 100.

    Returns:
        Tuple[NerDataset, NerDataset]: trainset and testset
    """
    if use_pos:
        trainset: NerDataset = NerDatasetPos(config.TRAIN,
                                             vocab=vocab,
                                             window_size=window_size)
        devset: NerDataset = NerDatasetPos(config.DEV,
                                           vocab=trainset.vocab,
                                           window_size=window_size)
        return trainset, devset

    trainset: NerDataset = NerDataset(config.TRAIN, vocab=vocab)
    devset: NerDataset = NerDataset(config.DEV, vocab=trainset.vocab)
    return trainset, devset
