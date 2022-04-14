"""Loads and builds the model
"""

from typing import List, Optional
import numpy as np
import torch
from model import Model  # type: ignore
from . import nerdtagger, config, dataset, lstm


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()

    print('building model')
    return StudentModel(device=device)


class RandomBaseline(Model):
    options = [(3111, 'B-CORP'), (3752, 'B-CW'),
               (3571, 'B-GRP'), (4799, 'B-LOC'), (5397, 'B-PER'),
               (2923, 'B-PROD'), (3111, 'I-CORP'), (6030, 'I-CW'),
               (6467, 'I-GRP'), (2751, 'I-LOC'), (6141, 'I-PER'),
               (1800, 'I-PROD'), (203394, 'O')]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[
            str(np.random.choice(self._options, 1, p=self._weights)[0])
            for _x in x
        ]
                for x in tokens]


class StudentModel(Model):

    def __init__(self, device: Optional[str] = None):
        self.device: torch.device = config.DEVICE
        if device:
            self.device = torch.device(device)

        self.char_vocab = dataset.CharVocabulary(path=config.MODEL /
                                                 'vocab-char.pkl')
        self.vocab = dataset.Vocabulary(path=config.MODEL / 'vocab-glove.pkl')

        self.model = lstm.NerModelChar(
            n_classes=13,
            embedding_dim=100,
            char_embedding_dim=50,
            char_vocab=self.char_vocab,
            vocab_size=len(self.vocab),
            padding_idx=self.vocab.pad,
            hidden_size=100,
            char_hidden_size=50,
            bidirectional=True,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(config.MODEL / '9646-charembedding-glove.pth',
                       map_location=self.device))

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:

        return lstm.predict_char(self.model, self.vocab, self.char_vocab,
                                 tokens, self.device)
        # return lstm.predict(self.model, self.vocab, tokens, self.device)
