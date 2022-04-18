"""Loads and builds the model
"""

from typing import List, Optional
import numpy as np
import torch
from model import Model  # type: ignore
import torchcrf
from . import config, dataset, lstm


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

        self.vocab: dataset.Vocabulary = dataset.Vocabulary(path=config.MODEL /
                                                            'vocab-glove.pkl')

        self.model: lstm.NerModel = lstm.NerModel(n_classes=13,
                                                  embedding_dim=100,
                                                  vocab_size=len(self.vocab),
                                                  padding_idx=self.vocab.pad,
                                                  hidden_size=100,
                                                  bidirectional=True,
                                                  pretrained_emb=None).to(
                                                      self.device)

        self.crf: torchcrf.CRF = torchcrf.CRF(num_tags=14,
                                              batch_first=True).to(self.device)
        self.model.load_state_dict(
            torch.load(config.MODEL / '7572-stacked-100h-crf.pth',
                       map_location=self.device))

        self.crf.load_state_dict(
            torch.load(config.MODEL / 'crf-7572.pth', map_location=self.device))

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:

        # return lstm.predict_char(self.model, self.vocab, self.char_vocab,
        #  tokens, self.device)
        return lstm.predict(self.model,
                            self.vocab,
                            tokens,
                            self.device,
                            window_size=100,
                            crf=self.crf)
