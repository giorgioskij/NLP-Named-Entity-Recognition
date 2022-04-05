"""Loads and builds the model
"""

import numpy as np
from typing import List

from model import Model  # type: ignore

###
from . import lstm
from . import dataset
import torch
from typing import Optional
###


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()

    print('building model')

    return StudentModel(device=device)


class RandomBaseline(Model):
    options = [(3111, 'B-CORP'), (3752, 'B-CW'), (3571, 'B-GRP'),
               (4799, 'B-LOC'), (5397, 'B-PER'), (2923, 'B-PROD'),
               (3111, 'I-CORP'), (6030, 'I-CW'), (6467, 'I-GRP'),
               (2751, 'I-LOC'), (6141, 'I-PER'), (1800, 'I-PROD'),
               (203394, 'O')]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[
            str(np.random.choice(self._options, 1, p=self._weights)[0])
            for _x in x
        ] for x in tokens]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    ###
    def __init__(self, device: Optional[str] = None):

        self.device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device is not None:
            self.device = device

        self.vocab: dataset.Vocabulary = dataset.Vocabulary(
            threshold=2, path='model/vocab.pkl')

        # self.model: lstm.NerModel = lstm.NerModel(
        #     n_classes=13,
        #     embedding_dim=100,
        #     vocab_size=len(self.vocab),
        #     padding_idx=self.vocab.pad,
        #     hidden_size=100,
        #     bidirectional=True,
        # ).to(device)

        self.model: lstm.NerModel = lstm.NerModel(n_classes=13,
                                                  embedding_dim=100,
                                                  vocab_size=400_002,
                                                  padding_idx=0,
                                                  hidden_size=100,
                                                  bidirectional=True,
                                                  pretrained_emb=None,
                                                  freeze_weights=False).to(
                                                      self.device)

        self.model.load_state_dict(
            torch.load('model/6322_pre_bi.pth',
                       map_location=torch.device(self.device)))

    ###

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        labels = []

        self.model.eval()

        for sentence in tokens:
            inputs = torch.tensor([self.vocab[word] for word in sentence
                                   ]).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)  # pylint: disable=all
            predictions = torch.argmax(outputs, dim=-1).flatten()
            str_predictions = [
                self.vocab.get_label(p.item())  # type: ignore
                for p in predictions
            ]
            labels.append(str_predictions)

        return labels
