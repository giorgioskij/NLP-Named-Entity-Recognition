import numpy as np
from typing import List, Tuple

from model import Model

###
from . import lstm
from . import data
import torch
import pickle
###


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()

    print('building model')

    return StudentModel(device=device)


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary


    ###
    def __init__(self, device: str = None):

        self.device = device
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.vocab = data.Vocabulary(threshold=2, path='model/vocab.pkl')

        self.model: lstm.NerModel = lstm.NerModel(
            n_classes=13,
            embedding_dim=100,
            vocab_size=len(self.vocab),
            padding_idx=self.vocab.pad,
            hidden_size=100
        ).to(device)

        self.model.load_state_dict(torch.load(
            'model/03Apr-23:37.pth',
            map_location=torch.device(device)
        ))
    ###


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        labels = []

        self.model.eval()

        for sentence in tokens:
            inputs = torch.tensor(
                [self.vocab[word] for word in sentence]
            ).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, dim = -1).flatten()
            str_predictions = [self.vocab.get_label(p) for p in predictions]
            labels.append(str_predictions)

        return labels
