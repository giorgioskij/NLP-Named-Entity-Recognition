from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.utils.data
import stanza
from seqeval import metrics

from . import dataset, lstm, config


class NerdTagger:
    """
    This class encapsulates and abstracts a Named Entity Tagger.
    It is useful to keep the main files clean and empty and keep track of the 
    different implmementations.
    """

    def __init__(self,
                 style: str = 'original',
                 retrain: bool = False,
                 device: Optional[torch.device] = None):
        """Initialize a NerdTagger of a specific type

        Args:
            style (str, optional):
                the type of tagger to use ['original', 'glove', 'stanza'].
                Defaults to 'original'.
            retrain (bool, optional):
                whether to retrain the model. Defaults to False.
            device (Optional[torch.device], optional):
                'cpu' or 'cuda'. Defaults to None.

        Raises:
            ValueError
        """
        self.model: lstm.NerModel
        self.lower: bool
        self.vocab: dataset.Vocabulary

        self.style: str = style
        self.device: torch.device = config.DEVICE
        if device is not None:
            self.device = device

        if style == 'classic':
            print('Generating good old fashioned Bi-LSTM')
            self.lower = True
            self.vocab = dataset.Vocabulary(threshold=2,
                                            path=config.MODEL / 'vocab.pkl')
            self.model = lstm.NerModel(n_classes=13,
                                       embedding_dim=100,
                                       vocab_size=len(self.vocab),
                                       padding_idx=self.vocab.pad,
                                       hidden_size=100,
                                       bidirectional=True,
                                       use_pos=False).to(self.device)
            if not retrain:
                self.model.load_state_dict(
                    torch.load(config.MODEL / '6033bi.pth',
                               map_location=self.device))

        elif 'pos' in self.style.lower():
            print('Generating Bi-LSTM with glove embeddings '
                  'and stanza\'s pos extractor, hidden size 200 and double'
                  'linear layer')
            self.lower = True
            self.vocab = dataset.Vocabulary(path=config.MODEL /
                                            'glove-vocab.pkl')
            self.model = lstm.NerModel(n_classes=13,
                                       embedding_dim=100,
                                       vocab_size=len(self.vocab),
                                       padding_idx=self.vocab.pad,
                                       hidden_size=100,
                                       bidirectional=True,
                                       pretrained_emb=None,
                                       freeze_weights=False,
                                       double_linear=False,
                                       use_pos=True).to(self.device)
            self.model.load_state_dict(
                torch.load(config.MODEL / 'emb-100-pos.pth'))

        elif self.style == 'glove':
            print('Generating Bi-LSTM with glove embeddings')
            self.lower = True
            self.vocab = dataset.Vocabulary(path=config.MODEL /
                                            'glove-vocab.pkl')
            self.model = lstm.NerModel(n_classes=13,
                                       embedding_dim=100,
                                       vocab_size=len(self.vocab),
                                       padding_idx=self.vocab.pad,
                                       hidden_size=100,
                                       bidirectional=True,
                                       pretrained_emb=None,
                                       freeze_weights=False,
                                       double_linear=False,
                                       use_pos=False).to(self.device)
            if retrain:
                self.model.load_state_dict(
                    torch.load(config.MODEL / 'emb-100.pth',
                               map_location=self.device))
            else:
                self.model.load_state_dict(
                    torch.load(config.MODEL / '6965-bi-nofreeze-glove.pth',
                               map_location=self.device))

        elif self.style == 'stanza':
            # compare with performance of stanza
            import stanza
            stanza.download('en', logging_level='FATAL')
            self.stanza = stanza.Pipeline(lang='en',
                                          processors='tokenize,ner',
                                          tokenize_pretokenized=True,
                                          logging_level='FATAL')
            names = [('PERSON', 'PER'), ('LOC', 'LOC'), ('GPE', 'LOC'),
                     ('NORP', 'GRP'), ('ORG', 'CORP'), ('PRODUCT', 'PROD'),
                     ('WORK_OF_ART', 'CW')]
            self.conversion: Dict[str, str] = {}
            for theirs, ours in names:
                self.conversion['B-' + theirs] = 'B-' + ours
                self.conversion['I-' + theirs] = 'I-' + ours
                self.conversion['E-' + theirs] = 'I-' + ours
                self.conversion['S-' + theirs] = 'B-' + ours

        else:
            raise ValueError('I don\'t know this style of model')

        if style != 'stanza':
            self.model.eval()

        if retrain:
            self.train_model()

    def train_model(self):
        if 'pos' in self.style.lower():
            stanza.download('en', logging_level='FATAL')
            pos_tagger: stanza.Pipeline = stanza.Pipeline(
                'en',
                processors='tokenize, pos',
                tokenize_pretokenized=True,
                logging_level='FATAL')

            trainset = dataset.NerDatasetPos(path=config.TRAIN,
                                             pos_tagger=pos_tagger,
                                             vocab=self.vocab)
            # self.vocab: dataset.Vocabulary = trainset.vocab
            devset = dataset.NerDatasetPos(path=config.DEV,
                                           vocab=self.vocab,
                                           pos_tagger=pos_tagger)

        else:
            trainset = dataset.NerDataset(path=config.TRAIN, vocab=self.vocab)
            # self.vocab: dataset.Vocabulary = trainset.vocab
            devset = dataset.NerDataset(path=config.DEV, vocab=self.vocab)

        trainloader, devloader = dataset.get_dataloaders(trainset,
                                                         devset,
                                                         batch_size_train=128,
                                                         batch_size_dev=1024)

        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=0.001,
                                    weight_decay=0,
                                    momentum=0.9)

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_label_id,
                                      reduction='sum')

        params = lstm.TrainParams(optimizer=optimizer,
                                  scheduler=None,
                                  vocab=self.vocab,
                                  loss_fn=loss_fn,
                                  epochs=400,
                                  log_steps=None,
                                  verbose=True,
                                  device=self.device,
                                  f1_average='macro',
                                  save_path=config.MODEL)

        print(f'Starting to train the model: \n{self.model}')
        print(
            f'Style: {self.style} | Vocab size: {len(self.vocab)} | Device: {self.device}'
        )
        lstm.train(self.model, trainloader, devloader, params)

    def test(self, path: Path = config.DEV) -> Tuple[float, float, float]:
        """Test the tagger against a dataset

        Args:
            path (Path, optional): dataset to use for testing. 
            Defaults to Path('data/dev.tsv').

        Returns:
            Tuple[float, float, float]: accuracy, loss, f1
        """

        if self.style == 'stanza':
            return self.test_stanza(path=path)

        devset = dataset.NerDataset(path=path, vocab=self.vocab)
        devloader = torch.utils.data.DataLoader(devset, batch_size=1024)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_label_id,
                                      reduction='sum')
        params = lstm.TrainParams(optimizer=None,
                                  scheduler=None,
                                  vocab=self.vocab,
                                  loss_fn=loss_fn,
                                  epochs=400,
                                  log_steps=None,
                                  verbose=True,
                                  device=self.device,
                                  f1_average='macro',
                                  save_path=config.MODEL)
        acc, loss, f1 = lstm.test(self.model, devloader, params=params)
        return acc, loss, f1

    def test_stanza(self,
                    path: Path = config.DEV) -> Tuple[float, float, float]:

        devset = dataset.NerDataset(path=path,
                                    vocab=dataset.Vocabulary(path=config.MODEL /
                                                             'vocab.pkl'))

        true_labels = list(map(lambda x: x[1], devset.sentences))
        tokens = list(map(lambda x: x[0], devset.sentences))
        pred_labels: List[List[str]] = self.predict_stanza(tokens)

        f1: float = metrics.f1_score(true_labels, pred_labels,
                                     average='macro')  # type: ignore

        return 0., 0., f1

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """Predict a bunch of human-readable sentences

        Args:
            tokens (List[List[str]]): list of tokenized sentences to predict

        Returns:
            List[List[str]]: tags of the tokens
        """

        if self.style == 'stanza':
            return self.predict_stanza(tokens)

        labels = []
        self.model.eval()
        for sentence in tokens:
            inputs = torch.tensor([
                self.vocab[word.lower() if self.lower else word]
                for word in sentence
            ]).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)  # pylint: disable=all
            predictions = torch.argmax(outputs, dim=-1).flatten()
            str_predictions = [
                self.vocab.get_label(p.item())  # type: ignore
                for p in predictions
            ]
            labels.append(str_predictions)
        return labels

    def predict_stanza(self, tokens: List[List[str]]) -> List[List[str]]:
        # compare with stanza just to get an idea: spoiler - it's bad
        stanza_out = self.stanza(tokens).get(  # type: ignore
            'ner', as_sentences=True, from_token=True)

        out: List[List[str]] = [[
            self.conversion[tag] if tag in self.conversion else 'O'
            for tag in sentence
        ]
                                for sentence in stanza_out]
        return out
