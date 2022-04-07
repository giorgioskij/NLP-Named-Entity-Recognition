from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
import torch.utils.data

from . import dataset, lstm


class NerdTagger:
    
    def __init__(self, style: str = 'original', device: Optional[str] = None):
        self.model: lstm.NerModel
        self.lower: bool
        self.vocab: dataset.Vocabulary
        
        self.device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device is not None:
            self.device = device
        
        if style == 'original':
            print('Generating good old fashioned Bi-LSTM')
            self.lower = True
            self.vocab = dataset.Vocabulary(threshold=2,
                                            path=Path('model/vocab.pkl'))
            self.model = lstm.NerModel(n_classes=13,
                                       embedding_dim=100,
                                       vocab_size=len(self.vocab),
                                       padding_idx=self.vocab.pad,
                                       hidden_size=100,
                                       bidirectional=True).to(self.device)
            self.model.load_state_dict(
                torch.load('model/6033bi.pth',
                           map_location=torch.device(self.device)))
        
        elif style == 'glove':
            print('Generating Bi-LSTM with glove embeddings')
            self.lower = True
            self.model = lstm.NerModel(n_classes=13,
                                       embedding_dim=100,
                                       vocab_size=400_002,
                                       padding_idx=0,
                                       hidden_size=100,
                                       bidirectional=True,
                                       pretrained_emb=None,
                                       freeze_weights=False).to(self.device)
            self.model.load_state_dict(torch.load(
                'model/6965-bi-nofreeze-glove.pth',
                map_location=torch.device(self.device)))
            self.vocab = dataset.Vocabulary(path=Path('model/glove_vocab.pkl'))
        
        else:
            raise ValueError('I don\'t know this style of model')
        
        self.model.eval()
    
    def train_new_model(self):
        trainset = dataset.NerDataset(threshold=2, window_size=100)
        self.vocab: dataset.Vocabulary = trainset.vocab
        devset = dataset.NerDataset(path=Path('data/dev.tsv'),
                                    vocab=self.vocab)
        
        trainloader, devloader = dataset.get_dataloaders(trainset,
                                                         devset,
                                                         batch_size_train=128,
                                                         batch_size_dev=1024)
        
        self.model = lstm.NerModel(n_classes=13,
                                   embedding_dim=100,
                                   vocab_size=len(self.vocab),
                                   padding_idx=self.vocab.pad,
                                   hidden_size=100,
                                   bidirectional=True).to(self.device)
        
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=1,
                                    weight_decay=0.00001,
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
                                  save_path=Path('model/'))
        
        lstm.train(self.model, trainloader, devloader, params)
    
    def test(self, path: Path = Path('data/dev.tsv')):
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
                                  save_path=Path('model/'))
        acc, loss, f1 = lstm.test(self.model, devloader, params=params)
        return acc, loss, f1
    
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        labels = []
        self.model.eval()
        for sentence in tokens:
            inputs = torch.tensor(
                [self.vocab[word.lower() if self.lower else word]
                 for word in sentence]).unsqueeze(0).to(self.device)
            outputs = self.model(inputs)  # pylint: disable=all
            predictions = torch.argmax(outputs, dim=-1).flatten()
            str_predictions = [
                self.vocab.get_label(p.item())  # type: ignore
                for p in predictions
            ]
            labels.append(str_predictions)
        return labels
