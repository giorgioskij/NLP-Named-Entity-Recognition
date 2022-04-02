"""Main file for interactive training and testing
"""

import torch
import data
import lstm
import pathlib

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.manual_seed(42)
trainset = data.NerDataset(threshold=2)
vocab: data.Vocabulary = trainset.vocab
devset= data.NerDataset(path='../../data/dev.tsv', vocab=vocab)
trainloader, devloader = data.get_dataloaders(trainset, devset, batch_size=128)

model = lstm.NerModel(
    n_classes=13,
    embedding_dim=100,
    vocab_size=len(vocab),
    padding_idx=(vocab.pad),
    hidden_size=100,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

lstm.train(model,
          trainloader,
          devloader,
          optimizer,
          patience=1,
          epochs=50,
          log_steps=100,
          device = device,
          save_path=pathlib.Path('../../model/'))

