"""Main file for interactive training and testing
"""

import torch
import data
import lstm
import pathlib

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

torch.manual_seed(42)
trainset = data.NerDataset(threshold=2, window_size=100)
vocab: data.Vocabulary = trainset.vocab
devset= data.NerDataset(path='../../data/dev.tsv', vocab=vocab)
trainloader, devloader = data.get_dataloaders(
    trainset,
    devset,
    batch_size_train=128,
    batch_size_dev=1024)

model = lstm.NerModel(
    n_classes=13,
    embedding_dim=100,
    vocab_size=len(vocab),
    padding_idx=(vocab.pad),
    hidden_size=100,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


model.load_state_dict(torch.load('../../model/03Apr-23:37.pth'))
# lstm.train(model,
#           trainloader,
#           devloader,
#           optimizer,
#           vocab=vocab,
#           patience=1,
#           epochs=50,
#           log_steps=None,
#           device=device,
#           early_stop=False,
#           f1_average='macro',
#           save_path=pathlib.Path('../../model/'))
