"""
Implements Deep Learning-related stuff to perform Named Entity Classification
"""

import torch
from torch import nn
from torch.nn import functional as F
import time
import pathlib


class NerModel(nn.Module):
    """An LSTM model to perform NEC
    """

    def __init__(self,
                 n_classes: int,
                 embedding_dim: int,
                 vocab_size: int,
                 padding_idx: int,
                 hidden_size: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size)

        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=n_classes)

    def forward(self, x):
        x = F.dropout(self.embedding(x))
        x, (_, _) = self.lstm(x)
        x = F.dropout(x)
        x = self.linear(x)

        return x


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    devloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    patience: int = 1,
    epochs: int = 50,
    log_steps: int = 100,
    verbose: bool = True,
    device: str = None,
    save_path: pathlib.Path = pathlib.Path('../../model/')
    ) -> str:

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    current_patience: int = patience
    previous_accuracy = 0
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        # train for an epoch
        model.train()
        total_correct, total_count, total_loss = 0, 0, 0
        epoch_start_time = time.time()
        if verbose: print(f'Starting epoch {epoch + 1}...')
        for i, data in enumerate(trainloader):

            inputs, labels = data[0].to(device), data[1].to(device)
            loss, correct = step(model, loss_fn, inputs, labels, optimizer)

            total_loss += loss
            total_correct += correct
            total_count += len(labels.flatten())
            if ((i+1) % log_steps) == 0:
                print(
                    f'| epoch {epoch+1:3d} '
                    f'| {i+1:3d}/{len(trainloader):3d} batches '
                    f'| accuracy {total_correct / total_count:.2%}'
                    f'| training loss {loss / i:.4f}'
                    f'| elapsed: {time.time() - epoch_start_time:5.2f}'
                )
                total_correct, total_count, total_loss = 0, 0, 0

        # test the model
        model.eval()
        accuracy, loss = test(model, devloader, loss_fn, device, verbose=False)

        # early stopping
        if accuracy < previous_accuracy:
            current_patience -= 1
        else:
            current_patience = patience
        if current_patience < 0: break
        previous_accuracy = accuracy

        print('-' * 59)
        print(
            f'| end of epoch {epoch+1:3d} '
            f'| time: {time.time() - epoch_start_time:5.2f}s '
            f'| valid accuracy {accuracy:.2%} '
        )
        print('-' * 59)


    model_name = f'{accuracy:.4f}'[2:]
    path = save_path / f'{model_name}lstm.pth'
    torch.save(model.state_dict(), path)
    print(f'Saved model at {path}')

    return model


def step(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer = None):

    outputs = model(inputs)
    outputs = outputs.view(-1, outputs.shape[-1])
    labels = labels.view(-1)

    loss = loss_fn(outputs, labels)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()

    return loss.item(), correct


def test(
    model: nn.Module,
    devloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module = None,
    device: str = None,
    verbose: bool = True):

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total_count, total_correct, total_loss = 0, 0, 0
    for data in devloader:
        inputs, labels = data[0].to(device), data[1].to(device)

        loss, correct = step(model, loss_fn, inputs, labels)
        total_loss += loss
        total_correct += correct
        total_count += len(labels.flatten())

    accuracy = total_correct / total_count
    loss = total_loss / len(devloader)

    if verbose:
        print(f'Eval accuracy: {accuracy:.2%}, eval loss: {loss:.2%}')
    return accuracy, loss

