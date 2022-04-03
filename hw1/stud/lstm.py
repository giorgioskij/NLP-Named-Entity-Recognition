"""
Implements Deep Learning-related stuff to perform Named Entity Classification
"""

import torch
from torch import nn
import time
import pathlib
from datetime import datetime
from seqeval import metrics


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
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False)

        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=n_classes)

        self.dropout = nn.Dropout()

    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = self.dropout(embeddings)
        # print(f'emb: {embeddings.shape}')

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        # print(f'lstm: {lstm_out.shape}')

        clf_out = self.linear(lstm_out)

        # clf_out = clf_out.view(128 * 7, -1)

        return clf_out


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    devloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab,
    patience: int = 1,
    epochs: int = 50,
    log_steps: int = 100,
    verbose: bool = True,
    device: str = 'cpu',
    early_stop: bool = True,
    f1_average: str = 'macro',
    save_path: pathlib.Path = pathlib.Path('../../model/')
    ) -> str:

    model.to(device)
    path = save_path / f"{datetime.now().strftime('%d%b-%H:%M')}.pth"
    current_patience: int = patience
    best_f1_score = 0
    previous_accuracy = 0
    pad_label_id = vocab.get_label_id('PAD')
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=pad_label_id, reduction='sum')

    for epoch in range(epochs):

        # train for an epoch
        model.train()
        total_correct, total_count, total_loss = 0, 0, 0
        epoch_start_time = time.time()
        if verbose and log_steps is not None:
            print(f'Starting epoch {epoch + 1}...')
        for i, data in enumerate(trainloader):

            # forward and backpropagation
            inputs, labels = data[0].to(device), data[1].flatten().to(device)
            loss, outputs = step(model, loss_fn, inputs, labels, optimizer)

            # evaluate predictions
            predictions = torch.argmax(outputs, dim=1)

            # exclude padding from stats
            real_predictions = predictions[labels != pad_label_id]
            real_labels = labels[labels != pad_label_id]
            correct = (real_predictions == real_labels).sum().item()

            # update stats
            total_loss += loss
            total_correct += correct
            total_count += len(real_labels)

            if log_steps is not None and ((i+1) % log_steps) == 0:
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
        accuracy, loss, f1 = test(model,
                                  devloader,
                                  vocab=vocab,
                                  loss_fn=loss_fn,
                                  device=device,
                                  f1_average=f1_average,
                                  verbose=False)

        # save the best model
        if f1 > best_f1_score:
            best_f1_score = f1
            torch.save(model.state_dict(), path)

        # early stopping
        if early_stop:
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
            f'| valid F1-score {f1:.2%}'
        )
        print('-' * 59)

    print(f'Saved best model at {path}')
    return model


def step(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer = None):

    outputs = model(inputs)
    outputs = outputs.view(-1, outputs.shape[-1])
    # labels = labels.view(-1)

    loss = loss_fn(outputs, labels)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), outputs


def test(
    model: nn.Module,
    devloader: torch.utils.data.DataLoader,
    vocab,
    loss_fn: nn.Module = None,
    f1_average: str = 'macro',
    pad_label_id: int = 13,
    device: str = 'cpu',
    verbose: bool = True):

    pad_label_id = vocab.get_label_id(vocab.pad_label)

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_label_id, reduction='sum')

    model.to(device)
    model.eval()

    total_count, total_correct, total_loss = 0, 0, 0

    true_labels = []
    predicted_labels = []

    for data in devloader:
        # forward pass
        inputs, labels = data[0].to(device), data[1].flatten().to(device)
        loss, outputs = step(model, loss_fn, inputs, labels)

        # evaluate predictions
        predictions = torch.argmax(outputs, dim=1)

        # exclude padding from stats
        real_predictions = predictions[labels != pad_label_id]
        real_labels = labels[labels != pad_label_id]
        correct = (real_predictions == real_labels).sum().item()

        # update stats
        total_loss += loss
        total_correct += correct
        total_count += len(real_labels)

        predicted_labels.extend(real_predictions.tolist())
        true_labels.extend(real_labels.tolist())


    accuracy = total_correct / total_count
    loss = total_loss / len(devloader)

    f1 = metrics.f1_score(
        [[vocab.get_label(i) for i in true_labels]],
        [[vocab.get_label(i) for i in predicted_labels]],
        average=f1_average)
    if verbose:
        print(f'Accuracy: {accuracy:.2%} | Loss: {loss:.4f} | F1: {f1:.2%}')
    return accuracy, loss, f1

