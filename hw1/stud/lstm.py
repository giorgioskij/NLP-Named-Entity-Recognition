"""
Implements Deep Learning-related stuff to perform Named Entity Classification
"""

from typing import Optional, Tuple
import time
import pathlib
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional
from seqeval import metrics

from . import dataset


@dataclass
class TrainParams:
    """
        Dataclass to group the needed hyperparameters for training
    """
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    vocab: dataset.Vocabulary
    loss_fn: nn.Module
    epochs: int
    log_steps: Optional[int]
    verbose: bool
    device: torch.device
    f1_average: str
    save_path: pathlib.Path


class NerModel(nn.Module):
    """An LSTM model to perform NEC
    """

    def __init__(self,
                 n_classes: int,
                 embedding_dim: int,
                 vocab_size: int,
                 padding_idx: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 pretrained_emb: Optional[torch.Tensor] = None,
                 freeze_weights: bool = False,
                 double_linear: bool = False,
                 use_pos: bool = False):
        super().__init__()

        self.double_linear: bool = double_linear
        self.use_pos: bool = use_pos

        if pretrained_emb is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_emb,
                freeze=freeze_weights,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
            if freeze_weights:
                self.embedding.requires_grad_(False)

        self.lstm = nn.LSTM(
            input_size=embedding_dim if not self.use_pos else embedding_dim +
            17,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional)

        if self.double_linear:
            self.linear1 = nn.Linear(in_features=hidden_size *
                                     2 if bidirectional else hidden_size,
                                     out_features=hidden_size)
            self.linear2 = nn.Linear(in_features=hidden_size,
                                     out_features=n_classes)

        else:
            self.linear = nn.Linear(in_features=hidden_size *
                                    2 if bidirectional else hidden_size,
                                    out_features=n_classes)

        self.dropout = nn.Dropout()

    def forward(self, x, pos_tags: Optional[torch.Tensor] = None):

        embeddings = self.embedding(x)
        embeddings = self.dropout(embeddings)
        # print(f'emb: {embeddings.shape}')

        if self.use_pos and pos_tags is not None:
            oh: torch.Tensor = functional.one_hot(pos_tags, num_classes=17)
            embeddings = torch.cat((embeddings, oh), dim=-1)
            # embeddings = torch.cat((embeddings, pos_tags.unsqueeze(-1)), dim=2)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        if self.double_linear:
            clf_out = self.linear1(lstm_out)
            clf_out = self.dropout(torch.relu(clf_out))
            clf_out = self.linear2(clf_out)
        else:
            clf_out = self.linear(lstm_out)

        return clf_out


class NerModelPos(NerModel):
    """
    An LSTM model that takes as input both word indices to compute embeddings
    and a pos tag for each wordd
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int,
        vocab_size: int,
        padding_idx: int,
        hidden_size: int,
        bidirectional: bool = False,
        pretrained_emb: Optional[torch.Tensor] = None,
        freeze_weights: bool = False,
        double_linear: bool = False,
    ):
        super().__init__(n_classes, embedding_dim + 1, vocab_size, padding_idx,
                         hidden_size, bidirectional, pretrained_emb,
                         freeze_weights, double_linear)


def train(model: NerModel, trainloader: torch.utils.data.DataLoader,
          devloader: torch.utils.data.DataLoader, params: TrainParams) -> None:
    """Trains the model

    Args:
        model (nn.Module): the model to train
        trainloader (nn.utils.data.DataLoader): dataloader for training
        devloader (nn.utils.data.DataLoader): dataloader for evaluation
        params (TrainParams): parameters
    """

    model.to(params.device)
    path = params.save_path / f"{datetime.now().strftime('%d%b-%H:%M')}.pth"
    best_score, best_epoch = 0, 0

    try:
        for epoch in range(params.epochs):
            if params.verbose and params.log_steps is not None:
                print(f'Starting epoch {epoch + 1}...')
            epoch_start_time = time.time()

            # train for an epoch
            model.train()
            _, _, train_f1 = run_epoch(model=model,
                                       dataloader=trainloader,
                                       params=params)

            if params.scheduler is not None:
                lr = params.scheduler.get_lr()[0]  # type: ignore
                params.scheduler.step()

            # test the model
            model.eval()
            accuracy, _, f1 = run_epoch(model=model,
                                        dataloader=devloader,
                                        params=params,
                                        evaluate=True)

            # save the best model
            metric = f1
            if metric > best_score:
                best_epoch = epoch + 1
                best_score = metric
                torch.save(model.state_dict(), path)

            if params.verbose:
                if params.log_steps is not None:
                    print('-' * 59)
                print(
                    f'Epoch {epoch + 1:3d} ',
                    f'| {time.time() - epoch_start_time:5.2f}s ',
                    f'| lr {lr:6.5f} ' if params.scheduler is not None else '',
                    f'| Eval acc {accuracy:.2%} ',
                    f'| Eval f1 {f1:6.2%} ',
                    f'| Train f1 {train_f1:6.2%}',
                    sep='')
                if params.log_steps is not None:
                    print('-' * 59)

        if params.verbose:
            print(f'Saved model from epoch {best_epoch} '
                  f'with score {best_score:.2%} at {path}')

        # model.load_state_dict(torch.load(path))
    except KeyboardInterrupt:
        print('Stopping training...')
        print(f'Model from epoch {best_epoch} '
              f'with score {best_score:.2%} is saved at {path}')
    return None


def run_epoch(model: NerModel,
              dataloader: torch.utils.data.DataLoader,
              params: TrainParams,
              evaluate: bool = False) -> Tuple[float, float, float]:
    """Runs a single epoch on the given dataloader

    Args:
        index (int): Index of the current epoch to display
        model (nn.Module): model to use
        dataloader (torch.utils.data.DataLoader): dataloader to use
        params (TrainParams): parameters
        test (bool): if it's an evaluation epoch

    Returns:
        Tuple[float, float, float]: accuracy, loss, f1 score
    """

    model.to(params.device)
    if evaluate:
        model.eval()

    total_correct, total_count, total_loss = 0, 0, 0
    true_labels = []
    predicted_labels = []

    for i, data in enumerate(dataloader):

        # move data to gpu
        inputs, labels = (data[0].to(params.device),
                          data[1].to(params.device).flatten())

        postags = data[2].to(params.device) if model.use_pos else None

        # forward and backward pass
        loss, outputs = step(model, params.loss_fn, inputs, labels, postags,
                             params.optimizer if not evaluate else None)

        # evaluate predictions
        predictions = torch.argmax(outputs, dim=1)

        # exclude padding from stats
        real_predictions = predictions[labels != params.vocab.pad_label_id]
        real_labels = labels[labels != params.vocab.pad_label_id]
        correct = (real_predictions == real_labels).sum().item()

        # update stats
        total_loss += loss
        total_correct += correct
        total_count += len(real_labels)
        predicted_labels.extend(real_predictions.tolist())
        true_labels.extend(real_labels.tolist())

        if (params.verbose and params.log_steps is not None and
            ((i + 1) % params.log_steps) == 0):
            print(f'| {i + 1:3d}/{len(dataloader):3d} batches '
                  f'| accuracy {total_correct / total_count:.2%}'
                  f'| loss {loss / i:.4f}')
            total_correct, total_count, total_loss = 0, 0, 0

    accuracy: float = total_correct / total_count
    loss = total_loss / len(dataloader)

    f1: float = metrics.f1_score(
        [[params.vocab.get_label(i) for i in true_labels]],
        [[params.vocab.get_label(i) for i in predicted_labels]],
        average=params.f1_average)  # type: ignore

    return accuracy, loss, f1


def step(model: NerModel,
         loss_fn: nn.Module,
         inputs: torch.Tensor,
         labels: torch.Tensor,
         postags: Optional[torch.Tensor],
         optimizer: Optional[torch.optim.Optimizer] = None):

    if postags is not None:
        outputs = model(inputs, postags)
    else:
        outputs = model(inputs)
    outputs = outputs.view(-1, outputs.shape[-1])
    # labels = labels.view(-1)

    loss = loss_fn(outputs, labels)

    if optimizer is not None:
        loss.backward()

        # CAREFUL - MANUAL WEIGHT UPDATE
        # with torch.no_grad():
        #     lr = 0.01
        #     for param in model.parameters():
        #         param -=  param.grad * lr
        #         param.grad.zero_()

        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), outputs


def test(model: NerModel, dataloader: torch.utils.data.DataLoader,
         params: TrainParams):
    acc, loss, f1 = run_epoch(model, dataloader, params, True)

    if params.verbose:
        print(f'Accuracy: {acc:.2%} | Loss: {loss:.4f} | F1: {f1:.2%}')
    return acc, loss, f1
