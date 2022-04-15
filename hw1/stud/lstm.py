"""
Implements Deep Learning-related stuff to perform Named Entity Classification
"""

from typing import List, Optional, Tuple
import time
import pathlib
from datetime import datetime
from dataclasses import dataclass
import os

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional
import torch.backends.cudnn
from seqeval import metrics

from . import dataset, config

# TODO: add fixed seed


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


class NerModelChar(nn.Module):
    """
        An LSTM model to perform Named Entity Classification which uses
        character embedding as well as word embeddings
    """

    def __init__(self,
                 n_classes: int,
                 embedding_dim: int,
                 char_embedding_dim: int,
                 char_vocab: dataset.CharVocabulary,
                 vocab_size: int,
                 padding_idx: int,
                 hidden_size: int,
                 char_hidden_size: int,
                 bidirectional: bool = False,
                 pretrained_emb: Optional[torch.Tensor] = None,
                 freeze_weights: bool = False,
                 use_pos: bool = False,
                 double_linear: bool = False):
        super().__init__()

        self.use_pos = use_pos
        self.double_linear = double_linear

        # word embedding
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

        # character embedding
        self.char_vocab: dataset.CharVocabulary = char_vocab
        self.char_embedding = nn.Embedding(len(self.char_vocab),
                                           char_embedding_dim)

        #
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim,
                                 hidden_size=char_hidden_size,
                                 batch_first=True,
                                 bidirectional=False,
                                 num_layers=2,
                                 dropout=0.5)

        # main lstm module
        self.lstm = nn.LSTM(input_size=embedding_dim + char_hidden_size * 2,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional,
                            num_layers=2,
                            dropout=0.5)

        self.linear = nn.Linear(in_features=hidden_size *
                                2 if bidirectional else hidden_size,
                                out_features=n_classes)

        self.dropout = nn.Dropout()

    def forward(self, x, chars):
        # x shape: [batch, window]
        # chars shape: [batch, window, n_chars]
        batch_size = x.shape[0]
        window_size = x.shape[1]

        # get char embeddings: [batch, window, n_chars, char_emb]
        char_embeddings = self.char_embedding(chars)
        # flatten: [batch * window, chars, char_emb]
        char_embeddings = char_embeddings.flatten(start_dim=0, end_dim=1)

        # char_lstm_out: [batch * window, n_chars, char_hidden]
        # char_lstm_hidden: [2, batch * window, char_hidden]
        # char_lsmt_cell: [2, batch * window, char_hidden]
        char_lstm_out, (char_lstm_hidden,
                        char_lstm_cell) = self.char_lstm(char_embeddings)

        # char_lstm_hidden: [batch * window, 2, char_hidden]
        char_lstm_hidden = char_lstm_hidden.transpose(0, 1)

        # char_out: [batch, window, 2 * char_hidden]
        char_out = char_lstm_hidden.reshape(batch_size, window_size,
                                            char_lstm_hidden.shape[2] * 2)

        # get word embeddings: [batch, window, word_hidden]
        embeddings = self.embedding(x)

        # cat word and char embeddings: [batch, window, word_hidden+char_hidden]
        concatenated = torch.cat((embeddings, char_out), dim=2)

        # main lstm: [batch, window, 2 * (word_hidden + char_hidden)]
        lstm_out, _ = self.lstm(concatenated)

        # classifier: [batch, window, n_classes]
        clf_out = self.linear(lstm_out)

        return clf_out


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
            bidirectional=bidirectional,
            num_layers=2,
            dropout=0.5)

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

        # extra stuff
        # self.lstm2 = nn.LSTM(input_size=n_classes,
        #                      hidden_size=50,
        #                      batch_first=True,
        #                      bidirectional=bidirectional,
        #                      dropout=0.5)

        # self.clf = nn.Linear(in_features=100, out_features=n_classes)

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

        # extra stuff
        # lstm2_out, _ = self.lstm2(torch.relu(clf_out))
        # clf_out = self.clf(lstm2_out)

        return clf_out


# class NerModelPos(NerModel):
#     """
#     An LSTM model that takes as input both word indices to compute embeddings
#     and a pos tag for each wordd
#     """

#     def __init__(
#         self,
#         n_classes: int,
#         embedding_dim: int,
#         vocab_size: int,
#         padding_idx: int,
#         hidden_size: int,
#         bidirectional: bool = False,
#         pretrained_emb: Optional[torch.Tensor] = None,
#         freeze_weights: bool = False,
#         double_linear: bool = False,
#     ):
#         super().__init__(n_classes, embedding_dim + 1, vocab_size, padding_idx,
#                          hidden_size, bidirectional, pretrained_emb,
#                          freeze_weights, double_linear)


def train(model: NerModel,
          trainloader: torch.utils.data.DataLoader,
          devloader: torch.utils.data.DataLoader,
          params: TrainParams,
          logic: bool = False) -> None:
    """Trains the model

    Args:
        model (nn.Module): the model to train
        trainloader (nn.utils.data.DataLoader): dataloader for training
        devloader (nn.utils.data.DataLoader): dataloader for evaluation
        params (TrainParams): parameters
    """

    torch.manual_seed(config.SEED)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                                        evaluate=True,
                                        logic=logic)

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
              evaluate: bool = False,
              logic: bool = True) -> Tuple[float, float, float]:
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
        chars = (data[2].to(params.device)
                 if isinstance(model, NerModelChar) else None)

        # forward and backward pass
        loss, outputs = step(model, params.loss_fn, inputs, labels, postags,
                             chars, params.optimizer if not evaluate else None)

        # evaluate predictions
        predictions = torch.argmax(outputs, dim=1)

        if evaluate and logic:
            predictions = apply_logic(predictions)

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


# def apply_logic(tags: torch.Tensor) -> torch.Tensor:

#     new_predictions: torch.Tensor = torch.zeros_like(tags).long()
#     for i, sentence in enumerate(tags):
#         for j, tag in enumerate(sentence):
#             if not j:
#                 if (6 <= tag <= 11):
#                     new_predictions[i][j] = tag - 5
#                 else:
#                     new_predictions[i][j] = tag
#                 new_predictions[i][j] = tag
#             elif (6 <= tag <= 11) and (sentence[j - i] != (tag - 5)):
#                 new_predictions[i][j] = tag - 5
#             else:
#                 new_predictions[i][i] = tag

#     return new_predictions


def apply_logic(tags: torch.LongTensor) -> torch.Tensor:
    """
    Applies logic rules to verify a tag sequence and correct mistakes.
    It uses the assumption that an I-TAG which follows anything but an I-TAG or 
    a B-TAG of the same class is surely wrong, and tries to infer a new one 
    based on a policy.

    Args:
        tags (torch.LongTensor): the tag sequence

    Returns:
        torch.Tensor: the correct tag sequence
    """
    new_tags: torch.Tensor = tags.clone()
    for i in range(1, len(new_tags)):
        tag = int(new_tags[i])
        prev_tag = int(tags[i - 1])
        # if tag is an I-TAG and the previous has not the same category
        if 6 <= tag <= 11 and prev_tag != (tag - 6) and prev_tag != tag:
            next_tag = int(tags[i + 1]) if i < len(tags) - 1 else 12
            new_tags[i] = infer_tag(prev_tag, tag, next_tag)

    return new_tags


# i-th tag is an I-TAG that is not following its corresponding B-TAG
def infer_tag(prev_tag: int, current_tag: int, next_tag: int) -> int:
    """
    A minimal policy to infer a new tag based on the previous and the next one.
    It IS possible to design a policy that never lowers the seqeval-F1 metric.
    This one, however, accepts the possibility to introduce mistakes in very
    rare cases, to favour more reasonable scenarios.

    Args:
        prev_tag (int): the previous tag
        current_tag (int): the tag to replace
        next_tag (int): the next tag

    Returns:
        int: the new inferred tag

    --- Follows an exhaustive table of the policy ---
    O  - I1 - I2  ->    B2
    O  - I1 - O/B ->    B1

    B0 - I1 - O/B ->    B1 (I0 could be ok but it can be worse)
    B0 - I1 - I0  ->    I0 (never worse would be B0, but I0 is more reasonable)
    B0 - I1 - I2  ->    B2

    I0 - I1 - O/B ->    B1 (I0 could be ok but it can be worse)
    I0 - I1 - I2  ->    B2
    I0 - I1 - I0  ->    I0 (never worse would be B0, but I0 is more reasonable)

     B1 if:
        O  - I1 - O/B
        B0 - I1 - O/B
        I0 - I1 - O/B
        whenever it's followed by O/B

    B2 if:
        O  - I1 - I2
        B0 - I1 - I2
        I0 - I1 - I2
        whenever it's followed by I2 (I2 has different tag from previous)

    I0 if:
        B0 - I1 - I0
        I0 - I1 - I0
        whenever it's followed by I0 (I0 has same tag as previous)
    
    If we want to avoid any case in which the policy decreases the F1 score,
    we use B0 instead of I0
    """

    # if it's followed by an O-TAG or a B-TAG, return the B version of itself
    if next_tag < 6 or next_tag > 11:
        return current_tag - 6
    # if next is an I-TAG of a different category from the previous
    # return B-TAG with the same category as the following I-TAG
    if next_tag != prev_tag and next_tag != prev_tag + 6:
        return next_tag - 6
    # otherwise
    # return next_tag # more reasonable option
    return current_tag - 6  # option that never worsens F1


def step(model: NerModel,
         loss_fn: nn.Module,
         inputs: torch.Tensor,
         labels: torch.Tensor,
         postags: Optional[torch.Tensor],
         chars: Optional[torch.Tensor],
         optimizer: Optional[torch.optim.Optimizer] = None):

    if postags is not None:
        outputs = model(inputs, postags)
    elif chars is not None:
        outputs = model(inputs, chars)
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


def test(model: NerModel,
         dataloader: torch.utils.data.DataLoader,
         params: TrainParams,
         logic: bool = False):
    acc, loss, f1 = run_epoch(model, dataloader, params, True, logic)

    if params.verbose:
        print(f'Accuracy: {acc:.2%} | Loss: {loss:.4f} | F1: {f1:.2%}')
    return acc, loss, f1


def predict(model: nn.Module, vocab: dataset.Vocabulary,
            tokens: List[List[str]], device: torch.device) -> List[List[str]]:
    """Predict a bunch of human-readable sentences

    Args:
        tokens (List[List[str]]): list of tokenized sentences to predict

    Returns:
        List[List[str]]: tags of the tokens
    """

    labels = []
    model.eval()
    for sentence in tokens:
        inputs = torch.tensor([vocab[word.lower()] for word in sentence
                              ]).unsqueeze(0).to(device)
        outputs = model(inputs)  # pylint: disable=all
        predictions = torch.argmax(outputs, dim=-1).flatten()
        str_predictions = [
            vocab.get_label(p.item())  # type: ignore
            for p in predictions
        ]
        labels.append(str_predictions)
    return labels


def predict_char(model: nn.Module, vocab: dataset.Vocabulary,
                 char_vocab: dataset.CharVocabulary, tokens: List[List[str]],
                 device: torch.device) -> List[List[str]]:
    """Predict a bunch of human-readable sentences

    Args:
        tokens (List[List[str]]): list of tokenized sentences to predict

    Returns:
        List[List[str]]: tags of the tokens
    """

    labels = []
    model.eval()
    for sentence in tokens:
        inputs = torch.tensor([vocab[word.lower()] for word in sentence
                              ]).unsqueeze(0).to(device)
        chars: List[List[int]] = [
            [char_vocab.get_char_id(c) for c in word] for word in sentence
        ]
        maxlen = max(len(w) for w in chars)
        padded_chars: List[List[int]] = []
        for word in chars:
            padded_chars.append(word + [char_vocab.pad] * (maxlen - len(word)))
        padded_chars = torch.tensor(padded_chars).unsqueeze(0).to(
            device)  # type: ignore
        outputs = model(inputs, padded_chars)  # pylint: disable=all
        predictions = torch.argmax(outputs, dim=-1).flatten()
        str_predictions = [
            vocab.get_label(p.item())  # type: ignore
            for p in predictions
        ]
        labels.append(str_predictions)
    return labels


# from evaluate.py
def read_dataset(path: str) -> Tuple[List[List[str]], List[List[str]]]:

    tokens_s = []
    labels_s = []

    tokens = []
    labels = []

    with open(path) as f:

        for line in f:

            line = line.strip()

            if line.startswith("#\t"):
                tokens = []
                labels = []
            elif line == "":
                tokens_s.append(tokens)
                labels_s.append(labels)
            else:
                token, label = line.split("\t")
                tokens.append(token)
                labels.append(label)

    assert len(tokens_s) == len(labels_s)

    return tokens_s, labels_s