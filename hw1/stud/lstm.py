"""
Implements Deep Learning-related stuff to perform Named Entity Classification
"""

from typing import Iterable, List, Optional, Tuple
import time
import pathlib
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional
import torch.backends.cudnn
from seqeval import metrics

import torchcrf
from . import dataset, config


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
                 char_out_channels: int,
                 bidirectional: bool = False,
                 pretrained_emb: Optional[torch.Tensor] = None,
                 freeze_weights: bool = False,
                 use_pos: bool = False,
                 double_linear: bool = False,
                 char_mode: str = 'conv'):
        """
        Initializes a model to perform NER using character embeddings as well
        as word embeddings.

        Args:
            n_classes (int): number of classes
            embedding_dim (int): dimension of the word embeddings
            char_embedding_dim (int): dimension of the character embeddings
            char_vocab (dataset.CharVocabulary): char vocabulary to use
            vocab_size (int): size of the word vocabulary
            padding_idx (int): index of the padding word
            hidden_size (int): size of the hidden layer of the main LSTM
            char_hidden_size (int): hidden size of the character LSTM
            char_out_channels (int): out-channels of the char convolution layer
            bidirectional (bool, optional):
                Whether the main LSTM is bidirectional. Defaults to False.
            pretrained_emb (Optional[torch.Tensor], optional):
                pretrained word embeddings to load. Defaults to None.
            freeze_weights (bool, optional):
                if pretrained embeddings are specified, whether to freeze them.
                Defaults to False.
            use_pos (bool, optional):
                whether to use pos tags. Defaults to False.
            double_linear (bool, optional):
                if true, the classifier is made by two fc-layers.
                Defaults to False.
            char_mode (str, optional):
                what type of model to use to obtain character embeddings,
                between 'lstm' and 'conv'. Defaults to 'conv'.
        """
        super().__init__()

        self.use_pos = use_pos
        self.double_linear = double_linear
        self.char_out_channels = char_out_channels
        self.char_mode: str = char_mode

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
                                           char_embedding_dim,
                                           padding_idx=self.char_vocab.pad)

        # cnn for characters
        self.char_cnn = nn.Conv2d(in_channels=1,
                                  out_channels=self.char_out_channels,
                                  kernel_size=(3, char_embedding_dim),
                                  padding=(2, 0))

        # lstm for characters
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim,
                                 hidden_size=char_hidden_size,
                                 batch_first=True,
                                 bidirectional=True,
                                 num_layers=2,
                                 dropout=0.5)

        # main lstm module
        self.lstm = nn.LSTM(input_size=embedding_dim + self.char_out_channels,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional,
                            num_layers=2,
                            dropout=0.5)

        self.linear = nn.Linear(in_features=hidden_size *
                                2 if bidirectional else hidden_size,
                                out_features=hidden_size * 2)

        self.linear2 = nn.Linear(in_features=hidden_size * 2,
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

        if self.char_mode.lower() == 'lstm':

            # char_lstm_out: [batch * window, n_chars, char_hidden]
            # char_lstm_hidden: [2, batch * window, char_hidden]
            # char_lsmt_cell: [2, batch * window, char_hidden]
            char_lstm_out, (char_lstm_hidden,
                            char_lstm_cell) = self.char_lstm(char_embeddings)

            # # char_lstm_hidden: [batch * window, 2, char_hidden]
            # char_lstm_hidden = char_lstm_hidden.transpose(0, 1)
            # # char_out: [batch, window, 2 * char_hidden]
            # char_out = char_lstm_hidden.reshape(batch_size, window_size,
            #                                     char_lstm_hidden.shape[2] * 2)

            # take only the last timestep: [batch * window, char_hidden]
            char_out = char_lstm_out[:, -1, :].reshape(batch_size, window_size,
                                                       char_lstm_out.shape[2])
        else:
            char_embeddings = char_embeddings.unsqueeze(1)

            # convolution and maxpool
            chars_cnn_out = self.char_cnn(char_embeddings)
            char_out = functional.max_pool2d(
                chars_cnn_out, kernel_size=(chars_cnn_out.shape[2],
                                            1)).view(chars_cnn_out.shape[0],
                                                     self.char_out_channels)

            # reshape to divide batch and word
            char_out = char_out.reshape(batch_size, window_size,
                                        char_out.shape[1])

        # get word embeddings: [batch, window, word_hidden]
        embeddings = self.embedding(x)

        # cat word and char embeddings: [batch, window, word_hidden+char_hidden]
        concatenated = torch.cat((embeddings, char_out), dim=2)

        # main lstm: [batch, window, 2 * (word_hidden + char_hidden)]
        lstm_out, _ = self.lstm(concatenated)

        # add dropout
        lstm_out = self.dropout(lstm_out)

        # classifier: [batch, window, n_classes]
        clf_out = self.linear(lstm_out)
        clf_out = self.dropout(torch.relu(clf_out))
        clf_out = self.linear2(lstm_out)
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
        """Initializes a model to perform NER

        Args:
            n_classes (int): number of classes
            embedding_dim (int): dimension of the word embeddings
            vocab_size (int): size of the word vocabulary
            padding_idx (int): index of the padding word
            hidden_size (int): size of the hidden layer of the main LSTM
            bidirectional (bool, optional):
                Whether the main LSTM is bidirectional. Defaults to False.
            pretrained_emb (Optional[torch.Tensor], optional):
                pretrained word embeddings to load. Defaults to None.
            freeze_weights (bool, optional):
                if pretrained embeddings are specified, whether to freeze them.
                Defaults to False.
            double_linear (bool, optional):
                if true, the classifier is made by two fc-layers.
                Defaults to False.
            use_pos (bool, optional):
                whether to use pos tags. Defaults to False.
        """
        super().__init__()

        self.n_classes: int = n_classes
        self.double_linear: bool = double_linear
        self.use_pos: bool = use_pos
        self.padding_idx: int = padding_idx

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

        self.dropout = nn.Dropout()

    def forward(self, x, pos_tags: Optional[torch.Tensor] = None):

        embeddings = self.embedding(x)
        embeddings = self.dropout(embeddings)

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


def train(
    model: NerModel,
    trainloader: torch.utils.data.DataLoader,
    devloader: torch.utils.data.DataLoader,
    params: TrainParams,
    logic: bool = False,
    use_crf: bool = False
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Trains the model

    Args:
        model (nn.Module): the model to train
        trainloader (nn.utils.data.DataLoader): dataloader for training
        devloader (nn.utils.data.DataLoader): dataloader for evaluation
        params (TrainParams): parameters
        logic (bool, optional): whether to use logic rules. Defaults to False
        use_crf (bool, optional): whether to use CRF. Defaults to False

    Returns:
        List[float]: History of the training loss
        List[float]: History of the training f1-score 
        List[float]: History of the evaluation loss
        List[float]: History of the evaluation f1-score
    """

    torch.manual_seed(config.SEED)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    f1_train_hist = []
    f1_eval_hist = []
    loss_train_hist = []
    loss_eval_hist = []

    if use_crf:
        crf: Optional[nn.Module] = torchcrf.CRF(num_tags=model.n_classes + 1,
                                                batch_first=True).to(
                                                    params.device)
        crf_opt = torch.optim.SGD(crf.parameters(), lr=0.001)
    else:
        crf = None
        crf_opt = None

    model.to(params.device)
    path = params.save_path / f"{datetime.now().strftime('%d%b-%H:%M')}.pth"
    crf_path = params.save_path / f"crf-{datetime.now().strftime('%d%b-%H:%M')}.pth"
    best_score, best_epoch = 0, 0

    try:
        for epoch in range(params.epochs):
            if params.verbose and params.log_steps is not None:
                print(f'Starting epoch {epoch + 1}...')
            epoch_start_time = time.time()

            # train for an epoch
            model.train()
            _, train_loss, train_f1 = run_epoch(model=model,
                                                dataloader=trainloader,
                                                params=params,
                                                crf=crf,
                                                crf_opt=crf_opt)
            f1_train_hist.append(train_f1)
            loss_train_hist.append(train_loss)

            if params.scheduler is not None:
                lr = params.scheduler.get_lr()[0]  # type: ignore
                params.scheduler.step()

            # test the model
            model.eval()
            accuracy, loss, f1 = run_epoch(model=model,
                                           dataloader=devloader,
                                           params=params,
                                           evaluate=True,
                                           logic=logic,
                                           crf=crf,
                                           crf_opt=crf_opt)
            f1_eval_hist.append(f1)
            loss_eval_hist.append(loss)

            # save the best model
            metric = f1
            if metric > best_score:
                best_epoch = epoch + 1
                best_score = metric
                torch.save(model.state_dict(), path)
                if crf is not None:
                    torch.save(crf.state_dict(), crf_path)

            if params.verbose:
                if params.log_steps is not None:
                    print('-' * 59)
                print(f'Epoch {epoch + 1:3d} ',
                      f'| {time.time() - epoch_start_time:5.2f}s ',
                      f'| Eval acc {accuracy:.2%} ',
                      f'| Eval f1 {f1:6.2%} ',
                      f'| Train f1 {train_f1:6.2%}',
                      sep='')
                if params.log_steps is not None:
                    print('-' * 59)

        if params.verbose:
            print(f'Saved model from epoch {best_epoch} '
                  f'with score {best_score:.2%} at {path}')
            if crf is not None:
                print(f'Saved crf at {crf_path}')

        # model.load_state_dict(torch.load(path))
    except KeyboardInterrupt:
        print('Stopping training...')
        print(f'Model from epoch {best_epoch} '
              f'with score {best_score:.2%} is saved at {path}')
        return loss_train_hist, f1_train_hist, loss_eval_hist, f1_eval_hist

    return loss_train_hist, f1_train_hist, loss_eval_hist, f1_eval_hist


def run_epoch(
    model: NerModel,
    dataloader: torch.utils.data.DataLoader,
    params: TrainParams,
    evaluate: bool = False,
    logic: bool = True,
    crf: Optional[torchcrf.CRF] = None,
    crf_opt: Optional[torch.optim.Optimizer] = None
) -> Tuple[float, float, float]:
    """Runs a single epoch on the given dataloader

    Args:
        model (nn.Module): model to use
        dataloader (torch.utils.data.DataLoader): dataloader to use
        params (TrainParams): parameters
        evaluate (bool, optional):
            if it's an evaluation epoch. Defaults to False
        logic (bool, optional):
            whether to use logic rules. Defaults to False
        crf (torchcrf.CRF, optional):
            the CRF model to use. Defaults to None
        crf_opt (torch.optim.Optimizer, optional):
            if crf is specified, the optimizer to use for the CRF model.
            Defaults to None

    Returns:
        float: accuracy
        float: loss
        float: f1-score
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

        batch_size, window_size = inputs.shape[:2]

        postags = data[2].to(params.device) if model.use_pos else None
        chars = (data[2].to(params.device)
                 if isinstance(model, NerModelChar) else None)

        # forward and backward pass
        loss, outputs = step(
            model, params.loss_fn, inputs, labels, postags, chars,
            params.optimizer if not evaluate and crf is None else None)

        if crf is not None:
            outputs = outputs.reshape(batch_size, window_size, -1)
            outputs = torch.cat(
                (outputs, torch.zeros(outputs.shape[0], outputs.shape[1], 1).to(
                    params.device)),
                dim=2)
            labels = labels.reshape(batch_size, window_size)
            mask = (labels != params.vocab.pad_label_id)
            loss = -crf(outputs, labels, mask)

            if not evaluate and crf_opt and params.optimizer:
                loss.backward()
                params.optimizer.step()
                crf_opt.step()
                params.optimizer.zero_grad()
                crf_opt.zero_grad()

            # evaluate predictions
            predictions = crf.decode(outputs, mask)
            real_predictions = torch.LongTensor(
                [p for s in predictions for p in s]).to(params.device)

        else:
            predictions = torch.argmax(outputs, dim=1)
            real_predictions = predictions[labels != params.vocab.pad_label_id]

        if evaluate and logic:
            predictions = apply_logic(torch.LongTensor(predictions))

        # exclude padding from stats
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

    return accuracy, float(loss), f1


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
    new_tags: torch.Tensor = torch.Tensor(tags).clone()
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
         logic: bool = False,
         crf: Optional[torchcrf.CRF] = None):
    acc, loss, f1 = run_epoch(model, dataloader, params, True, logic, crf=crf)

    if params.verbose:
        print(f'Accuracy: {acc:.2%} | Loss: {loss:.4f} | F1: {f1:.2%}')
    return acc, loss, f1


def predict(model: NerModel,
            vocab: dataset.Vocabulary,
            tokens: List[List[str]],
            device: torch.device,
            window_size: int = 100,
            crf: Optional[torchcrf.CRF] = None) -> List[List[str]]:
    """Predict a bunch of human-readable sentences

    Args:
        tokens (List[List[str]]): list of tokenized sentences to predict

    Returns:
        List[List[str]]: tags of the tokens
    """

    labels = []
    model.eval()
    if crf is not None:
        crf.eval()
    for sentence in tokens:
        original_len = len(sentence)
        indexed_sentence = [
            vocab.get_word_id(word.lower()) for word in sentence
        ]
        indexed_sentence += [vocab.pad] * (window_size - len(indexed_sentence))
        inputs = torch.tensor(indexed_sentence).unsqueeze(0).to(device)

        # inputs = torch.tensor([vocab[word.lower()] for word in sentence
        #   ]).unsqueeze(0).to(device)

        outputs = model(inputs)  # pylint: disable=all
        if crf is not None:
            outputs = outputs.reshape(inputs.shape[0], inputs.shape[1], -1)
            outputs = torch.cat(
                (outputs, torch.zeros(outputs.shape[0], outputs.shape[1],
                                      1).to(device)),
                dim=2)
            predictions = crf.decode(outputs)[0]
        else:
            predictions = torch.argmax(outputs, dim=-1).flatten()

        predictions = predictions[:original_len]
        str_predictions = [
            vocab.get_label(int(p))  # type: ignore
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
