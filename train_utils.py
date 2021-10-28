import torch
from torch import nn
import numpy as np


def detach(states):
    """
    Detach recurent states from its history to achieve bptt.
    Note, if states is None, None will be returned

    Input:
    states: list of Tensor: all the states

    Output:
    list of Tensor: all the detached states
    """
    if states is None:
        return None
    else:
        return [state.detach() for state in states]


def evaluate_on_dataset(data_reader, model, which_data, batch_size, seq_len, device):
    """
    Evaluate a model on a validation or testing dataset

    Input:
    data_reader: Corpus: the object that stores the text data
    model: torch.nn.Module: the object that stores the LM
    which_data: string: either "valid" for validation data, or "test" for text data
    batch_size: int: the batch size of the evaluation
    seq_len: int: the length of each sequence in each batch
    device: torch.device: specifying where the data will be operated on

    (note that, in priciple, the seq_len and batch_size variables should not affect the evaluation result)

    Output:
    loss: float: the average cross entropy
    """
    # enter evaluation mode
    model.eval()

    # start evaluation
    total_loss = 0
    n_loss = 0
    criterion = nn.CrossEntropyLoss()
    states = None
    for inputs, targets in data_reader.generate_data(
        batch_size, seq_len, which_data, device
    ):
        # inference
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        # accumulate loss
        total_loss += loss.item()
        n_loss += 1
    total_loss /= n_loss

    # reset model to training mode
    model.train()

    return total_loss
