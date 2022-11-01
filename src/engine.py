from typing import Iterable, Tuple
import torch
import torch.nn as nn


def train_one_epoch(model: nn.Module, iterator: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.DeviceObjType) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Function to train a model given a data iter, criterion, optimizer, and device
    Returns a tuple containing the epoch loss and epoch accuracy
    '''
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for image, label in iterator:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        y_pred = model(image)

        loss = criterion(y_pred, label)

        # TODO calculate accuracy

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # epoch_acc += acc

    return epoch_loss / len(iterator)#, epoch_acc / len(iterator)


def evaluate(model: nn.Module, iterator: Iterable, criterion: nn.Module,
    device: torch.DeviceObjType) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Evaluates the model on the supplied data iterator with the supplied criterion
    '''
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for image, label in iterator:
            image = image.to(device)
            label = label.to(device)

            y_pred = model(image)

            loss = criterion(y_pred, label)

            # TODO calculate accuracy

            epoch_loss += loss.item()
            # epoch_acc += acc.item()
    return epoch_loss / len(iterator)#, epoch_acc / len(iterator)