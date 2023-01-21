from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    iterator: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.DeviceObjType,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to train a model given a data iter, criterion, optimizer, and device
    Returns a tuple containing the epoch loss and epoch accuracy
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for image, label in iterator:
        image = image.to(
            device
        )  # [BATCH_SIZE=256, RGB=3, IMAGE_SIZE=32, IMAGE_SIZE=32]
        label = label.to(device)  # [BATCH_SIZE=256]

        optimizer.zero_grad()

        y_pred = model(image)  # [BATCH_SIZE=256, NUM_CLASSES=10]

        loss = criterion(y_pred, label)

        # calculate accuracy (reference: HW4 Q4 def calculate_accuracy(y_pred, y))
        top_y_pred = y_pred.argmax(
            1, keepdim=True
        )  # [BATCH_SIZE=256,1] each an int in [0, NUM_CLASSES-1=9]
        num_correct = top_y_pred.eq(label.view_as(top_y_pred)).sum()
        acc = num_correct.float() / label.shape[0]

        loss.backward()
        optimizer.step()

        if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            lr_scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(
    model: nn.Module,
    iterator: Iterable,
    criterion: nn.Module,
    device: torch.DeviceObjType,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the model on the supplied data iterator with the supplied criterion
    """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for image, label in iterator:
            image = image.to(
                device
            )  # [BATCH_SIZE=256, RGB=3, IMAGE_SIZE=32, IMAGE_SIZE=32]
            label = label.to(device)  # [BATCH_SIZE=256]

            y_pred = model(image)  # [BATCH_SIZE=256, NUM_CLASSES=10]

            loss = criterion(y_pred, label)

            # calculate accuracy
            top_y_pred = y_pred.argmax(
                1, keepdim=True
            )  # [BATCH_SIZE=256,1] each an int in [0, NUM_CLASSES-1=9]
            num_correct = top_y_pred.eq(label.view_as(top_y_pred)).sum()
            acc = num_correct.float() / label.shape[0]

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
