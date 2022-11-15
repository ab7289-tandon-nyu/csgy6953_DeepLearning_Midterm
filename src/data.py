import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import copy
from typing import Tuple, Callable


## 2. Prepare to normalize data (the same way for TRAIN and TEST)
def summarize_train_data(
    train_data: datasets.cifar.CIFAR10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute means and standard deviations along the R,G,B channel"""
    means = train_data.data.mean(axis=(0, 1, 2)) / 255
    stds = train_data.data.std(axis=(0, 1, 2)) / 255
    # EACH returns a tensor of shape (3,) = a vector of size 3 (= R,G,B)
    return means, stds


## 4. Load and custom-partition data
def partition_train_data(train_data, valid_ratio):
    """partition TRAIN data into TRAIN and VALID"""

    # the partition:len(train_data) == num_valid_examples + num_train_examples
    num_valid_examples = int(len(train_data) * valid_ratio)
    num_train_examples = len(train_data) - num_valid_examples

    train_data, valid_data = data.random_split(
        train_data, [num_train_examples, num_valid_examples]
    )

    return train_data, copy.deepcopy(valid_data)


## 1. through 4.
def get_transformed_data(
    make_transforms: Callable, valid_ratio: float
) -> Tuple[
    torch.utils.data.dataset.Subset,
    torch.utils.data.dataset.Subset,
    datasets.cifar.CIFAR10,
]:
    """
    Where transform = augment & normalize
    @param: make_transforms (Callable) - how to augment & normalize train and non-train data
        (e.g. this function defined in `from src.transforms import make_transforms`)
    @param: valid_ratio (float) - the share of train_data that will redesignate as valid_data
    """

    ## 1. Download TRAIN (50K) data
    ROOT = ".data"
    train_data = datasets.CIFAR10(root=ROOT, train=True, download=True)  # len=50K

    ## 2. Prepare to normalize data (the same way for TRAIN and TEST)
    train_data_means, train_data_stds = summarize_train_data(train_data)

    ## 3. Augment TRAIN data; Normalize data (the same way  for TRAIN and TEST)
    train_transforms, test_transforms = make_transforms(
        train_data_means, train_data_stds
    )

    ## 4. Load and custom-partition data

    # load TRAIN (50K) and TEST (10K) data
    train_data = datasets.CIFAR10(
        ROOT, train=True, download=True, transform=train_transforms
    )
    test_data = datasets.CIFAR10(
        ROOT, train=False, download=True, transform=test_transforms
    )

    # custom-partition data: TRAIN -> VALID(valid_ratio) & TRAIN(1 - valid_ratio)
    train_data, valid_data = partition_train_data(train_data, valid_ratio)

    # will transform (= augment & normalize) VALID data the same way as we do TEST data
    valid_data.dataset.transform = test_transforms

    return train_data, valid_data, test_data


## 5. Data loader
def make_data_loaders(
    train_data: torch.utils.data.dataset.Subset,
    valid_data: torch.utils.data.dataset.Subset,
    test_data: datasets.cifar.CIFAR10,
    batch_size: int,
) -> Tuple[
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
]:

    # training requires shuffling
    train_iterator = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    valid_iterator = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False
    )

    test_iterator = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_iterator, valid_iterator, test_iterator
