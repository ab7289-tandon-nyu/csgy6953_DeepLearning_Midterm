import torch
import torchvision.transforms as transforms

from typing import Tuple


def make_transforms(means: torch.Tensor, std_devs: torch.Tensor) -> Tuple:
    '''
    added a comment
    Given a tensor of computed means and a tensor of computed standard devations,
    return's a tuple containing a train and test transform pipelines
    '''
    train_transforms = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                             std=std_devs)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                             std=std_devs)
    ])

    return train_transforms, test_transforms
