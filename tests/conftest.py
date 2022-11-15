from typing import Iterable, Optional

import pytest
import torch

from src.model import StemConfig, ResidualBlockType


def check_conv_bias(iter: Iterable, use_bias: bool):
    """
    Checks whether all Conv2D modules in the model
    have or don't have a bias
    """
    for module in iter:
        if isinstance(module, torch.nn.Conv2d) and module.bias != use_bias:
            return False
    return True


def check_dropout(iter: Iterable, dropout: Optional[float] = None):
    """
    Checks to see if the passed in model has the designated dropout
    """
    for module in iter:
        if isinstance(module, torch.nn.Dropout2d) and dropout is None:
            return False
        elif isinstance(module, torch.nn.Dropout2d) and dropout and module.p != dropout:
            return False
    return True


@pytest.fixture
def base_stem_config():
    stem = StemConfig(64, kernel_size=3, stride=1, padding=1)
    return stem


@pytest.fixture
def base_architecture_dropout():
    arch = [
        (ResidualBlockType.BASIC, 1, 64, 0.5),
        (ResidualBlockType.BASIC, 1, 128, 0.5),
        (ResidualBlockType.BASIC, 1, 256, 0.5),
        (ResidualBlockType.BASIC, 1, 512, 0.5),
    ]
    return arch


@pytest.fixture
def base_architecture():
    arch = [
        (ResidualBlockType.BASIC, 1, 64),
        (ResidualBlockType.BASIC, 1, 128),
        (ResidualBlockType.BASIC, 1, 256),
        (ResidualBlockType.BASIC, 1, 512),
    ]
    return arch
