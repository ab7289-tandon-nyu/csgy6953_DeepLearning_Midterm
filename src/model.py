import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional


class ResidualBlock(nn.Module):
    '''
    Class representing a convolutional residual block 
    '''

    def __init__(self, num_channels: int, use_stem: bool = False, strides: int = 1, dropout: Optional[float] = None, kernel_size: int = 3):
        '''
        Creates a new instance of a Residual Block
        @param: num_channels (int) - the number of output channels for all convolutions in 
            the block
        @param: use_stem (bool) - whether a 1x1 convolution is needed to downsample the
            residual
        @param: strides (int) - the number of strides to use in the convolutions, defaults to 1
        @param: dropout (float) - if present, adds a dropout between the hidden layers
        '''
        super().__init__()
        self.num_channels = num_channels
        self.use_stem = use_stem
        self.strides = strides

        self.kernel_size = kernel_size

        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        if self.kernel_size == 3:
            self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)

        # partial solution 1:
        if kernel_size == 2:
            self.conv1 = nn.LazyConv2d(num_channels, kernel_size=2, padding=1, stride=strides)
            self.conv2 = nn.LazyConv2d(num_channels, kernel_size=2, padding=0)
            # fetal issue: when input is [256,1,1], MUST pad in order to apply 2x2 kernel

        # # partial solution 2:
        # if self.kernel_size == 2:
        #     self.conv1 = nn.LazyConv2d(num_channels, kernel_size=2, padding=1, stride=strides)
        #     self.conv2 = nn.LazyConv2d(num_channels, kernel_size=2, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.out = nn.ReLU(inplace=True)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        self.conv_stem = None
        if use_stem:

            if kernel_size == 3:
                self.conv_stem = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
            if kernel_size == 2:
                self.conv_stem = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        print()
        print('--------ResidualBlock:--------')
        print(f'(use_stem={self.use_stem})')
        print('block input:', inputs.shape)

        shortcut = inputs

        print()
        print('F(x):', self.conv1)
        x = self.relu(self.bn1(self.conv1(inputs)))
        print('output:', x.shape)

        if self.dropout is not None:
            x = self.dropout(x)
        
        print()
        print('F(x):', self.conv2)
        x = self.bn2(self.conv2(x))
        print('output:', x.shape)

        if self.use_stem:
            # downsample skip connection
            print()
            print('S(x):', self.conv_stem)
            shortcut = self.conv_stem(shortcut)
            print('output:', shortcut.shape)

        # partial solution 2:
        # if self.kernel_size == 2:
        #     if x.shape[-1] > shortcut.shape[-1]:
        #         x = F.pad(x, pad = (-1, -1, -1, -1))
        #         print('negative padding =>', x.shape)
        #         # fetal error: when F(x)'s output becomes [512,2,2] whiel input was [256,1,1], this would reduce F(x)'s output to [512,0,0]
 
        # add in skip connection
        x += shortcut

        return self.out(x)


class StemConfig:
    '''
    convenience class to encapsulate configuration options
    for the ResNet stem
    '''

    def __init__(self, num_channels, kernel_size, stride, padding):
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class ResNet(nn.Module):
    '''
    Class representing a full ResNet model
    '''

    def __init__(self, architecture: List[Tuple[int, int, float]], stem_config: Optional[StemConfig], output_size: int = 10, *args, **kwargs):
        '''
        returns an instance of a ResNet
        '''
        super().__init__()
        if stem_config is not None:
            self.stem = self.create_stem(
                stem_config.num_channels,
                stem_config.kernel_size,
                stem_config.stride,
                stem_config.padding
            )
        else:
            self.stem = self.create_stem()
        self.classifier = self.create_classifier(output_size)

        self.body = nn.Sequential()
        for idx, block_def in enumerate(architecture):
            self.body.add_module(
                f"block_{idx+2}", self.create_block(*block_def, first_block=(idx == 0)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the inputs through the network
        """
        x = self.stem(inputs)
        x = self.body(x)
        return self.classifier(x)

    def create_stem(self, num_channels: int = 64, kernel_size: int = 7, stride: int = 2, padding: int = 3) \
            -> nn.Sequential:
        """
        Creates a sequential stem as the first component of the model
        """
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=kernel_size,
                          padding=padding, stride=stride),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def create_classifier(self, num_classes: int) -> nn.Sequential:
        '''
        Creates a sequential classifier head at the very 
        '''
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def create_block(self, num_residuals: int, num_channels: int, dropout: float, kernel_size: int, first_block: bool = False) -> nn.Sequential:
        layer = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                layer.append(ResidualBlock(num_channels, dropout=dropout, kernel_size=kernel_size, use_stem=True, strides=2))
            else:
                layer.append(ResidualBlock(num_channels, dropout=dropout, kernel_size=kernel_size))
        return nn.Sequential(*layer)
