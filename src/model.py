import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''
    Class representing a convolutional residual block 
    '''

    def __init__(self, num_channels: int, use_stem: bool = False, strides: int = 1):
        '''
        Creates a new instance of a Residual Block
        @param: num_channels (int) - the number of output channels for all convolutions in 
            the block
        @param: use_stem (bool) - whether a 1x1 convolution is needed to downsample the
            residual
        @param: strides (int) - the number of strides to use in the convolutions, defaults to 1
        '''
        super().__init__()
        self.num_channels = num_channels
        self.use_stem = use_stem
        self.strides = strides

        self.conv1 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(
            num_channels, kernel_size=3, padding=1, stride=strides)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.ReLU(inplace=True)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        self.conv_stem = None
        if use_stem:
            self.conv_stem = nn.LazyConv2d(
                num_channels, kernel_size=1, stride=strides)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = inputs
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        if self.use_stem:
            # downsample skip connection
            shortcut = self.conv_stem(shortcut)

        # add in skip connection
        x += shortcut
        return self.out(x)


class ResNet(nn.Module):
    '''
    Class representing a full ResNet model
    '''

    def __init__(self, *args, **kwargs):
        '''
        returns an instance of a ResNet
        '''
        pass

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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
