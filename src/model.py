from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ResidualBlockType(Enum):
    """
    Enum class to represent the residual block type for ResNet
    """

    BASIC = 0
    BOTTLENECK = 1


class LayerType(Enum):
    """
    Enum class to represent layer within for ResidualBlock and for BottleneckResidualBlock
    """

    # Disambiguation: here "layer" refers to the individual layer within a block,
    # not a "residual layer" containing one or more blocks
    CONV = 0


class LayerLoc(Enum):
    """
    Enum class to represent a layer's location within a block
    """

    MAIN_BLOCK_CONV1 = 0
    MAIN_BLOCK_CONV2 = 1
    MAIN_BLOCK_CONV3 = 2

    SHORTCUT_IDENTITY = 6   # identity
    SHORTCUT_CONV_STEM = 7


def generate_layer(
    block_type: ResidualBlockType,
    layer_type: LayerType, 
    layer_loc: LayerLoc, # position of this layer within the block starting from index 1 
    num_channels: int,
    main_block_kernel_size: int,
    strides: int = 1,
    factor: int = 4,
    use_bias: bool = False,
):
    """
    Returns a layer with the most appropriate parameters such as padding 
    """
    if block_type == ResidualBlockType.BASIC:
        if layer_type == LayerType.CONV:

            if main_block_kernel_size == 3:
                layer_locator = {
                    LayerLoc.MAIN_BLOCK_CONV1: nn.LazyConv2d(
                        num_channels,
                        kernel_size=3, padding=1, # ResidualBlock.conv1
                        stride=strides, bias=use_bias
                        ), 
                     LayerLoc.MAIN_BLOCK_CONV2: nn.LazyConv2d(
                        num_channels,
                        kernel_size=3, padding=1, # ResidualBlock.conv2
                        bias=use_bias
                        ),
                     LayerLoc.SHORTCUT_IDENTITY: nn.Identity(),
                     LayerLoc.SHORTCUT_CONV_STEM: nn.LazyConv2d(
                        num_channels, 
                        kernel_size=1, stride=strides, # ResidualBlock.conv_stem
                        bias=use_bias
                        )
                    }
            
            # partial solution 1:
            # fatal issue: when input is [256,1,1], MUST pad in order to apply 2x2 kernel
            if main_block_kernel_size == 2:
                layer_locator = {
                    LayerLoc.MAIN_BLOCK_CONV1: nn.LazyConv2d(
                        num_channels,
                        kernel_size=2, padding=1, # ResidualBlock.conv1
                        stride=strides, bias=use_bias
                        ), 
                     LayerLoc.MAIN_BLOCK_CONV2: nn.LazyConv2d(
                        num_channels,
                        kernel_size=2, padding=0, # ResidualBlock.conv2
                        bias=use_bias
                        ),
                     LayerLoc.SHORTCUT_IDENTITY: nn.Identity(),
                     LayerLoc.SHORTCUT_CONV_STEM: nn.LazyConv2d(
                        num_channels, 
                        kernel_size=1, stride=strides, # ResidualBlock.conv_stem
                        bias=use_bias
                        )
                    }
            
    if block_type == ResidualBlockType.BOTTLENECK:
        if layer_type == LayerType.CONV:

            if main_block_kernel_size == 3:
                layer_locator = {
                    LayerLoc.MAIN_BLOCK_CONV1: nn.LazyConv2d(
                        num_channels // factor,
                        kernel_size=1, padding=0, # BottleneckResidualBlock.conv1
                        bias=use_bias
                        ), 
                    LayerLoc.MAIN_BLOCK_CONV2: nn.LazyConv2d(
                        num_channels // factor,
                        kernel_size=3, padding=1, stride=strides, # BottleneckResidualBlock.conv2
                        bias=use_bias
                        ),
                    LayerLoc.MAIN_BLOCK_CONV3: nn.LazyConv2d(
                        num_channels, 
                        kernel_size=1, padding=0, # BottleneckResidualBlock.conv3
                        bias=use_bias
                        ),
                    LayerLoc.SHORTCUT_IDENTITY: nn.Identity(),
                    LayerLoc.SHORTCUT_CONV_STEM: nn.LazyConv2d(
                        num_channels, 
                        kernel_size=1, stride=strides, # BottleneckResidualBlock.conv_stem
                        bias=use_bias
                        )
                    }
    return layer_locator[layer_loc]
    

class ResidualBlock(nn.Module):
    """
    Class representing a convolutional residual block
    """

    def __init__(
        self,
        num_channels: int,
        use_stem: bool = False,
        strides: int = 1,
        dropout: Optional[float] = None,
        use_bias: bool = False,
        main_block_kernel_size: int = 3
    ):
        """
        Creates a new instance of a Residual Block
        @param: num_channels (int) - the number of output channels for all convolutions in
            the block
        @param: use_stem (bool) - whether a 1x1 convolution is needed to downsample the
            residual
        @param: strides (int) - the number of strides to use in the convolutions, defaults to 1
        @param: dropout (float) - if present, adds a dropout between the hidden layers
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_stem = use_stem
        self.strides = strides

        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.conv1 = generate_layer(
            block_type = ResidualBlockType.BASIC,
            layer_type = LayerType.CONV,
            layer_loc = LayerLoc.MAIN_BLOCK_CONV1,
            num_channels = num_channels,
            main_block_kernel_size = main_block_kernel_size,
            strides = strides,
            use_bias = use_bias,
        )
        self.conv2 = generate_layer(
            lock_type = ResidualBlockType.BASIC,
            layer_type = LayerType.CONV,
            layer_loc = LayerLoc.MAIN_BLOCK_CONV2,
            num_channels = num_channels,
            main_block_kernel_size = main_block_kernel_size,
            use_bias = use_bias,
        )
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.ReLU(inplace=True)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

        self.identity = None
        self.conv_stem = None
        if use_stem:
            self.conv_stem = generate_layer(
                block_type = ResidualBlockType.BASIC,
                layer_type = LayerType.CONV,
                layer_loc = LayerLoc.SHORTCUT_CONV_STEM,
                num_channels = num_channels, 
                main_block_kernel_size = main_block_kernel_size,
                strides = strides,
                use_bias = use_bias
            )
        else:
            self.identity = generate_layer(
                block_type = ResidualBlockType.BASIC,
                layer_type = LayerType.CONV,
                layer_loc = LayerLoc.SHORTCUT_IDENTITY,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = inputs

        x = self.relu(self.bn1(self.conv1(inputs)))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        
        if self.use_stem:
            # downsample skip connection
            shortcut = self.conv_stem(shortcut)
        else:
            shortcut = self.identity(shortcut)

        # add in skip connection
        x += shortcut
        return self.out(x)


class BottleneckResidualBlock(nn.Module):
    """
    Class representing a convolutional residual block with a bottleneck
    This class was built with reference to:
    https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """

    def __init__(
        self,
        num_channels: int,
        use_stem: bool = False,
        strides: int = 1,
        factor: int = 4,
        dropout: Optional[float] = None,
        use_bias: bool = False,
    ):
        """
        Creates a new instance of a Residual BottleNeck Block
        @param: num_channels (int) - the number of output channels for all convolutions in the block
        @param: use_stem (bool) - whether a 1x1 convolution is needed to downsample the residual
        @param: strides (int) - the number of strides to use in the convolutions, defaults to 1
        @param: factor (int) - the factor by which the input channels will be reduced for the bottleneck
        @param: dropout (float) - if present, adds a dropout between the hidden layers
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_stem = use_stem
        self.strides = strides
        self.factor = factor
        self.dropout1 = nn.Dropout(dropout) if dropout is not None else None
        self.dropout2 = nn.Dropout(dropout) if dropout is not None else None

        # First convolutional layer with normalization
        self.conv1 = nn.LazyConv2d(
            num_channels // factor, kernel_size=1, padding=0, bias=use_bias
        )
        self.bn1 = nn.LazyBatchNorm2d()

        # Second convolutional layer with normalization
        self.conv2 = nn.LazyConv2d(
            num_channels // factor,
            kernel_size=3,
            padding=1,
            stride=strides,
            bias=use_bias,
        )
        self.bn2 = nn.LazyBatchNorm2d()

        # Third convolutional layer with normalization
        self.conv3 = nn.LazyConv2d(
            num_channels, kernel_size=1, padding=0, bias=use_bias
        )
        self.bn3 = nn.LazyBatchNorm2d()

        self.relu = nn.ReLU(inplace=True)

        self.conv_stem = None
        if use_stem:
            # Bottleneck residual block
            self.conv_stem = nn.LazyConv2d(
                num_channels, kernel_size=1, stride=strides, bias=use_bias
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = inputs
        x = self.relu(self.bn1(self.conv1(inputs)))
        if self.dropout1 is not None:
            x = self.dropout1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout2 is not None:
            x = self.dropout2(x)
        x = self.bn3(self.conv3(x))
        if self.use_stem:
            # downsample skip connection
            shortcut = self.conv_stem(shortcut)

        # add in skip connection
        x += shortcut
        return self.relu(x)


class StemConfig:
    """
    convenience class to encapsulate configuration options
    for the ResNet stem
    """

    def __init__(self, num_channels, kernel_size, stride, padding):
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


def generate_block(
    block_type: ResidualBlockType,
    num_channels: int,
    use_stem: bool = False,
    strides: int = 1,
    factor: int = 4,
    dropout: Optional[float] = None,
    use_bias: bool = False,
):
    """
    Returns either a Residual Block or a ResidualBottleneck
    """
    if block_type == ResidualBlockType.BASIC:
        return ResidualBlock(
            num_channels,
            use_stem=use_stem,
            strides=strides,
            dropout=dropout,
            use_bias=use_bias,
        )
    else:
        return BottleneckResidualBlock(
            num_channels,
            use_stem=use_stem,
            strides=strides,
            factor=factor,
            dropout=dropout,
            use_bias=use_bias,
        )


class ResNet(nn.Module):
    """
    Class representing a full ResNet model
    """

    def __init__(
        self,
        architecture: List[Tuple[ResidualBlockType, int, int, float]],
        stem_config: Optional[StemConfig],
        output_size: int = 10,
        use_bias: bool = False,
        *args,
        **kwargs,
    ):
        """
        returns an instance of a ResNet
        """
        super().__init__()
        self.use_bias = use_bias
        if stem_config is not None:
            self.stem = self.create_stem(
                stem_config.num_channels,
                stem_config.kernel_size,
                stem_config.stride,
                stem_config.padding,
                use_bias=use_bias,
            )
        else:
            self.stem = self.create_stem(use_bias=use_bias)
        self.classifier = self.create_classifier(output_size, use_bias=use_bias)

        self.body = nn.Sequential()
        for idx, block_def in enumerate(architecture):
            self.body.add_module(
                f"block_{idx+2}",
                self.create_block(
                    *block_def, first_block=(idx == 0), use_bias=use_bias
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the inputs through the network
        """
        x = self.stem(inputs)
        x = self.body(x)
        return self.classifier(x)

    def create_stem(
        self,
        num_channels: int = 64,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
        use_bias: bool = False,
    ) -> nn.Sequential:
        """
        Creates a sequential stem as the first component of the model
        """
        return nn.Sequential(
            nn.LazyConv2d(
                num_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=use_bias,
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(inplace=True),
        )

    def create_classifier(
        self, num_classes: int, use_bias: bool = False
    ) -> nn.Sequential:
        """
        Creates a sequential classifier head at the very
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LazyLinear(num_classes)
        )

    def create_block(
        self,
        block_type: ResidualBlockType,
        num_residuals: int,
        num_channels: int,
        dropout: Optional[float] = None,
        first_block: bool = False,
        use_bias: bool = False,
    ) -> nn.Sequential:
        """
        Given our inputs, generates either a ResidualBlock or ResidualBottleNeck and addes it to our
        sequence of layers
        """
        layer = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                layer.append(
                    generate_block(
                        block_type,
                        num_channels,
                        use_stem=True,
                        strides=2,
                        dropout=dropout,
                        use_bias=use_bias,
                    )
                )
            else:
                layer.append(
                    generate_block(
                        block_type, num_channels, dropout=dropout, use_bias=use_bias
                    )
                )
        return nn.Sequential(*layer)
