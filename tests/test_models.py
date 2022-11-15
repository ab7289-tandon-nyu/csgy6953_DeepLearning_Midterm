import pytest
import torch

from src.model import BottleneckResidualBlock, ResidualBlock, ResNet, StemConfig

from .conftest import check_conv_bias, check_dropout


def test_stem_creation():
    """
    Tests whether the stem config is created successfully
    """
    test_num_channels = 256
    test_kernel_size = 3
    test_padding = 1
    test_stride = 1

    config = StemConfig(
        num_channels=test_num_channels,
        kernel_size=test_kernel_size,
        stride=test_stride,
        padding=test_padding,
    )

    assert config.num_channels == test_num_channels
    assert config.kernel_size == test_kernel_size
    assert config.padding == test_padding
    assert config.stride == test_stride


def test_residual_block_init():
    """
    Tests that the initialization of the ResidualBlock works
    as expected with defaults applied
    """
    test_num_channels = 64

    block = ResidualBlock(test_num_channels)

    assert block.num_channels == test_num_channels
    assert block.use_stem == False
    assert block.strides == 1
    assert block.dropout == None
    assert check_conv_bias(block.parameters(), False)


def test_residual_block_init_with_extras():
    """
    Tests that the initialization of the ResidualBlock works
    as expected with defaults overridden
    """
    test_num_channesl = 64
    test_use_stem = True
    test_strides = 2
    test_dropout = 0.2
    test_bias = True

    block = ResidualBlock(
        test_num_channesl,
        use_stem=test_use_stem,
        strides=test_strides,
        dropout=test_dropout,
        use_bias=test_bias,
    )

    assert block.num_channels == test_num_channesl
    assert block.use_stem == test_use_stem
    assert block.strides == test_strides
    assert block.dropout is not None
    assert block.dropout.p == test_dropout
    assert check_conv_bias(block.parameters(), test_bias)


def test_residualblock_forward_no_downsample():
    """
    Tests that the forward method of Residual blocks work as
    expected
    """
    test_num_channels = 64

    block = ResidualBlock(test_num_channels)

    inputs = torch.ones((1, 64, 4, 4))
    output = block(inputs)
    assert output.size() == (1, 64, 4, 4)


def test_residualblock_forward_with_downsample():
    """
    Tests that the forward method of a ResidualBlock works
    as expected when we are downsampling
    """
    test_num_channels = 64

    block = ResidualBlock(test_num_channels, use_stem=True, strides=2)

    inputs = torch.ones((1, 64, 4, 4))
    outputs = block(inputs)
    assert outputs.size() == (1, 64, 2, 2)


def test_residualblock_bad_inputs():
    """
    Insures that the residualblock throws an error when we supply bad
    inputs
    """
    test_num_channels = 64

    block = ResidualBlock(test_num_channels)

    inputs = torch.ones((1, 3, 4, 4))
    with pytest.raises(RuntimeError):
        _ = block(inputs)


def test_bottleneck_init():

    test_num_channels = 128

    block = BottleneckResidualBlock(test_num_channels)

    assert block.num_channels == test_num_channels
    assert block.use_stem == False
    assert block.strides == 1
    assert block.factor == 4
    assert block.dropout1 is None
    assert block.dropout2 is None
    assert check_conv_bias(block.parameters(), False)
    assert check_dropout(block.parameters())


def test_bottlneck_init_with_extras():

    test_num_channels = 128
    test_use_stem = True
    test_strides = 2
    test_factor = 2
    test_dropout = 0.25
    test_bias = True

    block = BottleneckResidualBlock(
        test_num_channels,
        use_stem=test_use_stem,
        strides=test_strides,
        factor=test_factor,
        dropout=test_dropout,
        use_bias=test_bias,
    )

    assert block.num_channels == test_num_channels
    assert block.use_stem == test_use_stem
    assert block.strides == test_strides
    assert block.factor == test_factor
    assert block.dropout1 is not None
    assert block.dropout2 is not None
    assert check_conv_bias(block.parameters(), True)
    assert check_dropout(block.parameters(), test_dropout)


def test_bottleneck_forward():
    test_num_channels = 128

    block = BottleneckResidualBlock(test_num_channels)

    test_input = torch.ones((1, 128, 4, 4))
    outputs = block(test_input)
    assert outputs is not None
    assert outputs.size() == (1, 128, 4, 4)


def test_bottleneck_forward_downsample():

    test_num_channels = 128
    test_stride = 2
    test_use_stem = True

    block = BottleneckResidualBlock(
        test_num_channels, use_stem=test_use_stem, strides=test_stride
    )

    test_input = torch.ones((1, 128, 4, 4))
    outputs = block(test_input)
    assert outputs is not None
    assert outputs.size() == (1, 128, 2, 2)


def test_resnet_init(base_stem_config, base_architecture):
    """
    Tests the basic initialization of our ResNet
    """
    model = ResNet(base_architecture, stem_config=base_stem_config)
    assert model.use_bias == False
    assert model.stem is not None and isinstance(model.stem, torch.nn.Sequential)
    assert model.body is not None and isinstance(model.body, torch.nn.Sequential)
    assert model.classifier is not None and isinstance(
        model.classifier, torch.nn.Sequential
    )


def test_resnet_init_no_bias(base_stem_config, base_architecture):
    """
    Tests that the use_bias flag correctly removes the bias from all
    Conv2d modules
    """
    model = ResNet(base_architecture, stem_config=base_stem_config, use_bias=False)
    assert check_conv_bias(model.parameters(), False)
    model = ResNet(base_architecture, stem_config=base_stem_config)
    assert check_conv_bias(model.parameters(), False)


def test_resnet_init_bias(base_architecture, base_stem_config):
    """
    Tests that the use_bias flag correctly removes the bias from all
    Conv2d modules
    """
    model = ResNet(base_architecture, stem_config=base_stem_config, use_bias=True)
    assert check_conv_bias(model.parameters(), True)


def test_resnet_init_dropout(base_architecture_dropout, base_stem_config):
    """
    Tests that the Dropout parameters pass through the model correctly
    """
    model = ResNet(base_architecture_dropout, stem_config=base_stem_config)
    assert check_dropout(model.parameters(), base_architecture_dropout[0][2])


def test_resent_init_no_dropout(base_architecture, base_stem_config):
    """
    Tests that the Dropout parameters pass through the model correctly
    """
    model = ResNet(base_architecture, stem_config=base_stem_config)
    assert check_dropout(model.parameters())


def test_resnet_forward(base_architecture, base_stem_config):
    """
    Tests that the forward method for ResNets works as expected
    """
    model = ResNet(base_architecture, stem_config=base_stem_config)

    inputs = torch.ones((1, 3, 32, 32))
    outputs = model(inputs)
    assert outputs.size() == (1, 10)
