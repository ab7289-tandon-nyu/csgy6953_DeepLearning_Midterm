import torch
from torch import nn

# precondition: any lazy convolutional layers 
# have been converted to normal convolution layers, 
# for example by passing a dummy input tensor through the model 
def initialize_parameters(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    num_parameters = 0
    num_parameters_requiring_grad = 0
    for p in model.parameters():
        numel = p.numel()
        num_parameters += numel
        if p.requires_grad:
            num_parameters_requiring_grad += numel
    return num_parameters, num_parameters_requiring_grad
