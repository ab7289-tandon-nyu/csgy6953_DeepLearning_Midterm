# How to load and transform CIFAR-10 data
# source: HW2 Q4 (AlexNet-like deep CNN classifies CIFAR-10 images)

import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import copy


## Download TRAIN (50K) data
ROOT = '.data'
train_data = datasets.CIFAR10(root = ROOT, train = True, download = True) # len = 50,000


## Prepare to normalize data (the same way for TRAIN and TEST)
def summarize_train_data(train_data):
    '''Compute means and standard deviations along the R,G,B channel'''
    means = train_data.data.mean(axis = (0, 1, 2)) / 255
    stds  = train_data.data.std( axis = (0, 1, 2)) / 255
    # EACH returns a tensor of shape (3,) = a vector of size 3 (= R,G,B)
    return means, stds
train_data_means, train_data_stds = summarize_train_data(train_data)


## Augment TRAIN data; Normalize data (the same way  for TRAIN and TEST)
from src.transforms import make_transforms
train_transforms, test_transforms = make_transforms(train_data_means, train_data_stds)


## Load and custom-partition data

# train_data is of length 50,000
train_data = datasets.CIFAR10(ROOT,train = True, download  = True, transform = train_transforms)
# test_data is of length 10,000
test_data  = datasets.CIFAR10(ROOT, train = False, download  = True, transform = test_transforms)

def partition_train_data(train_data, valid_ratio):
    '''partition TRAIN data into TRAIN and VALID'''

    # the partition:len(train_data) == num_valid_examples + num_train_examples
    num_valid_examples = int(len(train_data) * valid_ratio)
    num_train_examples = len(train_data) - num_valid_examples

    train_data, valid_data = \
    data.random_split(train_data, [num_train_examples, num_valid_examples])
    
    return train_data, copy.deepcopy(valid_data)

VALID_RATIO = 0.1
# 10% of TRAIN will become VALID
# 90% of TRAIN will remain TRAIN
train_data, valid_data = partition_train_data(train_data, VALID_RATIO)

# will Augment and Normalize VALID data the same way as we do TEST data
valid_data.dataset.transform = test_transforms

## Data loader

def data_loaders(train_data, valid_data, test_data, batch_size):

    # training requires shuffling
    train_iterator = \
    torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_iterator = \
    torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    test_iterator = \
    torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    return train_iterator, valid_iterator, test_iterator