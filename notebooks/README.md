### wandb.ai Model Tracking

#### Set up


On the command line
```
pip install wandb
wandb login
```

Upon CLI login, the service will request a key for account authorization. Once you have logged into wandb.ai, find this key with the instructions on this page 

https://wandb.ai/dlf22_mini_project

#### Usage

For all Python notebooks
```python
import wandb

# Make sure the "entity" is spelled correctly, that is our team name
# https://docs.wandb.ai/ref/python/init
wandb.init(name="RUN_NAME_HERE", project="PROJECT_NAME_HERE", entity="dlf22_mini_project")

# Logging
# https://docs.wandb.ai/ref/python/log
wandb.log({"loss": loss})

# Optional
# https://docs.wandb.ai/ref/python/watch
# wandb.watch(model)
```

### Load and Transform CIFAR-10 Data

Source: HW2 Q4 (AlexNet-like deep CNN classifies CIFAR-10 images)

Precondition: `csgy6953_DeepLearning_Midterm.git` was to local directory

Note: **definition of `VALID_RAIO` differs from that in Source**

```
from src.transforms import make_transforms
from src.data       import get_transformed_data
from src.data       import make_data_loaders

BATCH_SIZE  = 256
VALID_RATIO = 0.1  # 10% of TRAIN becomes VALID; 90% of TRAIN remains TRAIN

train_data, valid_data, test_data = \
get_transformed_data(make_transforms = make_transforms, valid_ratio = VALID_RATIO)

train_iterator, valid_iterator, test_iterator = \
make_data_loaders(train_data, valid_data, test_data, BATCH_SIZE)
```