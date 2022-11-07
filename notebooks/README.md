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
