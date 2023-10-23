import os

import wandb

os.environ['WANDB_MODE'] = 'online'
import sys
from argparse import Namespace

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(dir_path)

import time

import torch
from torch.cuda.amp import autocast

from src.core.utils import get_curtime
from src.distance.dist_net import DIST
from src.distance.train import train_wb as dist_train


def main():
    wandb.init(project='dist_consep_tune')
    dataset_name='consep'
    model_name = 'dist'
    dir_root = "/root/autodl-tmp/datasets/{}".format(dataset_name)
    save_dir  = "/root/autodl-tmp/train/model_data/{}/{}/{}".format(dataset_name,model_name, get_curtime())

    dist_train(wandb.config, dataset_name, dir_root, save_dir, )
    
    
if __name__ == "__main__":
    # Define sweep config
    sweep_configuration = {
        'method': 'random',
        'name': 'tune_ts',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': 
        {

            'epochs':{'value':100},
            'val_interval':{'value':1},
            'save_interval':{'value':50},
            'num_workers':{'value':8},

            'batch_size': {'values': [2, 4, 8, 16, 32]},
            'lr': {'distribution': 'uniform', 'max': 0.001, 'min': 0.00001},
            'weight_decay':{'distribution': 'uniform', 'max': 0.0001, 'min': 0.00001},
            'num_features':{'values':[3, 5, 6, 8, 10, 20, 30, 50]},
         },
        'early_terminate':{'type': 'hyperband', 'min_iter': 3}

    }


    sweep_id = wandb.sweep(
      sweep=sweep_configuration, 
      project='dist_consep_tune'
      )

    wandb.agent(sweep_id, function=main, count=50)
    
    
def main():
    wandb.init(project='dist_consep_tune')
    dataset_name='consep'
    model_name = 'dist'
    dir_root = "/root/autodl-tmp/viax/datasets/{}".format(dataset_name)
    log_dir  = "/root/autodl-tmp/train/viax/model_data/{}/{}/{}".format(dataset_name, model_name, get_curtime())

    num_classes = 5
    train(wandb.config, num_classes, model_name, dataset_name, dir_root, log_dir)
    

if __name__ == "__main__":
    sweep_configuration = read_yaml_to_dict('/root/autodl-tmp/viax/train/dist/dist_config_consep.yaml')
    sweep_id = wandb.sweep(
      sweep=sweep_configuration, 
      project='unet_consep_tune'
      )
    wandb.agent(sweep_id, function=main, count=50)
