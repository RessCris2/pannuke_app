# -*- coding:utf-8 -*-
import importlib
import random
import cv2
import numpy as np
# from dataset import get_dataset

import torch.optim as optim
import sys
sys.path.append("/root/autodl-tmp/archive/v2/models/hovernet/")
import time
from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events
from models.targets import gen_targets, prep_sample
from models.net_desc import create_model
from models.run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output
sys.path.append("/root/autodl-tmp/archive/metrics/")
from utils import rm_n_mkdir

def get_config(nr_type, mode):
    """
        add 'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode
        mode 只能取值 fast 或者 original。

    Args:
        nr_type:
        mode:

    Returns:

    """
    return {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        "phase_list": [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, nr_types=nr_type, 
                            freeze=True, mode=mode
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-4,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        "pretrained": "/root/autodl-tmp/com_models/hover_net/pretrained/ImageNet-ResNet50-Preact_pytorch.tar",
                        # 'pretrained': None,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 8, "valid": 8,},  # engine name : value
                "nr_epochs": 50,
            },
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, nr_types=nr_type, 
                            freeze=False, mode=mode
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-4,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        "pretrained": -1,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 4, "valid": 2,}, # batch size per gpu
                "nr_epochs": 50,
            },
        ],
        # ------------------------------------------------------------------
        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        "run_engine": {
            "train": {
                # TODO: align here, file path or what? what about CV?
                "dataset": "consep",  # whats about compound dataset ?
                "nr_procs": 16,  # number of threads for dataloader
                "run_step": train_step,  # TODO: function name or function variable ?
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "consep",  # whats about compound dataset ?
                "nr_procs": 2,  # number of threads for dataloader
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, nr_types=nr_type)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }


# for pannuke
class Config():
    """Configuration file."""

    def __init__(self, nr_type, train_dir_list, valid_dir_list, log_dir, model_mode='fast', type_classification = True
                ):
        """
            nr_type = 6 # number of nuclear types (including background)
            model_mode not in ["original", "fast"]:
                pannuke: fast
                monusac: fast
                consep: original
                cpm17: original
                    aug_shape = [540, 540]
                    act_shape = [256, 256]
                    out_shape = [164, 164]
            type_classification:
                pannuke: True
                monusac: True
                consep: True
                cpm17: False
        """
        self.seed = 10
        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False
        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = type_classification

        model_mode = model_mode # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        
        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        if model_mode == "original":
            aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
            act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
            out_shape = [80, 80] # patch shape at output of network
            
        elif model_mode == "fast":
            aug_shape = [540, 540]
            act_shape = [256, 256]
            out_shape = [164, 164]

        if model_mode == "original":
            if act_shape != [270, 270] or out_shape != [80, 80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256, 256] or out_shape != [164, 164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        # self.dataset_name = "pannuke" # extracts dataset info from dataset.py
        # self.log_dir = "logs/pannuke/" # where checkpoints will be saved
        
#         self.dataset_name = dataset_name # extracts dataset info from dataset.py
        self.log_dir = log_dir # where checkpoints will be saved
        self.train_dir_list = train_dir_list
        self.valid_dir_list = valid_dir_list
        
        # paths to training and validation patches
#         self.train_dir_list = [
#             "/root/autodl-tmp/com_models/hover_net/dataset/training_data/{}/train/".format(self.dataset_name)
#         ]
#         self.valid_dir_list = [
#             "/root/autodl-tmp/com_models/hover_net/dataset/training_data/{}/valid/".format(self.dataset_name)
#         ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        # self.dataset = get_dataset(self.dataset_name)
        self.model_config = get_config(nr_type, model_mode)



if __name__ == "__main__":
    dataset_name = 'pannuke'
    nr_type = 6
    train_dir_list = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/train".format(dataset_name)
    valid_dir_list = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/val".format(dataset_name)

    time_str = time.strftime("%Y%m%d%H%M", time.localtime())
    log_dir = "/root/autodl-tmp/archive/v2/model_data/{0}/{1}".format(dataset_name, time_str)
    rm_n_mkdir(log_dir)
    # self.dataset 可能有雷