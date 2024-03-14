import glob
from itertools import chain
import os
import argparse
import random
import zipfile
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from linformer import Linformer
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.nn.functional as F
import io
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from vit_pytorch.efficient import ViT
#from vit_pytorch.efficient import ViT
from vit_pytorch.vit import ViT
from vit_pytorch.deepvit import DeepViT


import wandb

import sys
# import custom utilize

from util.utilize import *
import util.utilize as ut


# lr: 1e-3 3e-3 1e-4 3e-4
# split 0 1 2 run name: backbone lr loss split
# lr: 1e-3 3e-3 1e-4 3e-4
# split 0 1 2 run name: backbone lr loss split
lrs, splits , bal_aug , bal_val= [2e-3,1e-4], [0,1,2], [False], [False]
for isbalval in bal_val:
    for isaug in bal_aug:
        for lr in lrs:
            for split in splits:
                run_name = f"covid_vit_lr{lr}_split{split}_isaug{1 if isaug else 0}_isbalval{1 if isbalval else 0}"
                print(run_name)
                args = {
                    'device': torch.device("cuda:1"),
                    # 'model': get_model_octa_resume(outsize=5, path='/code/covid_ckpts/octa_split1/1527_val_acc0.694444477558136.pt', dropout=0.15), # 写一个函数获取不同的model和预训练model，这样方便些
                    # 'model': get_model_conv(pretrain_out=4,outsize=5, path='/code/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt'),
                    'model': get_vani(outsize=5, dropout=0.25),
                    # 'model': get_model_oct_withpretrain(pretrain_out=4,outsize=5, path='/code/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt', dropout=0.15),
                    'save_path': f'/code/covid_ckpts/oct_vani/', 
                    'bce_weight': 1,     
                    'epochs': 200, 
                    'lr': lr, 
                    'batch_size': 300, 
                    'datasets': get_dataUNI(split_idx=split, aug_class=isaug, bal_val = isbalval),
                    'vote_loader': DataLoader(get_dataUNI(split_idx=split, aug_class=isaug, bal_val = isbalval, infer_3d=True)[1], batch_size=1, shuffle=False),
                    'is_echo': False,
                    'optimizer': optim.Adam,
                    'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                    'train_loader': None,
                    'eval_loader': None,
                    'shuffle': True,
                    'is_MIX': True, # 是否使用mixloss input
                    # 'wandb': ['visual-intelligence-laboratory','delete_vit3d'],
                    'wandb': ['2065136374','oct_vani',run_name],
                    'metric_path': '/code/chen/COVID-Net/log/covid-vit-vani.csv'  , #所有实验统一保存结果的csv
                    'decay': 1e-3,
                }
                ut.device = args['device']
                # do not change the excution 
                torch.autograd.set_detect_anomaly(True)
                train_epoch(**args)