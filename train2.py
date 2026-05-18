import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from util.utilize import *
import util.utilize as ut


# Hyperparameter grid search configuration — fine-tuning with pretrained backbone
# lr: 1e-3 3e-3 1e-4 3e-4
# split 0 1 2 run name: backbone lr loss split
lrs, splits, bal_aug, bal_val = [5e-4], [0, 1, 2], [True], [True]
for isbalval in bal_val:
    for isaug in bal_aug:
        for lr in lrs:
            for split in splits:
                run_name = f"covid_vit_lr{lr}_split{split}_isaug{1 if isaug else 0}_isbalval{1 if isbalval else 0}"
                print(run_name)
                args = {
                    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    # 'model': get_model_octa_resume(outsize=5, path='<ckpt_path>', dropout=0.15),
                    # 'model': get_vani(outsize=5, dropout=0.25),
                    # 'model': get_model_oct_withpretrain(pretrain_out=2, outsize=5, path='<pretrain_ckpt_path>', dropout=0.15),
                    'model': get_model_oct_withpretrain(pretrain_out=4, outsize=5, path='<pretrain_ckpt_path>', dropout=0.45),
                    'save_path': './checkpoints/randz-bal-all/',
                    'bce_weight': 1,
                    'epochs': 300,
                    'lr': lr,
                    'batch_size': 300,
                    'datasets': get_dataUNI(split_idx=split, aug_class=isaug, bal_val=isbalval),
                    'vote_loader': DataLoader(get_dataUNI(split_idx=split, aug_class=isaug, bal_val=isbalval, infer_3d=True)[1], batch_size=1, shuffle=False),
                    'is_echo': False,
                    'optimizer': optim.Adam,
                    'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                    'train_loader': None,
                    'eval_loader': None,
                    'shuffle': True,
                    'is_MIX': True,  # use mixloss input
                    'wandb': ['<wandb_entity>', '<wandb_project>', run_name],
                    'metric_path': './log/metrics.csv',
                    'decay': 1e-3,
                }
                ut.device = args['device']
                torch.autograd.set_detect_anomaly(True)
                train_epoch(**args)