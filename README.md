# ViT net for OCTA saturation  
The net is based on [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
## Environment 
check repo of lucidrains/vit-pytorch
```
git clone https://github.com/hycarbon-b/oct_vit.git
## Train
```

```
1. modify the datasets func to load data 
2. modify args dict in train.py
 args = {
                    'device': torch.device("cuda:1"),
                    # 'model': get_model_octa_resume(outsize=5, path='ckpt_path', dropout=0.15),
                    # 'model': get_model_conv(pretrain_out=4,outsize=5, path='/code/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt'),
                    'model': get_vani(outsize=5, dropout=0.25),
                    # 'model': get_model_oct_withpretrain(pretrain_out=4,outsize=5, path='/code/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt', dropout=0.15),
                    'save_path': 'save_path', 
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
                    'is_MIX': True, # use mixloss input
                    'wandb': ['wandb account','project name',run_name],
                    'decay': 1e-3,
                }
```

```
python train.py
```
## Usage of mixloss
mixloss() take inputs of class logits +  regression value as input, and implement $BCEwithlogits() + MSELoss()$ as criterion for backpropagation 
$$mixloss = \alpha * bce + (1-\alpha) * mse $$
Tips: it supports bal-bce
usage: 
```
import oct_vit/util/utilize as util     # ensure the current work folder containning oct_vit

criterion = mixloss(a, pos_weight)     # a is the weight of bce
# target = [0.9, 0.2, 0.1, 0.2, 98]   4class logits + regression value
# label = [1, 0, 0, 0, 97]
loss = mixloss(target, loss)
```



