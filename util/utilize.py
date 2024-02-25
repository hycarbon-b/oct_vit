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
import sys
sys.path.append('/code/chen/COVID-Net')
from vit_pytorch.vit import ViT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch import ViT3

import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import queue
device = torch.device("cuda:2")
best = 0.5
best_tp = 0.1
def metric(y_true, y_pred):
    # 通过confusing matrix计算sensitivity、specificity
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 计算每个类别的sensitivity TP / TP+FN
    fp = cm.sum(axis=0)- np.diag(cm)
    fn = cm.sum(axis=1)- np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum()-(fp + fn + tp)

    sensitivity = np.where(tp+fn == 0, 0, tp/(tp+fn))
    specificity = np.where(tn+fp == 0, 0, tn/(tn+fp))
    trp = np.mean(sensitivity)
    trn = np.mean(specificity)
    print(f"TPR: {trp}, TNR: {trn}")
    return trp, trn



def get_dataOCT(path):
    '''
    get all Dataset needed 
    --oct
        --train(path)
            -class1
            -class2
    '''
    cls = os.listdir(path) # get the class name(folders name)
    pair = {i:cls.index(i) for i in cls}
    datalist = glob.glob(path+'**/*.jpeg',recursive=True)
    datalistOCT = []
    for i in cls:
        files = os.listdir(os.path.join(path,i))
        files = [os.path.join(path,i,j) for j in files]
        datalistOCT.extend(random.sample(files,8616))
    random.shuffle(datalistOCT)
    print('oct_kaggle sample count: ',len(datalistOCT))
    labels = [pair[(i.split('/')[-2])] for i in datalistOCT]
    # return datalistOCT, labels# oct_kaggledataset
    train_list, label_list = datalistOCT, labels
    train_list, test_list, train_label, test_label = train_test_split(train_list, label_list, 
                                                                    test_size=0.15, 
                                                                    random_state=33)
    train_data = oct_kaggleDataset(train_list, train_label)
    valid_data = oct_kaggleDataset(test_list,test_label)
    test_data = oct_kaggleDataset(test_list, label_list)
    return [train_data, valid_data, test_data]

def get_dataOCTA(path='/code/images-with-labels/'):
    '''
    get all Dataset needed
    这是我自己做的增强的数据集封装，实验用get_dataUNI()
    '''

    # trian_list contain file path of the images
    train_list = glob.glob(path+'**/*.nii.gz', recursive=True)
    # 一个patient一个文件夹
    sample_folder = set(['/'.join(i.split('/')[:-1]) for i in train_list])
    train_sample_folder, test_sample_folder = train_test_split(list(sample_folder), test_size=0.2, random_state=33)
    # 没有区分OS OD
    # train_list = [os.path.join(i,sorted(os.listdir(i), key=timing)[-1]) for i in sample_folder]
    # 区分OS OD
    train_list = []
    # for train
    for i in train_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        train_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    # load label from train_list and excel
    train_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in train_list[:]]
    train_list, train_label = balance(train_list, train_label)

    # for test
    test_list = []
    for i in test_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        test_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    test_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in test_list[:]]
    test_list, test_label = balance(test_list, test_label)

    # octa dataset
    train_data = CovidDataset(train_list, train_label)
    valid_data = CovidDataset(test_list,test_label)
    test_data = CovidDataset(test_list, train_label)
    return [train_data, valid_data, train_list, test_list, train_label, test_label]

def get_dataOCTA_many(path='/code/images-with-labels/'):
    '''
    get all Dataset needed
    这是我自己做的增强的数据集封装，实验用get_dataUNI()
    强行扩充数据集
    '''

    # trian_list contain file path of the images
    train_list = glob.glob(path+'**/*.nii.gz', recursive=True)
    # 一个patient一个文件夹
    sample_folder = set(['/'.join(i.split('/')[:-1]) for i in train_list])
    train_sample_folder, test_sample_folder = train_test_split(list(sample_folder), test_size=0.2, random_state=33)
    # 没有区分OS OD
    # train_list = [os.path.join(i,sorted(os.listdir(i), key=timing)[-1]) for i in sample_folder]
    # 区分OS OD
    train_list = []
    # for train
    for i in train_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        train_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    # load label from train_list and excel
    train_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in train_list[:]]
    train_list, train_label = many(train_list, train_label)

    # for test
    test_list = []
    for i in test_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        test_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    test_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in test_list[:]]
    test_list, test_label = balance(test_list, test_label)

    # octa dataset
    train_data = CovidDataset(train_list, train_label)
    valid_data = CovidDataset(test_list,test_label)
    test_data = CovidDataset(test_list, train_label)
    return [train_data, valid_data, train_list, test_list, train_label, test_label]
def get_3d16(path='/code/images-with-labels/', split_idx = 1):
    '''
    get all Dataset needed
    这是组内统一 3-fold 的train-test split
    
    '''
    # read from split file(train_1 test_1...)
    with open(f'/code/train_{split_idx}', 'r') as f:
        patients = f.readlines() # patients[0] = 'xxxx\n'
    patients = [i.replace('\n', '') for i in patients]
    train_sample_folder = [os.path.join(path, i) for i in patients]
    
    with open(f'/code/test_{split_idx}', 'r') as f:
        patients = f.readlines() # patients[0] = 'xxxx\n'
    patients = [i.replace('\n', '') for i in patients]
    test_sample_folder = [os.path.join(path, i) for i in patients]


    # 没有区分OS OD
    # train_list = [os.path.join(i,sorted(os.listdir(i), key=timing)[-1]) for i in sample_folder]
    # 区分OS OD
    train_list = []
    # for train
    for i in train_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        train_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    # load label from train_list and excel
    train_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in train_list[:]]
    train_list, train_label = many(train_list, train_label)

    # for test
    test_list = []
    for i in test_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        test_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    test_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in test_list[:]]
    test_list, test_label = balance(test_list, test_label)
    # 修改label_list 为 [类别，gt] 4class：88 92 95 100 右闭区间
    for i in range(len(train_label)):
        if train_label[i] < 89:
            train_label[i] = [0, train_label[i]]
        elif train_label[i] < 93:
            train_label[i] = [1, train_label[i]]
        elif train_label[i] < 96:
            train_label[i] = [2, train_label[i]]
        else:
            train_label[i] = [3, train_label[i]]
    for i in range(len(test_label)):
        if test_label[i] < 89:
            test_label[i] = [0, test_label[i]]
        elif test_label[i] < 93:
            test_label[i] = [1, test_label[i]]
        elif test_label[i] < 96:
            test_label[i] = [2, test_label[i]]
        else:
            test_label[i] = [3, test_label[i]]

    # octa dataset 4 class
    train_data = octa4Dataset3d(train_list, train_label)
    valid_data = octa4Dataset3d(test_list,test_label)
    return [train_data, valid_data, train_list, test_list, train_label, test_label]


# 我自己用的数据增强使用到的两个函数 
def timing(s):
    '''从文件名中提取时间， 用于排序'''
    s = s.replace('-', '_').split('_')
    return s[-6] + s[-8].rjust(2, '0') + s[-7].rjust(2, '0') + s[-5].rjust(2, '0') + s[-4].rjust(2, '0') + s[-3].rjust(2, '0')
def balance(train_list, label_list):
    ''' 用于离散取值平衡 '''
    # 去重
    all = list(set(label_list))
    times = [label_list.count(i) for i in all]
    new_train=train_list
    new_label=label_list
    for i in all:
        tmp_train = []
        for t in range(len(label_list)):
            if label_list[t]==i:
                tmp_train.append(train_list[t])
        for j in range(max(times)-label_list.count(i)):
            new_train.append(tmp_train[random.randint(0,len(tmp_train)-1)]) 
            new_label.append(i)
    return new_train, new_label
def classbalance(train_list, labels):
    ''' 用于类别间平衡 '''
    # 去重
    class_list = labels.copy()
    for i in range(len(class_list)):
        if class_list[i] < 89:
            class_list[i] = 0
        elif class_list[i] < 93:
            class_list[i] = 1
        elif class_list[i] < 96:
            class_list[i] = 2
        else:
            class_list[i] = 3
    all = list(set(class_list))
    times = [class_list.count(i) for i in all]

    new_train=train_list
    new_label=labels
    for i in all:
        tmp_train,tmp_lable = [],[]
        for t in range(len(class_list)):
            if class_list[t]==i:
                tmp_train.append(train_list[t])
                tmp_lable.append(labels[t])
        for j in range(max(times)-class_list.count(i)):
            idx = random.randint(0,len(tmp_train)-1)
            new_train.append(tmp_train[idx]) 
            new_label.append(tmp_lable[idx])
    return new_train, new_label

def many(train_list, label_list, num=100):
    ''' 和balance一样， 强行扩充数据集到指定数目 '''
    # 去重
    all = list(set(label_list))
    times = [label_list.count(i) for i in all]
    new_train=train_list
    new_label=label_list
    for i in all:
        tmp_train = []
        for t in range(len(label_list)):
            if label_list[t]==i:
                tmp_train.append(train_list[t])
        for j in range(num-label_list.count(i)):
            new_train.append(tmp_train[random.randint(0,len(tmp_train)-1)]) 
            new_label.append(i)
    return new_train, new_label

def get_dataUNI(path='/code/images-with-labels/', split_idx = 1):
    '''
    get all Dataset needed
    这是组内统一 3-fold 的train-test split
    
    '''
    # read from split file(train_1 test_1...)
    with open(f'/code/train_{split_idx}', 'r') as f:
        patients = f.readlines() # patients[0] = 'xxxx\n'
    patients = [i.replace('\n', '') for i in patients]
    train_sample_folder = [os.path.join(path, i) for i in patients]
    
    with open(f'/code/test_{split_idx}', 'r') as f:
        patients = f.readlines() # patients[0] = 'xxxx\n'
    patients = [i.replace('\n', '') for i in patients]
    test_sample_folder = [os.path.join(path, i) for i in patients]


    # 没有区分OS OD
    # train_list = [os.path.join(i,sorted(os.listdir(i), key=timing)[-1]) for i in sample_folder]
    # 区分OS OD
    train_list = []
    # for train
    for i in train_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        train_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    # load label from train_list and excel
    train_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in train_list[:]]
    train_list, train_label = classbalance(train_list, train_label)
    train_list, train_label = many(train_list, train_label)

    # for test
    test_list = []
    for i in test_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        test_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    test_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in test_list[:]]
    # test_list, test_label = balance(test_list, test_label)
    # 修改label_list 为 [类别，gt] 4class：88 92 95 100 右闭区间
    for i in range(len(train_label)):
        if train_label[i] < 89:
            train_label[i] = [0, train_label[i]]
        elif train_label[i] < 93:
            train_label[i] = [1, train_label[i]]
        elif train_label[i] < 96:
            train_label[i] = [2, train_label[i]]
        else:
            train_label[i] = [3, train_label[i]]
    for i in range(len(test_label)):
        if test_label[i] < 89:
            test_label[i] = [0, test_label[i]]
        elif test_label[i] < 93:
            test_label[i] = [1, test_label[i]]
        elif test_label[i] < 96:
            test_label[i] = [2, test_label[i]]
        else:
            test_label[i] = [3, test_label[i]]

    # octa dataset 4 class
    train_data = octa4Dataset(train_list, train_label)
    valid_data = octa4Dataset(test_list, test_label)
    return [train_data, valid_data, train_list, test_list, train_label, test_label]

    
class randDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
            
    def __getitem__(self, idx):
        # class to one-hot 
        label = torch.zeros(5) # 前四位是类别，最后一位是regression 
        label[self.labels[idx][0]] = 1
        label[4] = self.labels[idx][1]
        return torch.randn(3,224,224).to(device), label.to(device)
def get_dataRand(path='/code/images-with-labels/', split_idx = 1):
    '''
    get all Dataset needed
    这是组内统一 3-fold 的train-test split
    
    '''
    # read from split file(train_1 test_1...)
    with open(f'/code/train_{split_idx}', 'r') as f:
        patients = f.readlines() # patients[0] = 'xxxx\n'
    patients = [i.replace('\n', '') for i in patients]
    train_sample_folder = [os.path.join(path, i) for i in patients]
    
    with open(f'/code/test_{split_idx}', 'r') as f:
        patients = f.readlines() # patients[0] = 'xxxx\n'
    patients = [i.replace('\n', '') for i in patients]
    test_sample_folder = [os.path.join(path, i) for i in patients]


    # 没有区分OS OD
    # train_list = [os.path.join(i,sorted(os.listdir(i), key=timing)[-1]) for i in sample_folder]
    # 区分OS OD
    train_list = []
    # for train
    for i in train_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        train_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    # load label from train_list and excel
    train_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in train_list[:]]
    train_list, train_label = many(train_list, train_label)

    # for test
    test_list = []
    for i in test_sample_folder:
        os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
        od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
        test_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
    ex = pd.read_excel('/code/Sleep-results.xlsx')
    test_label = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in test_list[:]]
    test_list, test_label = balance(test_list, test_label)
    # 修改label_list 为 [类别，gt] 4class：88 92 95 100 右闭区间
    for i in range(len(train_label)):
        if train_label[i] < 89:
            train_label[i] = [0, train_label[i]]
        elif train_label[i] < 93:
            train_label[i] = [1, train_label[i]]
        elif train_label[i] < 96:
            train_label[i] = [2, train_label[i]]
        else:
            train_label[i] = [3, train_label[i]]
    for i in range(len(test_label)):
        if test_label[i] < 89:
            test_label[i] = [0, test_label[i]]
        elif test_label[i] < 93:
            test_label[i] = [1, test_label[i]]
        elif test_label[i] < 96:
            test_label[i] = [2, test_label[i]]
        else:
            test_label[i] = [3, test_label[i]]

    # octa dataset 4 class
    train_data = randDataset(train_list, train_label)
    valid_data = randDataset(test_list, test_label)
    return [train_data, valid_data, train_list, test_list, train_label, test_label]   

class CovidDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((260, 260)),
            transforms.RandomRotation(6),
            #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomCrop((224, 224)),  # Randomly crop the image to size (224, 224)
            #transforms.Resize((224, 224))
        ])
        self.labels = labels

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
            
    def __getitem__(self, idx):
        
        #img_3d = nib.load(self.file_list[idx]).get_fdata()
        img_3d_os = np.load(self.file_list[idx][0].replace('.nii.gz','.npy').replace('/code/images-with-labels/', '/code/images_npy/'), mmap_mode='r')
        img_3d_od = np.load(self.file_list[idx][0].replace('.nii.gz','.npy').replace('/code/images-with-labels/', '/code/images_npy/'), mmap_mode='r')
        # randomly select a 2D slice
        # z = np.random.randint(0,img_3d_os.shape[2]-1)
        # select 4 2D slice
        z_max = img_3d_os.shape[2]-1
        
        # os od水平拼接，然后在垂直上选取切片构成通道
        # z = [5,int(z_max/4), int(z_max/2), z_max-5]
        # imgs = [np.expand_dims(np.concatenate((img_3d_os[:, :, i],img_3d_od[:, :, i]),axis=0), axis=2) for i in z]
        # imgs = np.concatenate(imgs, axis=2)
        imgs = np.concatenate((img_3d_os[:, :, 12, None],img_3d_od[:, :, 12, None],img_3d_os[:, :, 30, None]), axis=2)

        #print('2d-img-shape:',imgs.shape)
        img_transformed = self.transform(imgs)
        #print('2d-img-shape=',img_transformed.size(),type(img_transformed))

        # class to one-hot
        label = torch.zeros(15)
        label[self.labels[idx]-85] = 1
        # return img_transformed.to(device), torch.from_numpy(np.expand_dims(self.labels[idx]-85, axis=0)).to(device)[0]
        return img_transformed.to(device), label.to(device)
class octa4Dataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((260, 260)),
            transforms.RandomRotation(30),
            #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomCrop((224, 224)),  # Randomly crop the image to size (224, 224)
            #transforms.Resize((224, 224))
        ])
        self.labels = labels

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
            
    def __getitem__(self, idx):
        
        #img_3d = nib.load(self.file_list[idx]).get_fdata()
        img_3d_os = np.load(self.file_list[idx][0].replace('.nii.gz','.npy').replace('/code/images-with-labels/', '/code/images_npy/'), mmap_mode='r')
        img_3d_od = np.load(self.file_list[idx][0].replace('.nii.gz','.npy').replace('/code/images-with-labels/', '/code/images_npy/'), mmap_mode='r')
        # randomly select a 2D slice
        z = np.random.randint(0,img_3d_os.shape[2]-10)
        # select 4 2D slice
        z_max = img_3d_os.shape[2]-1
        
        # os od水平拼接，然后在垂直上选取切片构成通道
        # z = [5,int(z_max/4), int(z_max/2), z_max-5]
        # imgs = [np.expand_dims(np.concatenate((img_3d_os[:, :, i],img_3d_od[:, :, i]),axis=0), axis=2) for i in z]
        # imgs = np.concatenate(imgs, axis=2)
        imgs = np.concatenate((img_3d_os[:, :, 15, None],img_3d_od[:, :, 15, None],img_3d_os[:, :, 20, None]), axis=2)

        #print('2d-img-shape:',imgs.shape)
        img_transformed = self.transform(imgs)
        #print('2d-img-shape=',img_transformed.size(),type(img_transformed))

        # class to one-hot 
        label = torch.zeros(5) # 前四位是类别，最后一位是regression 
        label[self.labels[idx][0]] = 1
        label[4] = self.labels[idx][1]
        # return img_transformed.to(device), torch.from_numpy(np.expand_dims(self.labels[idx]-85, axis=0)).to(device)[0]
        return img_transformed.to(device), label.to(device)
class octa4Dataset3d(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            #transforms.RandomRotation(20),
            #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            #transforms.RandomCrop((224, 224)),  # Randomly crop the image to size (224, 224)
            #transforms.Resize((224, 224))
        ])
        self.labels = labels

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
            
    def __getitem__(self, idx):
        
        #img_3d = nib.load(self.file_list[idx]).get_fdata()
        img_3d_os = np.load(self.file_list[idx][0].replace('.nii.gz','.npy').replace('/code/images-with-labels/', '/code/images_npy/'), mmap_mode='r')
        img_3d_od = np.load(self.file_list[idx][0].replace('.nii.gz','.npy').replace('/code/images-with-labels/', '/code/images_npy/'), mmap_mode='r')
        
        # 水平拼接，然后在垂直上选取切片构成通道
        z_max = img_3d_os.shape[2]-17
        z = np.random.randint(0,z_max)
        z = range(z,z+16)
        imgs_os, imgs_od = [img_3d_os[:, :, i, None] for i in z],[img_3d_od[:, :, i, None] for i in z]
        imgs = [np.concatenate((imgs_os[i],imgs_od[i]), axis=1) for i in range(16)]
        imgs = [self.transform(i) for i in imgs]
        imgs = torch.stack(imgs, dim = -1)

        # class to one-hot 
        label = torch.zeros(5) # 前四位是类别，最后一位是regression 
        label[self.labels[idx][0]] = 1
        label[4] = self.labels[idx][1]
        # return img_transformed.to(device), torch.from_numpy(np.expand_dims(self.labels[idx]-85, axis=0)).to(device)[0]
        return imgs.to(device), label.to(device) 
       
class oct_kaggleDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            #transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
            #transforms.RandomCrop((224, 224)),  # Randomly crop the image to size (224, 224)
        ])
        self.labels = labels

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        img_transformed = self.transform(img)
        cls = self.labels[idx]
        # cls to one-hot
        label = torch.zeros(4)
        label[cls] = 1
        # 复制成3通道
        img_transformed = torch.cat([img_transformed, img_transformed, img_transformed], dim=0)
        return img_transformed.to(device), label.to(device)
    

    
def get_model_oct_withpretrain(pretrain_out = 2,outsize=15, path='/code/chen/pretrain/net.pt', dropout=0.25):
    '''
    加载COVID-ViT的权重并且修改最后一层, outsize: 输出的类别数
    '''
    model = ViT(
        dim=1024,
        image_size=224,
        patch_size=32,
        num_classes=2,
        depth=12,
        heads=8,
        mlp_dim=1024,
        # transformer=efficient_transformer,
        channels=3,
        emb_dropout=dropout,
    ).to(device)
    model.mlp_head = nn.Linear(1024,pretrain_out)
    wts = torch.load(path)
    # FIXME: 不能完整得加载weights, 暂时先strict false
    model.load_state_dict(wts, strict=False)
    model.mlp_head = nn.Linear(1024,outsize)
    return model
def get_model_octa_resume(outsize=15, path=False, dropout=0.2):
    '''
    从上一次运行
    '''
    model = ViT(
        dim=1024,
        image_size=224,
        patch_size=32,
        num_classes=2,
        depth=12,
        heads=8,
        mlp_dim=1024,
        # transformer=efficient_transformer,
        channels=3,
        emb_dropout=dropout,
    ).to(device)
    model.mlp_head = nn.Linear(1024,outsize)
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
    elif path:
        model.load_state_dict(torch.load(path+'xg_vit_model_covid_2d.pt'))

    return model
class conv_head(nn.Module):
    def __init__(self, in_channels, out_channels=15):
        super(conv_head, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Linear(1024, self.out_channels),
            nn.Flatten()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add an extra dimension for the channels
        x = self.conv(x)
        return x
def get_3dvit_from(path = '',  outsize=5):
    efficient_transformer = Linformer(
        dim=1024,
        seq_len=512+1,  # 8x8x8+1 for 3D# 7x7 patches + 1 cls-token
        depth=6,
        heads=8,
        k=256   # was 64
    )
    model = ViT3(
        dim=1024,
        image_size=224,
        patch_size=8, 
        num_classes=2,
        depth=6,
        heads=8,
        mlp_dim=2048,
        transformer=efficient_transformer,
        channels=1
    ).to(device)
    # load pre-trained model
    # pretrained_net = torch.load(path)
    # model.load_state_dict(pretrained_net)
    model.mlp_head = nn.Linear(1024, outsize)
    return model

def get_model_conv(pretrain_out=2,outsize=16, path='/code/chen/pretrain/net.pt', dropout=0.25):
    '''
    加载权重并且修改最后一层, outsize: 输出的类别数
    '''
    model = ViT(
        dim=1024,
        image_size=224,
        patch_size=32,
        num_classes=2,
        depth=12,
        heads=8,
        mlp_dim=1024,
        # transformer=efficient_transformer,
        channels=3,
        dropout=0.25,
    ).to(device)
    model.mlp_head = nn.Linear(1024,pretrain_out)
    wts = torch.load(path)
    # FIXME: 不能完整得加载weights, 暂时先strict false
    model.load_state_dict(wts, strict=False)
    model.mlp_head = conv_head(1024,outsize)
    return model

def train_epoch(epochs, train_loader, model, criterion, optimizer, scheduler, eval_loader=None, save_path='temp/', **kwargs):
    # 打印dataset的长度
    print(f"Train dataset length: {len(train_loader.dataset)}, Val dataset length: {len(eval_loader.dataset)}")
    wandb.init(entity=kwargs['wandb'][0], project=kwargs['wandb'][1], name='vit-lr4e-4-bce')
    # wandb.init(project="temp")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    savings = queue.Queue()
    for i in range(3):
        savings.put(0)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch_cnt = 0
        model.to(device)
        for data, label in tqdm(train_loader):
            data = data.to(torch.float32)
            label = label
            output = model(data)
            # 每个batch进行一次pos_weight的balance
            pos_weight = get_pos_weight(label, 4)
            pos_weight = torch.ones(4).to(device)
            loss = criterion(output, label, pos_weight=pos_weight)
            if kwargs['is_MIX']: 
                output = output[:, :4]
                label = label[:, :4]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output, label = torch.argmax(output, dim=1) , torch.argmax(label, dim=1)
            metrics = metric(output.tolist(), label.tolist())
            wandb.log({'epoch': epoch, 
                       'train_loss': loss, 
                       'lr': optimizer.param_groups[0]['lr'],
                       #'train_mse': nn.MSELoss()(torch.argmax(output, dim=1).to(torch.float32), label),
                       'train_acc': (output == label).sum()/len(label),
                       })
            epoch_loss += loss / len(train_loader)
            # validating between every n batch
            if(batch_cnt% 6 == 0):
                eval(eval_loader, model, criterion, save_path=save_path, is_MIX=kwargs['is_MIX'], savings=savings)
            # TODO save the checkpoint every batch 
            torch.save(model.state_dict(), save_path + 'xg_vit_model_covid_2d.pt')
            batch_cnt += 1
        scheduler.step()
        eval(eval_loader, model, criterion, save_path=save_path, is_MIX=kwargs['is_MIX'], savings=savings)
    wandb.finish()

def eval(eval_loader, model, criterion, save_path, is_MIX, **kwargs):
    global best
    model.to(device)
    with torch.no_grad():
        epoch_loss = 0
        sum_acc = 0
        avg_sensitivity, avg_specificity = 0, 0
        for data, label in eval_loader:
            data = data.to(device).to(torch.float32)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            with open(os.path.join(save_path, 'out.txt'), 'a') as f:
                print(f'Output: {output[:,4].tolist()} , Label: {label[:,4].tolist()} \n', file=f) # 接着写入
            if is_MIX: 
                reg, reg_label = output[:,4].to(device).to(torch.float32) , label[:,4].to(device).to(torch.float32)
                output = output[:, :4]
                label = label[:, :4]
            epoch_loss += loss / len(eval_loader)
            output, label = torch.argmax(output, dim=1) , torch.argmax(label, dim=1)
            with open(os.path.join(save_path, 'out.txt'), 'a') as f:
                print(f"Result: {output.tolist()} , Label: {label.tolist()} \n", file=f)
            metrics = metric(output.tolist(), label.tolist())
            sensitivity, specificity = metrics[0].tolist(), metrics[1].tolist()
            avg_sensitivity += sensitivity/len(eval_loader)
            avg_specificity += specificity/len(eval_loader)
            sum_acc += (output == label).sum()/len(label) 
        val_acc = sum_acc / len(eval_loader)
        
        try:
            wandb.log({'val_loss': epoch_loss, 
                    #'Val_mse': nn.MSELoss()(reg, reg_label), 
                    'val_accuracy': val_acc,
                    'avg_sensitivity': avg_sensitivity,
                    'avg_specificity': avg_specificity,
                    })
        except:
            pass
        if(val_acc > best):
            best = val_acc
            torch.save(model.state_dict(), save_path + f'valacc{val_acc:.4f}_tpr_{metrics[0].tolist():.4f}.pt')
            kwargs['savings'].put(save_path + f'valacc{val_acc:.4f}_tpr_{metrics[0].tolist():.4f}.pt')
            try:
                os.remove(kwargs['savings'].get()) #不要报错
            except:
                pass
        elif(val_acc == best and avg_sensitivity > best_tp):
            torch.save(model.state_dict(), save_path + f'valacc{val_acc:.4f}_tpr_{metrics[0].tolist():.4f}.pt')
            kwargs['savings'].put(save_path + f'valacc{val_acc:.4f}_tpr_{metrics[0].tolist():.4f}.pt')
            try:
                os.remove(kwargs['savings'].get()) #不要报错
            except:
                pass
        return val_acc
def test(eval_loader, model, criterion, save_path,is_MIX, **kwargs):
    model.to(device)
    with torch.no_grad():
        epoch_loss = 0
        sum_acc = 0
        for data, label in eval_loader:
            data = data.to(device).to(torch.float32)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            epoch_loss += loss / len(eval_loader)
            if is_MIX: 
                output = output[:, :15]
            # print(f"Result: {(torch.argmax(output, dim=1) == torch.argmax(label, dim=1)).tolist()} , Label: {torch.argmax(label, dim=1).tolist()}")
            # outputs = [torch.argmax(output[i]).item() for i in range(eval_loader.batch_size)]
            # print(f"Output: {outputs}")
            sum_acc += (torch.argmax(output, dim=1) == torch.argmax(label, dim=1)).sum()/len(label) 
        val_acc = sum_acc / len(eval_loader)
        return val_acc

def predict(model):
    pass
def mixloss(bce_weight=0.5):
    def loss(output, label, pos_weight=torch.tensor([0.25,0.25,0.25,0.25])):
        # 取output的前15()个神经元 output shape(batch, 16)
        logits = output[:, :4].to(device)
        reg = output[:,4].to(device)
        logits_label = label[:, :4].to(device)
        reg_label = label[:,4].to(device).to(torch.float32)
        # 归一化两个loss的权重
        return (bce_weight)*nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))(logits, logits_label) + (1-bce_weight)*nn.MSELoss()(reg, reg_label)
    return loss
def get_pos_weight(label, num_classes=4):
    ''' 用于计算pos_weight '''
    # 计算pos_weight ,label = batch [logits, reg_value]
    label = torch.argmax(label[:, : num_classes], dim=1)
    label = label.tolist()
    counts = [label.count(i) for i in range(num_classes)]
    counts = [i if i != 0 else 1 for i in counts] # 防止除0
    # 计算每个类别的权重
    weights = torch.tensor([1.0 / count for count in counts])
    return weights


device = torch.device("cuda:2")
best = 0.5
best_tp = 0.1


if __name__ == "__main__":
    # test above
    # 1
    # print([i.__len__() for i in get_dataOCT('/code/oct_kaggle/OCT2017/train/')])

    # 2
    # print([i.__len__() for i in get_dataOCTA()])

    # 3
    # get_model_oct_withpretrain()

    # 4
    # train_loader = DataLoader(get_dataOCTA()[0], batch_size=256, shuffle=True)
    # val_loader = DataLoader(get_dataOCTA()[1], batch_size=256, shuffle=True)
    # model = get_model_oct_withpretrain(15)
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    # train_epoch(1000, train_loader, model, criterion, optimizer, scheduler, val_loader, '/code/covid_ckpts/temp/')


    # 5
    # loss = mixloss(1)
    # print(loss(torch.randn(10, 16), torch.randn(10, 16)))

    #6 
    # logits = torch.randn(2, 4)
    # pos = get_pos_weight(logits, 4)
    # print(logits, pos)
    
    # 7
    # model = get_3dvit_from(path = '/code/chen/pretrain/net.pt',  outsize=5).to(device)
    # data = get_3d16()[0][10][0].unsqueeze(0).to(torch.float32)
    # print(data.shape) # 1 1 224 224 16
    # print(model(data))

    # 8
    print(classbalance([1,2,3,4,5],[91,92,80,93,94]))