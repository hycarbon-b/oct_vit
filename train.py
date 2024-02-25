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

#from vit_pytorch.efficient import ViT
#from vit_pytorch.efficient import ViT
from vit_pytorch.vit import ViT
from vit_pytorch.deepvit import DeepViT
import wandb
device = torch.device("cuda:2")
wandb.init(project="my-project")
parser = argparse.ArgumentParser(description='Pytorch COVID-ViT 2D/3D Training')
#parser.add_argument('--save', default='/code/covid_ckpts/oct4class_pretrained/', type=str, help='model save path')
# parser.add_argument('--save', default='/code/covid_ckpts/oct4class_biglr/', type=str, help='model save path')
parser.add_argument('--save', default='/code/covid_ckpts/octa_test/', type=str, help='model save path')
parser.add_argument('--best', default=0, help='best accuracy')
args = parser.parse_args()
if not os.path.exists(args.save):
    os.makedirs(args.save)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight)
def mixLoss(output, label):
    logits = output[:, 0:15]
    return 7*criterion(output, label) + 3*nn.MSELoss()(torch.argmax(output, dim=1).to(torch.float32), label)
def train_epoch(epochs, train_loader, model, criterion, optimizer, scheduler, eval_loader=None):
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch_cnt = 0
        model.to(device)
        for data, label in tqdm(train_loader):
            data = data.to(torch.float32)
            label = label
            allocated = torch.cuda.memory_allocated(torch.device(device))
            output = model(data)
            print(f'Output: {output.shape}', f'Label: {label.shape}')
            # loss = 7*criterion(output, label) + 3*nn.MSELoss()(torch.argmax(output, dim=1).to(torch.float32), label) 
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({'epoch': epoch, 
                       'train_loss': loss, 
                       'train_avg_loss': loss/ len(train_loader),
                       'lr': optimizer.param_groups[0]['lr'],
                       #'train_mse': nn.MSELoss()(torch.argmax(output, dim=1).to(torch.float32), label),
                       'train_acc': (torch.argmax(output, dim=1) == torch.argmax(label, dim=1)).sum()/len(label)})

            epoch_loss += loss / len(train_loader)
            # DEBUG
            print(f"Allocated: {allocated}, Mixloss: {(loss/ len(train_loader)):.5f} ")
            batch_cnt += 1
            # validating between every n batch
            if(batch_cnt%12 == 0):
                eval(eval_loader, model, criterion)
            # TODO save the checkpoint every batch 
            torch.save(model.state_dict(), 'xg_vit_model_covid_2d.pt')
    
        print(f"Epoch {epoch+1}: | epoch-loss-avg: {epoch_loss:.5f}")
        # inference on whole train data
        wandb.log({'Val_whole_epoch': eval(train_loader, model, criterion)})
        if(epoch%100 == 0):
            torch.save(model.state_dict(), f'/code/covid_ckpts/epoch{epoch}.pt')

def eval(eval_loader, model, criterion):
    with torch.no_grad():
        epoch_loss = 0
        val_acc = 0
        batch_cnt = 0
        for data, label in eval_loader:
            data = data.to(device).to(torch.float32)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            epoch_loss += loss / len(eval_loader)
            print(f"Result: {(torch.argmax(output, dim=1) == torch.argmax(label, dim=1)).tolist()} , Label: {torch.argmax(label, dim=1).tolist()}")
            # outputs = [torch.argmax(output[i]).item() for i in range(eval_loader.batch_size)]
            # print(f"Output: {outputs}")
            val_acc += (torch.argmax(output, dim=1) == torch.argmax(label, dim=1)).sum()/len(label)
            batch_cnt += 1
        print(f"Evaluation loss: {epoch_loss:.5f}")    
        mean_batch_acc = val_acc / batch_cnt
        wandb.log({'Val_loss': epoch_loss, 
                   # 'Val_mse': nn.MSELoss()(torch.argmax(output, dim=1).to(torch.float32), label), 
                   'val_acc': mean_batch_acc})
        if(mean_batch_acc > args.best):
            args.best = mean_batch_acc
            torch.save(model.state_dict(), args.save + f'val_acc{mean_batch_acc}.pt')
        return mean_batch_acc

def predict(model):
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
    datas = CovidDataset(train_list, label_list, transform=transform)
    with torch.no_grad():
        mse = 0
        for i in range(len(train_list)):
            data = datas.__getitem__(i)[0]
            data = data.to(torch.float32).unsqueeze(0).to(device)
            label = datas.__getitem__(i)[1]
            output = model(data)
            output = torch.argmax(output[0])
            print(f"Predicted: {output}, Actual: {label}")
            mse += (output - label)**2
        print(f"MSE: {mse/len(train_list)}")

def get_dataOCT(path):
    '''
    获得oct_kaggle的datalist和labels
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
    return datalistOCT, labels


def balance(train_list, label_list):
    ''' 用于类别平衡 '''
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

class CovidDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.transform = transform
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
        imgs = np.concatenate((img_3d_os[:, :, 20, None],img_3d_od[:, :, 20, None],img_3d_os[:, :, 10, None]), axis=2)

        #print('2d-img-shape:',imgs.shape)
        img_transformed = self.transform(imgs)
        #print('2d-img-shape=',img_transformed.size(),type(img_transformed))

        # class to one-hot
        label = torch.zeros(15)
        label[self.labels[idx]-85] = 1
        # return img_transformed.to(device), torch.from_numpy(np.expand_dims(self.labels[idx]-85, axis=0)).to(device)[0]
        return img_transformed.to(device), label.to(device)
    
class oct_kaggleDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.transform = transform
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
def timing(s):
    # 从文件名中提取时间， 用于排序
    s = s.replace('-', '_').split('_')
    return s[-6] + s[-8].rjust(2, '0') + s[-7].rjust(2, '0') + s[-5].rjust(2, '0') + s[-4].rjust(2, '0') + s[-3].rjust(2, '0')


# Prepare the dataset

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((260, 260)),
        #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.RandomCrop((224, 224)),  # Randomly crop the image to size (224, 224)
    ]
)

# trian_list contain file path of the images
train_list = glob.glob('/code/images-with-labels/**/*.nii.gz', recursive=True)
# 一个patient一个文件夹
sample_folder = set(['/'.join(i.split('/')[:-1]) for i in train_list])

# 没有区分OS OD
# train_list = [os.path.join(i,sorted(os.listdir(i), key=timing)[-1]) for i in sample_folder]
# 区分OS OD
train_list = []

for i in sample_folder:
    os_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OS']
    od_list = [i for i in os.listdir(i) if i.split('_')[-2] == 'OD']
    train_list.append([os.path.join(i,sorted(os_list, key=timing)[-1]), os.path.join(i,sorted(od_list, key=timing)[-1])])
ex = pd.read_excel('/code/Sleep-results.xlsx')
# load label from train_list and excel
label_list = [(ex[ex.iloc[:,0] == i[0].split('/')[-2]])['sat_avg'].iloc[0] for i in train_list[:]]
train_list, label_list = balance(train_list, label_list)
# 划分数据集
train_list, test_list, train_label, test_label = train_test_split(train_list, label_list, 
                                                                 test_size=0.15, 
                                                                 random_state=33)
# octa dataset
train_data = CovidDataset(train_list, train_label, transform=transform)
valid_data = CovidDataset(test_list,test_label, transform=transform)
test_data = CovidDataset(test_list, label_list,transform=transform)

# oct_kaggledataset
# train_list, label_list = get_dataOCT('/code/oct_kaggle/OCT2017/train/')
# train_list, test_list, train_label, test_label = train_test_split(train_list, label_list, 
#                                                                  test_size=0.15, 
#                                                                  random_state=33)
# train_data = oct_kaggleDataset(train_list, train_label, transform=transform)
# valid_data = oct_kaggleDataset(test_list,test_label, transform=transform)
# test_data = oct_kaggleDataset(test_list, label_list,transform=transform)

# dataloader
batch_size = 320
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

# determine the model
device = torch.device(device)
efficient_transformer = Linformer(
    dim=1024,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=10,
    heads=8,
    k=64
)
# model = ViT(
#     dim=1024,
#     image_size=448,
#     patch_size=32,
#     num_classes=4,
#     depth=16,
#     heads=10,
#     # transformer=efficient_transformer,
#     mlp_dim=1024,
#     channels=1,
#     dropout=0.1,
# ).to(device)
# official pretrian
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
).to(device)
# wts = torch.load('/code/chen/pretrain/net.pt')
# model.load_state_dict(wts, strict=False)
model.mlp_head = nn.Linear(1024,4)




# initialize the weight 
# model.apply(init_weights)
# Resume from last run
# pretrained_net = torch.load('xg_vit_model_covid_2d.pt')
pretrained_net = torch.load('/code/covid_ckpts/oct4class_biglr/val_acc0.9759836196899414.pt')
model.load_state_dict(pretrained_net)
model.mlp_head = nn.Linear(1024,15)

# loss function
criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-6, momentum=0.)
# scheduler
# scheduler = StepLR(optimizer, step_size=20, gamma=0.6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

# train the model
print(f'Sample size: {len(train_data)}, Eval size: {len(valid_data)}')
train_epoch(1000, train_loader, model, criterion, optimizer, scheduler, valid_loader)
pretrained_net = torch.load('/code/xg_vit_model_covid_2d.pt')
model.load_state_dict(pretrained_net)
predict(model)
