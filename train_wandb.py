import os
import random
import time
import json
import wandb
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torch.nn import functional as F
from pycocotools.coco import COCO
import cv2
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

from utils import *
from dataloader import *
from scheduler import *
from evaluate import *

def train(config=None):
    
    wandb.init(config=config)
    config = wandb.config
    device = 'cuda'
    
    global args
        ### Hyper parameters ###
    SEED = config.SEED
    BATCH_SIZE = config.BATCH_SIZE
    EPOCHS = config.EPOCHS
    LR = config.LR
    Eta = config.Eta_Max
    OPTIMIZER = config.Optimizer
    img_size = config.IMG_SIZE
    save_model_name = f'{args.project_name}_{img_size}_seed{SEED}_batch{BATCH_SIZE}_LR{LR}_Eta{Eta}'
    accumulation_step = config.ACCUM

    break_mIoU = 0.3
    best_val_mIoU = 0.35
    
        ### SEED setting ###
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

        ### Dataset ###
    dataset_path = 'input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    
    train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.ColorJitter(),
            A.RandomRotate90(),
            A.HorizontalFlip (0.5),
                            A.Normalize (mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],max_pixel_value=1.0),
        ToTensorV2()])
        
    val_transform = A.Compose([
                        A.Resize(256, 256) ,
                            A.Normalize (mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],max_pixel_value=1.0),
                        ToTensorV2()])
    
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
     ### Model ###    
    model = get_model(args)
    model = model.to(device)
    
    wandb.watch(model)
    
        ### Train ###
    criterion = nn.CrossEntropyLoss()
    
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    elif OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=LR)
    elif OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),lr=LR)
    elif OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=LR)

    flag=True
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=EPOCHS, eta_max=Eta,  T_up=100, gamma=0.5)

    # fake metas
    img_metas =[[{
        'img_shape': (img_size, img_size, 3),
        'ori_shape': (img_size, img_size, 3),
        'pad_shape': (img_size, img_size, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
                'flip_direction': 'horizontal'
            }]]
    print("Start training..")

    for epoch in range(EPOCHS):
        epoch+=1
        avg_loss = 0
        batch_count = len(train_loader)
        model.train()

        for step, (images, masks) in enumerate(train_loader):
            start = time.time()
            images, masks = images.to(device), masks.long().to(device)

            imgs = [images]
            output = model(imgs,img_metas,return_loss=False)

            loss = criterion(output, masks)
            loss.backward()

            if (step+1)%accumulation_step==0:
                optimizer.step()
                optimizer.zero_grad()
    
            avg_loss += loss.item() / batch_count
            lr = scheduler.get_lr()[0]
            print(f"\rEpoch:{epoch:3d}  step:{step:3d}/{batch_count-1}  time:{time.time() - start:.3f}  LR:{lr:.6f}", end='')
            break
        scheduler.step()
        
        val_loss, val_mIoU, val_mIoU2, val_mIoU3 = validation_swin(model, val_loader, criterion, device,img_metas)     
        
        wandb.log({"loss": avg_loss, "val_loss": val_loss, "val_mIoU": val_mIoU, "val_mIoU2": val_mIoU2, "val_mIoU3": val_mIoU3})
        print(f"\n  loss: {avg_loss:.3f}  val_loss: {val_loss:.3f}  val_mIoU:{val_mIoU:.3f}  val_mIoU2:{val_mIoU2:.3f}  val_mIoU3:{val_mIoU3:.3f}")
        file_name=save_model_name + f'_epoch{epoch}_score1{val_mIoU:.3f}_score2{val_mIoU2:.3f}_score3{val_mIoU3:.3f}.pt'
        now_mIoU = (val_mIoU+val_mIoU2+val_mIoU3) /3  
        
        # if best_val_mIoU < now_mIoU:
        if val_mIoU3>=0.5 and val_mIoU>=0.4:
            save_model(model, saved_dir="weight", file_name=file_name)
            best_val_mIoU = now_mIoU
            flag=False
        elif epoch >= EPOCHS//3 and flag and now_mIoU < break_mIoU:
            break
    print("Finish training")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train with wandb')    
    parser.add_argument('--project_name', type=str,default='swin_b')
    # optimizer
    parser.add_argument('--network', type=str,
                        default='swin_b',
                        const='swin_b',
                        nargs='?',
                        choices=['swin_s','swin_b','swin_t'],
                        help='swin_s, swin_b (base), swin_t (tiny)')
    parser.add_argument('--accum',type=int, default=4)
    parser.add_argument('--count',type=int,default=1)
    parser.add_argument('--loss',type=int,default=3)
    parser.add_argument('--repo_root',type=str,default='')
    args = parser.parse_args()

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_mIoU3',
            'goal': 'minimize'   
            }
    }

    parameters_dict = {
        'SEED':{
            'values' : [2536]
        },
        'BATCH_SIZE': {
            'values': [8]
        },
        'ACCUM':{
            'values': [args.accum] 
        },
        'LR': {
            'values': [5e-5]
        },
        'Eta_Max':{
            'values': [1e-7]
        },
        'EPOCHS':{
            'values': [40]
        },
        'Optimizer':{
            'value': 'adamw'
        },
        'IMG_SIZE':{
            'values' : [256]
        },
         'project_name':{
            'value': args.project_name
        },
        'Loss':{
            'values':[args.loss]
        }
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    wandb.agent(sweep_id, train, count=args.count)