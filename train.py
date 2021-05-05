import os
import random
import time
import json
import pickle
import warnings 
import time
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Adam
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from madgrad import MADGRAD
# from torch.cuda.amp import autocast, GradScaler
# 사용하려면 pytorch upgrade 해야 됨 (현재 1.4.0) 
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch 

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import CustomDataLoader, AugmixDataLoader
# from src.models import FCN8s
from src.utils import seed_everything, make_dir, save_model, label_accuracy_score_batch, label_accuracy_score, add_hist, CosineAnnealingWarmupRestarts, get_lr, str2bool
from src.losses import DiceCELoss
from src.models import DeepLabv3Plus, DeepLabV3, UnetPlusPlus, PSPNet
from src.transforms import *

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

pos_weight = [
    0.3040,
    0.9994,
    0.9778,
    0.9097,
    0.9930,
    0.9911,
    0.9924,
    0.9713,
    0.9851,
    0.8821,
    0.9995,
    0.9947
]    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pos_weight = torch.tensor(pos_weight).float().to(device)

def train(args):
    wandb.login()
    config = dict(
        learning_rate = args.max_lr,
        architecture = args.model_name,
        seed = args.seed,
        batch_size = args.batch_size,
    )
    # project : project name
    # entity : entity name, same as wandb page (maybe user name?)
    # name : in project, instance name
    wandb.init(project="PSPNet", entity='ug_kim', name=args.wandb_name, save_code=True, config=config)

    print(args)

    def load_model(model_path, device):
        # best model 불러오기
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

        # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
        # model.eval()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

    # train.json / validation.json / test.json 디렉토리 설정
    
    dataset_path = '../input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    
    # set_image_size(args.image_size)

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    if args.transform == None:
        train_transform = A.Compose([
                                # Resize(512, 512),
                                # Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
                                A.Normalize(),
                                A.Resize(256, 256),
                                ToTensorV2()
                                ])

        val_transform = A.Compose([
                                # Resize(512, 512),
                                # Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
                                A.Normalize(),
                                A.Resize(256, 256),
                                ToTensorV2()
                                ])

    elif args.transform == "base":
        train_transform = A.Compose([
                                # Resize(512, 512),
                                # Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
                                A.Resize(256, 256),
                                A.Normalize(),
                                ToTensorV2()
                                ])

        val_transform = A.Compose([
                                # Resize(512, 512),
                                # Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
                                A.Resize(256, 256),
                                A.Normalize(),
                                ToTensorV2()
                                ])

    elif args.transform == "elastic":
        train_transform = elastic_trfm
        val_transform = trfm

    elif args.transform == "grid_distort":
        train_transform = grid_distort_trfm
        val_transform = trfm

    elif args.transform == "random_grid_shuffle":
        train_transform = random_grid_shuffle_trfm
        val_transform = trfm

    elif args.transform == "clahe":
        train_transform = clahe_trfm
        val_transform = trfm

    elif args.transform == "random_resize":
        train_transform = random_resize_trfm
        val_transform = trfm

    elif args.transform == "rotate":
        train_transform = rotate_trfm
        val_transform = trfm
    
    elif args.transform == "cutout":
        train_transform = cutout_trfm
        val_transform = trfm
    
    elif args.transform == "train":
        train_transform = mix_trfm
        val_transform = trfm

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])



    # sungbae trasnform
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    augmix_data=np.load('/opt/ml/my_code/aug.npy')

    # create own Dataset 2
    # train dataset
    # train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    # val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

    train_dataset = AugmixDataLoader(data_dir=train_path, mode="train", transform=train_transform, augmix=augmix_data.item(), augmix_prob=0.3)
    val_dataset = AugmixDataLoader(data_dir=val_path, mode='val', transform=val_transform, augmix=augmix_data.item(), augmix_prob=0.3)


    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            drop_last=True)
    
    model = None
    if args.model_name == "deeplabv3":
        model = DeepLabv3Plus()
    elif args.model_name == "deeplabv3+":
        model = DeepLabV3()
    elif args.model_name == "unet++":
        model = UnetPlusPlus()
    elif args.model_name == "pspnet":
        model = PSPNet()

    if args.is_load:           
        load_model_path = os.path.join(args.load_path, args.model_name + ".pt")
        load_model(load_model_path, device)
        print("finish load model !!!")

    model = model.to(device)
    criterion = DiceCELoss()
    # criterion = smp.losses.DiceLoss('multiclass')
    wandb.watch(model)

    optimizer = MADGRAD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 이 코드 추가하면 scheduler.step() 안해도 성능이 바뀐다
    first_cycle_steps = len(train_loader) * args.epochs // args.scheduler_step
    
    scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps=first_cycle_steps, 
            cycle_mult=1.0, 
            max_lr=args.max_lr, 
            min_lr=args.min_lr, 
            warmup_steps=int(first_cycle_steps * 0.25), 
            gamma=0.5
        )


    # log_dir = make_dir(args.log_dir)
    # logger = SummaryWriter(log_dir=log_dir)
    
    saved_dir = make_dir(args.saved_dir)

    # scaler = GradScaler()
    best_epoch = 0
    best_iou = 0.0
    best_loss = float('inf')

    print('Start training...')
    for epoch in tqdm(range(args.epochs)):
        train_iou = 0.0
        train_loss = 0.0
        val_iou = 0.0
        val_loss = 0.0

        lr_list = []

        model.train()
        for step, (images, masks, _) in enumerate(tqdm(train_loader)):
            images = torch.stack(images).to(device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)

            optimizer.zero_grad()
            # inference
            # with autocast():
                # outputs = model(images)
                # loss = criterion(outputs, masks)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            # lr_list.append(scheduler.get_lr()[0])

            wandb.log({
                "Learning rate" : get_lr(optimizer)
            })
            # lr_list.append(get_lr(optimizer))
            # save_pickle(lr_list, "/opt/ml/my_code/lr.pickle")

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            mIoU = label_accuracy_score_batch(masks.detach().cpu().numpy(), outputs, n_class=12)[2]

            train_loss += loss
            train_iou += mIoU
            
            # step 주기에 따른 loss 출력
            if (step + 1) % args.log_step == 0:
                print(f'\nEpoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {train_loss / args.log_step:.4f}, mIoU: {train_iou / args.log_step:.4f}')
                
                # tensorboard
                # logger.add_scalar("Train/loss", train_loss / args.log_step, epoch * len(train_loader) + step)
                # logger.add_scalar("Train/mIoU", train_iou / args.log_step, epoch * len(train_loader) + step)
                
                # wandb
                wandb.log({
                    "Train loss" : train_loss / args.log_step,
                    "Train mIoU" : train_iou / args.log_step
                })

                train_loss = 0.0
                train_iou = 0.0
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        print("\n Start validation step!")
        model.eval()
        hist = np.zeros((12, 12))
        with torch.no_grad():
            for step, (images, masks, _) in enumerate(tqdm(val_loader)):
                images = torch.stack(images).to(device)       # (batch, channel, height, width)
                masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

                hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)

        val_iou = label_accuracy_score(hist)[2]
            
        print(f"[Val] epoch {epoch + 1} | val_loss {val_loss / (step + 1):.4f} | val_iou {val_iou:.4f}")

        # tensorboard
        # logger.add_scalar("Val/loss", val_loss / (step + 1), epoch)
        # logger.add_scalar("Val/mIoU", val_iou, epoch)

        # wandb
        wandb.log({
            "Val loss" : val_loss / (step + 1),
            "Val mIoU" : val_iou
        })
            
        if val_iou >= best_iou:
            best_iou = val_iou
            best_epoch = epoch
            print(f"Best performance at epoch: {epoch + 1}, mIoU: {best_iou:.4f}")
            save_model(model, saved_dir, args.model_name + "_iou.pt")
            print("saved iou model!")

        if val_loss <= best_loss:
            best_loss = val_loss       
            print(f"Best performance at epoch: {epoch + 1}, loss: {best_loss:.4f}")
            save_model(model, saved_dir, args.model_name + "_loss.pt")
            print("saved loss model!")
        
        print(f"Current best miou is: {best_iou:.4f} at Epoch {best_epoch}")

    # save
    with open('lr.pickle', 'wb') as f:
        pickle.dump(lr_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--scheduler_step", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--saved_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="pspnet")
    parser.add_argument("--load_path", type=str, default="./checkpoints/exp7")
    parser.add_argument("--is_load", type=bool, default=False)
    parser.add_argument("--transform", type=str, default="train")
    parser.add_argument("--wandb_name", type=str, default="test")

    args = parser.parse_args()

    train(args)