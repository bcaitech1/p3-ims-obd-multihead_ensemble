import os
import random
import time
import json
import pickle
import warnings 
import time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Adam
import cv2
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
# from torch.cuda.amp import autocast, GradScaler
# 사용하려면 pytorch upgrade 해야 됨 (현재 1.4.0) 
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch 

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import CustomDataLoader
# from src.models import FCN8s
from src.utils import seed_everything, make_dir, save_model, label_accuracy_score_batch, label_accuracy_score, add_hist, CosineAnnealingWarmupRestarts, get_lr, save_pickle
from src.losses import DiceCELoss
from src.models import DeepLabv3Plus, DeepLabV3

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())


def train(args):
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
    
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

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

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)

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

    # model = DeepLabv3Plus()
    model = DeepLabV3()

    # model = smp.DeepLabV3(
    #         encoder_name='efficientnet-b0',
    #         encoder_weights='imagenet',
    #         in_channels=3,
    #         classes=12,
    # )

    # print("#" * 10)
    # print(model.parameters)
    # print("#" * 10)


    if args.is_load:           
        load_model_path = os.path.join(args.load_path, args.file_name)
        load_model(load_model_path, device)
        print("finish load model !!!")

    model = model.to(device)
    criterion = DiceCELoss()
    # criterion = smp.losses.DiceLoss('multiclass')

    # Optimizer 정의
    optimizer = AdamW(params=model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    first_cycle_steps = len(train_loader) * args.epochs // 3
    
    scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps=first_cycle_steps, 
            cycle_mult=1.0, 
            max_lr=args.max_lr, 
            min_lr=args.min_lr, 
            warmup_steps=int(first_cycle_steps * 0.25), 
            gamma=0.5
        )

    log_dir = make_dir(args.log_dir)
    logger = SummaryWriter(log_dir=log_dir)
    saved_dir = make_dir(args.saved_dir)

    # scaler = GradScaler()

    print('Start training...')
    for epoch in tqdm(range(args.epochs)):
        train_iou = 0.0
        train_loss = 0.0
        val_iou = 0.0
        val_loss = 0.0
        best_loss = float('inf')
        best_iou = 0.0

        lr_list = []

        model.train()
        for step, (images, masks, _) in enumerate(tqdm(train_loader)):
            images = torch.stack(images).to(device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)

            optimizer.zero_grad()
            # inference
            # with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # lr_list.append(scheduler.get_lr()[0])
            lr_list.append(get_lr(optimizer))
            save_pickle(lr_list, "/opt/ml/my_code/lr.pickle")

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            mIoU = label_accuracy_score_batch(masks.detach().cpu().numpy(), outputs, n_class=12)[2]

            train_loss += loss
            train_iou += mIoU
            
            # step 주기에 따른 loss 출력
            if (step + 1) % args.log_step == 0:
                print(f'Epoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {train_loss / args.log_step:.4f}, mIoU: {train_iou / args.log_step:.4f}')
                logger.add_scalar("Train/loss", train_loss / args.log_step, epoch * len(train_loader) + step)
                logger.add_scalar("Train/mIoU", train_iou / args.log_step, epoch * len(train_loader) + step)
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
        logger.add_scalar("Val/loss", val_loss / (step + 1), epoch)
        logger.add_scalar("Val/mIoU", val_iou, epoch)
        
        if val_iou >= best_iou:
            best_iou = val_iou
            print(f"Best performance at epoch: {epoch + 1}, mIoU: {best_iou:.4f}")
            save_model(model, saved_dir, args.file_name)

    # save
    with open('lr.pickle', 'wb') as f:
        pickle.dump(lr_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--log_dir", type=str, default="./logs")
    
    parser.add_argument("--saved_dir", type=str, default="./checkpoints")
    parser.add_argument("--file_name", type=str, default="deeplabv3.pt")
    parser.add_argument("--load_path", type=str, default="./checkpoints/exp7")
    parser.add_argument("--is_load", type=bool, default=False)

    args = parser.parse_args()

    train(args)