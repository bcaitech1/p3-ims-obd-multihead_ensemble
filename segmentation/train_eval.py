import os
import wandb
import logging
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss, BCEWithLogitsLoss


from src.utils import *
from src.dataset import SegmentationDataset
from src.models import FCN8s
from src.losses import *


from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast  
# 사용하려면 pytorch upgrade 해야 됨 (현재 1.4.0) 
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch 

from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")

    # Hyperparameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--trn_ratio', default=0.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--decay', default=1e-6, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num_workers', default='3', type=int)
    parser.add_argument('--cutout', default=0 , type=float)
    parser.add_argument('--early_stop', default=3 , type=int)
    
    # Model & Optimizer & Criterion.. 
    parser.add_argument('--model_type', default='FCN8s', type=str , help = 'ex) Unet , DeepLabV3, FCN8s ,SegNet')
    parser.add_argument('--criterion', default='CELoss', type=str , help = 'ex)DiceLoss , DiceBCELoss, IoULoss ,FocalLoss ... ')
    parser.add_argument('--optimizer', default='AdamW', type=str , help = 'ex) torch.optim.Adamw')
    parser.add_argument('--scheduler', default='StepLR', type=str , help = 'ex) torch.optim.lr_scheduler.StepLR')
    
    # container environment
    parser.add_argument('--data_path', default='/opt/ml/input/data', type=str)
    parser.add_argument('--version', default='v1', type=str)


    args = parser.parse_args()
    print(args)

    # for reproducibility
    seed_everything(args.seed)

    mean = np.array([0.46098186, 0.44022841, 0.41892368], dtype=np.float32)
    std  = np.array([0.21072529, 0.20763867, 0.21613272], dtype=np.float32)

    # define transform
    trn_tfms = A.Compose([

        A.Normalize(),
        A.Resize (256, 256 , p=1),

        ToTensorV2()
    ])

    val_tfms = A.Compose([
        A.Normalize(),
        A.Resize (256 , 256 , p=1),
        ToTensorV2()
    ])


    # define train & valid dataset
    if args.trn_ratio:

        total_annot = os.path.join(args.data_path, 'train_all.json')
        total_cat = make_cat_df(total_annot, debug=True)
        total_ds = SegmentationDataset(data_dir=total_annot, cat_df=total_cat, mode='train', transform=None)

        trn_size = int(len(total_ds)* args.trn_ratio)
        val_size = int(len(total_ds) - trn_size)
        trn_ds, val_ds = torch.utils.data.random_split(total_ds, [trn_size, val_size])

    else:
        trn_annot = os.path.join(args.data_path, 'train.json')
        val_annot = os.path.join(args.data_path, 'val.json')
        trn_cat = make_cat_df(trn_annot, debug=True)
        val_cat = make_cat_df(val_annot, debug=True)

        trn_ds = SegmentationDataset(data_dir=trn_annot, cat_df=trn_cat, mode='train', transform=trn_tfms)
        val_ds = SegmentationDataset(data_dir=val_annot, cat_df=val_cat, mode='valid', transform=val_tfms)


    # define dataloader
    trn_dl = DataLoader(dataset=trn_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers)

    val_dl = DataLoader(dataset=val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_module = getattr(import_module("src.models"), args.model_type)
    model = model_module(num_classes = 12).to(device)

    opt_module = getattr(import_module("torch.optim") , args.optimizer) #default : AdamW
    optimizer = opt_module(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    sch_module = getattr(import_module("torch.optim.lr_scheduler") , args.scheduler) #default : stepLR
    scheduler = sch_module(optimizer, step_size = 10)
    
    criterion = create_criterion(args.criterion)

    logger = logging.getLogger("Segmentation")
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, f'{args.version}.log'))
    logger.addHandler(file_handler)


    wandb.login()
    
    best_loss = float("INF")
    best_mIoU = 0
    early_cnt = 0
    for epoch in range(args.epochs):
        
        if early_cnt >= args.early_stop : break
            
        trn_loss, trn_mIoU, val_loss, val_mIoU = train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, scheduler, logger, device, args.cutout)


        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            save_model(model, version=args.version, save_type='loss')
            early_cnt = 0
        else :
            early_cnt += 1

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            save_model(model, version=args.version, save_type='mIoU')
    wandb.finish()


if __name__ == "__main__":
    main()