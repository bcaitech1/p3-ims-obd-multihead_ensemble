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
from src.models import *
from src.losses import *

from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast  
# 사용하려면 pytorch upgrade 해야 됨 (현재 1.4.0) 
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch 

from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    # Hyperparameters
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--trn_ratio', default=0.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--decay', default=1e-6, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num_workers', default='3', type=int)
    parser.add_argument('--cutout', default=0 , type=float)
    parser.add_argument('--cutmix', default=0 , type=float)
    parser.add_argument('--early_stop', default=3 , type=int)
    parser.add_argument('--augmix_prob', default=0 , type=float)
    
    # Model & Optimizer & Criterion.. 
    parser.add_argument('--model_type', default='FCN8s', type=str , help = 'ex) Unet , DeepLabV3, FCN8s ,SegNet,Effi_Unet_NS')
    parser.add_argument('--criterion', default='CELoss', type=str , help = 'ex)DiceLoss , DiceBCELoss, IoULoss ,FocalLoss ... ')
    parser.add_argument('--optimizer', default='AdamW', type=str , help = 'ex) torch.optim.Adamw')
    parser.add_argument('--scheduler', default='None', type=str , help = 'ex) torch.optim.lr_scheduler.StepLR')
    parser.add_argument('--aug', default='None', type=str , help = 'ex) torch.optim.lr_scheduler.StepLR')
    parser.add_argument('--use_augmix', default=False, type=bool , help = 'ex) torch.optim.lr_scheduler.StepLR')
   
    # container environment
    parser.add_argument('--data_path', default='/content/input', type=str)
    parser.add_argument('--version', default='v1', type=str)

    args = parser.parse_args()
    print(args)

    # for reproducibility
    seed_everything(args.seed)

    mean = np.array([0.46098186, 0.44022841, 0.41892368], dtype=np.float32)
    std  = np.array([0.21072529, 0.20763867, 0.21613272], dtype=np.float32)


    ## 여기서 npy 파일 있는 path 설정해주세요!!!!!! 
    ## parser로 준 augmix_prob & use_augmix 설정해주셔야돼요
    augmix_path = '/content/drive/MyDrive/code/augmix.npy'

    if args.use_augmix:
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        augmix_data = np.load(augmix_path)
        augmix_data = augmix_data.item()

    else:
        augmix_data = None
 

    # define transform
    # if args.aug == 'flip':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.HorizontalFlip(p=1),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'rotate':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.RandomRotate90 (p=0.5),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'rotate30':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.Rotate (limit = 30, p=0.5),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'gridmask':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       GridMask(num_grid = 4),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'Cutout':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.Cutout (num_holes=4, max_h_size=20, max_w_size=20,p=0.5),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'randomresize':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=0.5),
    #       A.RandomResizedCrop(256,256),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'clahe':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.CLAHE(p=1),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'griddistortion':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.GridDistortion(p=0.5),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'optical':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'elastic':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.ElasticTransform(p=0.5, alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])

    # elif args.aug == 'has':
    #   trn_tfms = A.Compose([
    #       A.Resize (256, 256 , p=1),
    #       A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    #       A.Normalize(),
    #       ToTensorV2()
    #   ])
    # else :

    trn_tfms = A.Compose([
          A.Resize (256, 256 , p=1),
          A.RandomResizedCrop(256,256),
          A.Rotate (limit = 30, p=0.5),
          A.CLAHE(p=1),
          A.Normalize(),
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

        trn_ds = SegmentationDataset(data_dir=trn_annot, cat_df=trn_cat, mode='train', transform=trn_tfms , augmix = augmix_data , prob = args.augmix_prob)
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

    opt_module = getattr(import_module("torch.optim") , args.optimizer) #default : Adam
    optimizer = opt_module(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    if  args.scheduler == "None": 
        scheduler = None
    else :
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
    wandb.init(project='segmentation', entity='hyerin', name = args.version , save_code = True)

    config = wandb.config
    config.learning_rate = args.lr
    config.seed = args.seed
    config.batch_size = args.batch_size
    wandb.watch(model)


    best_loss = float("INF")
    best_mIoU = 0
    early_cnt = 0
    for epoch in range(args.epochs):
        
        if early_cnt >= args.early_stop : break
            
        trn_loss, trn_mIoU, val_loss, val_mIoU = train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, logger, device, scheduler, args.cutout, args.cutmix, augmix_data )
        ## 
        wandb.log({"loss": val_loss , "IoU" :val_mIoU })

        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            save_model(model, version=args.version, save_type='loss')
            

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            save_model(model, version=args.version, save_type='mIoU')
            early_cnt = 0

        else :
            early_cnt += 1
    wandb.finish()

if __name__ == "__main__":
    main()