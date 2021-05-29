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
from segmentation_models_pytorch.utils.losses import DiceLoss


from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import train_valid
from src.utils import save_model
from src.dataset import SegmentationDataset, all_Dataset
from src.models import FCN8s
from src.losses import *
from src.scheduler import *


from torch.utils.data import DataLoader
from torchvision.models import vgg16


def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--trn_ratio', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--decay', default=1e-6, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--version', default='v1', type=str)
    parser.add_argument('--model_type', default='fcn8s', type=str)


    args = parser.parse_args()
    print(args)
  
    # for reproducibility
    seed_everything(args.seed)

    # define key paths
    main_path = '.'
    data_path = os.path.join(main_path, 'input', 'data')

    mean = np.array([0.46098186, 0.44022841, 0.41892368], dtype=np.float32)
    std  = np.array([0.21072529, 0.20763867, 0.21613272], dtype=np.float32)

    # define transform
    trn_tfms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30),
        A.OneOf([
            A.ElasticTransform(p=1, alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            A.CLAHE(p=1),
        ], p=0.6),

        A.Normalize(),
        ToTensorV2()
    ])
    val_tfms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

    np_load_old=np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    augmix_data=np.load('augmix_all.npy')

    # define train & valid dataset
    if args.trn_ratio:
        total_annot = os.path.join(data_path, 'train_all.json')
        total_cat = make_cat_df(total_annot, debug=True)
        total_ds = SegmentationDataset(data_dir=total_annot, cat_df=total_cat, mode='train', transform=trn_tfms)

        trn_size = int(len(total_ds)*args.trn_ratio)
        val_size = int(len(total_ds) - trn_size)
        trn_ds, val_ds = torch.utils.data.random_split(total_ds, [trn_size, val_size])

    else:
        trn_annot = os.path.join(data_path, 'train.json')
        val_annot = os.path.join(data_path, 'val.json')
        trn_cat = make_cat_df(trn_annot, debug=True)
        val_cat = make_cat_df(val_annot, debug=True)

        pseudo_csv = "presudo_result.csv"

        trn_ds = all_Dataset(trn_annot, trn_cat, pseudo_csv, mode='train', augmix=augmix_data.item(), transform=trn_tfms, binary_mask=True)
        val_ds = SegmentationDataset(data_dir=val_annot, cat_df=val_cat, mode='valid', transform=val_tfms, binary_mask=True)


    # define dataloader
    trn_dl = DataLoader(dataset=trn_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=2)

    val_dl = DataLoader(dataset=val_ds,
                        batch_size=args.batch_size//2,
                        shuffle=False,
                        num_workers=2)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'fcn8s':
        backbone = vgg16(pretrained=True)
        model = FCN8s(backbone=backbone)
    elif args.model_type == "unet":
        model = smp.Unet(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=12,  # model output channels (number of classes in your dataset)
        )
    elif args.model_type == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name='efficientnet-b2',
            encoder_weights='imagenet',
            in_channels=3,
            classes=12
        )
    elif args.model_type == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            in_channels=3,
            classes=12
        )
    elif args.model_type == 'pannet':
        model = smp.PAN(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            in_channels=3,
            classes=12
        )


    model = model.to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    first_cycle_steps=len(trn_dl)*args.epochs//3

    ## set lr scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1,
        max_lr=args.lr,
        min_lr=1e-5,
        warmup_steps=int(first_cycle_steps*0.25),
        gamma=0.5
        )


    # 상황에 맞게 loss 바꾸어 쓰기
    criterion = DiceCELoss()

    logger = logging.getLogger("Segmentation")
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, f'{args.version}.log'))
    logger.addHandler(file_handler)



    best_loss = float("INF")
    best_mIoU = 0
    for epoch in range(args.epochs):
        trn_loss, trn_mIoU, val_loss, val_mIoU = train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, scheduler, logger, device)

        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            save_model(model, version=args.version, save_type='loss')

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            save_model(model, version=args.version, save_type='mIoU')


if __name__ == "__main__":
    main()