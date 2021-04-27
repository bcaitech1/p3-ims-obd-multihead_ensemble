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

from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import train_valid
from src.utils import save_model
from src.dataset import SegmentationDataset
from src.models import FCN8s
from src.losses import *


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
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20),

        # A.OneOf([
        #     A.RGBShift(p=1.0),
        #     A.HueSaturationValue(p=1.0),
        #     A.ChannelShuffle(p=1.0),
        # ], p=.5),

        A.Normalize(),
        ToTensorV2()
    ])

    val_tfms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])


    # define train & valid dataset
    if args.trn_ratio:
        total_annot = os.path.join(data_path, 'train_all.json')
        total_cat = make_cat_df(total_annot, debug=True)
        total_ds = SegmentationDataset(data_dir=total_annot, cat_df=total_cat, mode='train', transform=None)

        trn_size = int(len(total_ds)*0.8)
        val_size = int(len(total_ds) - trn_size)
        trn_ds, val_ds = torch.utils.data.random_split(total_ds, [trn_size, val_size])
    else:
        trn_annot = os.path.join(data_path, 'train.json')
        val_annot = os.path.join(data_path, 'val.json')
        trn_cat = make_cat_df(trn_annot, debug=True)
        val_cat = make_cat_df(val_annot, debug=True)


        trn_ds = SegmentationDataset(data_dir=trn_annot, cat_df=trn_cat, mode='train', transform=trn_tfms)
        val_ds = SegmentationDataset(data_dir=val_annot, cat_df=val_cat, mode='valid', transform=val_tfms)


    # define dataloader
    trn_dl = DataLoader(dataset=trn_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=3)

    val_dl = DataLoader(dataset=val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=3)


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
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=12
        )

    model = model.to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    # criterion = nn.CrossEntropyLoss()
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
        trn_loss, trn_mIoU, val_loss, val_mIoU = train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, logger, device)

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