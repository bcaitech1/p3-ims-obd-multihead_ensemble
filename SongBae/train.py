import os
import wandb
import logging
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import train_valid
from src.utils import save_model
from src.dataset import SegmentationDataset
from src.losses import *
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import numpy as np 
import wandb
from  swin_transformer_pytorch import SwinTransformer
import warnings
from glob import glob
warnings.filterwarnings('ignore')
def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--trn_ratio', default=0.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--decay', default=1e-6, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--version', default='v1', type=str)
    parser.add_argument('--model_type', default='fcn8s', type=str)
    parser.add_argument('--use_augmix', default=1, type=int)
    parser.add_argument('--use_only_test', default=0, type=int)
    args = parser.parse_args()
    print(args)
    # for reproducibility
    seed_everything(args)
    # define key paths
    main_path = '.'
    data_path = os.path.join(main_path, 'input', 'data')
    # define transform
    trn_tfms = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.OneOf([
          A.CLAHE(p=1.0),
          A.RandomBrightness(p=1.0)],p=0.5),
      A.Normalize(),
        ToTensorV2()
    ])
    val_tfms = A.Compose([
      A.Normalize(),
        ToTensorV2()
    ])
    
    augmix_data = np.load('augmix_all.npy',allow_pickle=True)
    # define train & valid dataset
    image_list= glob('./input/data/batch_03/0*')
    if args.trn_ratio:
        total_annot = os.path.join(data_path, 'train_all.json')
        total_cat = make_cat_df(total_annot, debug=True)
        total_ds = SegmentationDataset(
            data_dir=total_annot, cat_df=total_cat, mode='train', transform=None)
        trn_size = int(len(total_ds)*0.8)
        val_size = int(len(total_ds) - trn_size)
        trn_ds, val_ds = torch.utils.data.random_split(
            total_ds, [trn_size, val_size])
    else:
        psudo_label='./presudo_result.csv'
        trn_annot = os.path.join(data_path, 'train.json')
        val_annot = os.path.join(data_path, 'val.json')
        trn_cat = make_cat_df(trn_annot, debug=True)
        val_cat = make_cat_df(val_annot, debug=True)
        trn_ds = SegmentationDataset(
            data_dir=trn_annot, cat_df=trn_cat, mode='train', psudo_label=psudo_label, transform=trn_tfms, augmix=augmix_data.item(), image_list=image_list,args=args)
        val_ds = SegmentationDataset(
            data_dir=val_annot, cat_df=val_cat, mode='val', transform=val_tfms,)
    # define dataloader
    trn_dl = DataLoader(dataset=trn_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4,
                        drop_last=True)
    val_dl = DataLoader(dataset=val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        drop_last=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if args.model_type == 'fcn8s':
        backbone = vgg16(pretrained=True)
        model = FCN8s(backbone=backbone)
    elif args.model_type == "unet":
        model = smp.Unet(
            # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=3,
            # model output channels (number of classes in your dataset)
            classes=12,
        )
    elif args.model_type == 'deeplabv3':
      model = smp.DeepLabV3(
          encoder_name='efficientnet-b3',
          encoder_weights='imagenet',
          in_channels=3,
          classes=12
      )
    elif args.model_type=='hrnet_ocr':
      import yaml 
      from src.models import get_seg_model
      config_path='./src/configs/hrnet_seg.yaml'
      with open(config_path) as f:
        cfg=yaml.load(f)
      model=get_seg_model(cfg)

    model = model.to(device)
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=args.lr, weight_decay=args.decay)
    #criterion = DiceCELoss()
    criterion=DiceCELoss()
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2)
    logger = logging.getLogger("Segmentation")
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(
        os.path.join(logger_dir, f'{args.version}.log'))
    logger.addHandler(file_handler)
    train_valid(args.epochs, model, trn_dl, val_dl, criterion, optimizer, logger, device,scheduler,args,augmix_data,args.use_augmix)


if __name__ == "__main__":
    main()
