import os
import numpy as np

from albumentations.augmentations.transforms import HorizontalFlip, RandomRotate90, VerticalFlip
from albumentations.core.composition import OneOf
import wandb
import torch
import argparse
import pandas as pd
import albumentations
import albumentations.pytorch

from prettyprinter import cpprint

from src.train import train
from src.utils import YamlConfigManager, seed_everything, get_dataloader

def main(cfg):
    SEED = cfg.values.seed
    TRAIN_BATCH_SIZE = cfg.values.train_args.train_batch_size
    VAL_BATCH_SIZE = cfg.values.val_args.val_batch_size
    IMAGE_SIZE = cfg.values.image_size
    USE_AUGMIX = cfg.values.augmix_args.use_augmix
    AUGMIX_PROB = cfg.values.augmix_args.augmix_prob

    seed_everything(SEED)

    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')

    train_path = 'train.json'
    val_path = 'val.json'

    augmix_path = './augmix.npy'

    if USE_AUGMIX:
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        augmix = np.load(augmix_path)
        augmix = augmix.item()

    else:
        augmix_data = None

    train_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(),
        albumentations.OneOf([
            albumentations.RandomRotate90(p=0.5),
            albumentations.Rotate(limit=30, p=0.5)
        ], p=0.5),
        albumentations.OneOf([
            albumentations.CLAHE(p=1),
            albumentations.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albumentations.GridDistortion(p=1),
            albumentations.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)
        ], p=0.6),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    val_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    train_loader = get_dataloader(data_dir=train_path, mode='train', transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True, augmix=augmix, augmix_prob=AUGMIX_PROB)
    val_loader = get_dataloader(data_dir=val_path, mode='val', transform=val_transform, batch_size=VAL_BATCH_SIZE, shuffle=False)

    train(cfg, train_loader, val_loader)


if __name__ == '__main__':
    wandb.init(project="P-stage3-semantic", reinit=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/config.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()    
    wandb.run.name = args.config
    wandb.run.save()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    wandb.config.update(cfg)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)