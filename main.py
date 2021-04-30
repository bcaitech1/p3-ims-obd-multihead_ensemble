import os
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

    seed_everything(SEED)

    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')

    train_path = 'train.json'
    val_path = 'val.json'

    train_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.ElasticTransform(),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    val_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    train_loader = get_dataloader(data_dir=train_path, mode='train', transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
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