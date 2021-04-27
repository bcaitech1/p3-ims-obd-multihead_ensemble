import os
from albumentations.augmentations.transforms import Normalize
import torch
import argparse
import pandas as pd
import albumentations
import albumentations.pytorch

from prettyprinter import cpprint

from src.train import train
from src.utils import YamlConfigManager, seed_everything, get_dataloader

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def main(cfg):
    SEED = cfg.values.seed
    TRAIN_BATCH_SIZE = cfg.values.train_args.train_batch_size
    VAL_BATCH_SIZE = cfg.values.val_args.val_batch_size

    seed_everything(SEED)

    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')

    train_path = 'train.json'
    val_path = 'val.json'

    train_transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    val_transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    train_loader = get_dataloader(data_dir=train_path, mode='train', transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = get_dataloader(data_dir=val_path, mode='val', transform=val_transform, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    train(cfg, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/config.yml')
    parser.add_argument('--config', type=str, default='base')
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)