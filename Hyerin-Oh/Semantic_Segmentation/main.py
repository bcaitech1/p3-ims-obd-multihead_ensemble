import os
import numpy as np
import wandb
import torch
import argparse
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

from prettyprinter import cpprint

from src.train import train
from src.utils import YamlConfigManager, seed_everything, get_dataloader
from src.add_train import *

def main(cfg):
    SEED = cfg.values.seed
    TRAIN_BATCH_SIZE = cfg.values.train_args.train_batch_size
    VAL_BATCH_SIZE = cfg.values.val_args.val_batch_size
    IMAGE_SIZE = cfg.values.image_size
    NUM_WORKERS = cfg.values.num_workers
    USE_AUGMIX = cfg.values.augmix_args.use_augmix
    AUGMIX_PROB = cfg.values.augmix_args.augmix_prob
    RUN_NAME = cfg.values.run_name
    

    # for reprodution
    seed_everything(SEED)
    
    print(f'Cuda is Available ? : {torch.cuda.is_available()}\n')
    
    # define path
    train_path = 'train.json'
    val_path = 'val.json'
    augmix_path = './augmix.npy'

    # define whether use augmix
    if USE_AUGMIX:
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        augmix = np.load(augmix_path)
        augmix = augmix.item()

    else:
        augmix_data = None
    
    # define augmentation
    train_transform = A.Compose([
          A.Resize (IMAGE_SIZE, IMAGE_SIZE , p=1),
          A.Rotate (limit = 30, p=1),
          A.Cutout (num_holes=4, max_h_size=20, max_w_size=20,p=0.5),
          A.CLAHE (p=1),
          A.OneOf([
              A.ElasticTransform(p=0.5, alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03),
              A.GridDistortion(p=0.5),
          ], p = 0.5),
          A.RandomBrightnessContrast(brightness_by_max=False, p=0.5),
          A.Normalize(),
          ToTensorV2()
        ])
    
    val_transform = A.Compose([
            A.Normalize(),
            A.Resize (IMAGE_SIZE , IMAGE_SIZE , p=1),
            ToTensorV2()
        ])


    
    # define loader
    train_loader = get_dataloader(data_dir=train_path, mode='train', transform=train_transform, batch_size=TRAIN_BATCH_SIZE, shuffle=True, augmix=augmix, augmix_prob=AUGMIX_PROB, num_workers=NUM_WORKERS)
    val_loader = get_dataloader(data_dir=val_path, mode='val', transform=val_transform, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    psuedo_loader = get_dataloader(data_dir=val_path, mode='psuedo', transform=val_transform, batch_size=VAL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # choose train mode : only train or train with pseudo data
    if cfg.values.train_mode == 'pseudo': 
        pseudo_train(cfg, args.run_name, train_loader, psuedo_loader, val_loader)
    else :
        train(cfg, args.run_name, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/config.yml')
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--run_name', type = str , default = 'base')
    args = parser.parse_args()    
    wandb.init(project='segmentation', entity='hyerin', name = args.run_name , save_code = True)
    wandb.run.name = args.config
    wandb.run.save()

    cfg = YamlConfigManager(args.config_file_path, args.config)
    wandb.config.update(cfg)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)