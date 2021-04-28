import os
import cv2
import math
import numpy as np
import argparse
import albumentations
import albumentations.pytorch
import segmentation_models_pytorch as smp

from importlib import import_module
from prettyprinter import cpprint

import torch

from src.utils import seed_everything, YamlConfigManager, get_dataloader
from src.model import *


def inference(cfg, limit):    
    SEED = cfg.values.seed    
    BACKBONE = cfg.values.backbone
    MODEL_ARC = cfg.values.model_arc
    NUM_CLASSES = cfg.values.num_classes
    SAVE_IMG_PATH = './prediction/'

    COLORS =[
        [129, 236, 236],
        [2, 132, 227],
        [232, 67, 147],
        [255, 234, 267],
        [0, 184, 148],
        [85, 239, 196],
        [48, 51, 107],
        [255, 159, 26],
        [255, 204, 204],
        [179, 57, 57],
        [248, 243, 212],
    ]

    COLORS = np.vstack([[0, 0, 0], COLORS]).astype('uint8')

    os.makedirs(os.path.join(SAVE_IMG_PATH, MODEL_ARC), exist_ok=True)

    checkpoint = cfg.values.checkpoint
    test_batch_size = 1

    # for reproducibility
    seed_everything(SEED)

    data_path = '/opt/ml/input/data'
    test_annot = os.path.join(data_path, 'test.json')
    checkpoint_path = f'/opt/ml/vim-hjk/results/{MODEL_ARC}'

    test_transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    test_loader = get_dataloader(data_dir=test_annot, mode='test', transform=None, batch_size=test_batch_size, shuffle=False)

    LIMIT = limit if isinstance(limit, int) else len(test_loader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC)

    model = model_module(
        encoder_name=BACKBONE,
        in_channels=3,
        classes=NUM_CLASSES
    )

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, checkpoint)))
    print('Start Inference.\n')
    model.eval()

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            image_infos = image_infos
            image = test_transform(image=np.stack(imgs)[0])['image'].unsqueeze(0)

            outs = model(image.to(device))
            oms = torch.argmax(outs.squeeze(), dim=0).detach().cpu().numpy()
                        
            org = np.stack(imgs)[0]
            mask = COLORS[oms]
            output = ((0.4 * org) + (0.6 * mask)).astype('uint8')
            
            cv2.imwrite(os.path.join(SAVE_IMG_PATH, MODEL_ARC, f'{step}.jpg'), output)
            
            if step % 10 == 0:
                print(f'Progress({step + 1}/{LIMIT})...')

            if (step + 1) == LIMIT: break
    print('\nInference Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_config_file_path', type=str, default='./config/eval_config.yml')
    parser.add_argument('--eval_config', type=str, default='base')
    parser.add_argument('--limit', type=str, default='all')
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.eval_config_file_path, args.eval_config)
    cpprint(cfg.values, sort_dict_keys=False)    

    args = parser.parse_args()

    try:
        limit = int(args.limit)
    except:
        limit = args.limit

    print('\n')
    inference(cfg, limit)
