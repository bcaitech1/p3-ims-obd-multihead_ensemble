import os
from albumentations.augmentations.transforms import VerticalFlip
import cv2
import yaml
import argparse
import ttach as tta
import numpy as np
import pandas as pd
import multiprocessing as mp
import albumentations
import albumentations.pytorch
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F


from importlib import import_module
from prettyprinter import cpprint
from tqdm import tqdm

import torch

from src.utils import seed_everything, YamlConfigManager, get_dataloader, dense_crf_wrapper
from src.model import *
from src.hrnet import *


class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes=12):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            )
    def forward(self, x):
        return self.model(x)


def test(cfg, crf):    
    SEED = cfg.values.seed    
    BACKBONE = cfg.values.backbone
    MODEL_ARC = cfg.values.model_arc
    IMAGE_SIZE = cfg.values.image_size
    NUM_CLASSES = cfg.values.num_classes
    NUM_MODELS = cfg.values.num_models

    checkpoints = cfg.values.checkpoints
    test_batch_size = cfg.values.test_batch_size

    # for reproducibility
    seed_everything(SEED)

    data_path = '/opt/ml/input/data'
    test_annot = os.path.join(data_path, 'test.json')
    checkpoint_path = f'/opt/ml/vim-hjk/results/{MODEL_ARC}'

    test_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90([0, 90])
        ]
    )

    size = 256
    resize = albumentations.Resize(size, size)
    
    test_loader = get_dataloader(data_dir=test_annot, mode='test', transform=test_transform, batch_size=test_batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    '''
    model_module = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC)

    aux_params=dict(
        pooling='avg',
        dropout=0.5,
        activation=None,
        classes=12
    )

    model = model_module(
        encoder_name=BACKBONE,
        in_channels=3,
        classes=NUM_CLASSES,
        aux_params=aux_params
    )
    '''
    config_path = '/opt/ml/vim-hjk/config/hrnet_config.yml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    weights = [os.path.join(checkpoint_path, checkpoint) for checkpoint in checkpoints]
    
    #1 Hongyeop
    model_1 = get_seg_model(cfg, test=True)
    model_1 = model_1.to(device)
    model_1.load_state_dict(torch.load(os.path.join(checkpoint_path, weights[0])))
    model_1 = tta.SegmentationTTAWrapper(model_1, tta_transforms, merge_mode='mean')

    # 2 ug
    model_2 = UnetPlusPlus()
    model_2 = model_2.to(device)
    model_2.load_state_dict(torch.load(os.path.join(checkpoint_path, weights[1])))

    print('Start prediction.')
    model_1.eval()
    model_2.eval()

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.compat.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader, desc='Test : ')):

            # inference (512 x 512)
            '''
            outs, _ = model(torch.stack(imgs).to(device))
            probs = F.softmax(outs, dim=1).data.cpu().numpy()
            '''
            images = torch.stack(imgs).to(device)
            outs = model_1(images)

            pred = outs
            ph, pw = pred.size(2), pred.size(3)
            h, w = IMAGE_SIZE, IMAGE_SIZE
            if ph != h or pw != w:
                pred = F.interpolate(input=pred, size=(
                    h, w), mode='bilinear', align_corners=True)
            preds = pred

            outs = model_2(images)

            preds = torch.stack((preds, outs), dim=0).mean(dim=0)


            if crf:              
                probs = F.softmax(preds, dim=1).data.cpu().numpy()  
                pool = mp.Pool(mp.cpu_count())
                images = torch.stack(imgs).data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                probs = np.array(pool.map(dense_crf_wrapper, zip(images, probs)))
                pool.close()

            oms = np.argmax(probs, axis=1)
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = resize(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def make_submission(cfg, crf):
    # sample_submisson.csv 열기
    submission = pd.read_csv('../code/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(cfg, crf)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    os.makedirs('./submission', exist_ok=True)
    submission.to_csv(f"./submission/{cfg.values.backbone}_{cfg.values.model_arc}.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_config_file_path', type=str, default='./config/eval_config.yml')
    parser.add_argument('--eval_config', type=str, default='base')
    parser.add_argument('--crf', type=bool, default=False)
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.eval_config_file_path, args.eval_config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    print(f'CRF : {args.crf}')
    make_submission(cfg, args.crf)