import os
import cv2
import argparse
import numpy as np
import pandas as pd
import albumentations
import albumentations.pytorch
import segmentation_models_pytorch as smp

from importlib import import_module
from prettyprinter import cpprint

import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from .utils import seed_everything, YamlConfigManager
from .utils import make_cat_df
from .utils import cls_colors
from .dataset import RecycleTrashDataset
from .model import *


def main(cfg):    
    SEED = cfg.values.seed    
    BACKBONE = cfg.values.backbone
    MODEL_ARC = cfg.values.model_arc
    NUM_CLASSES = cfg.values.num_classes
    DEBUG = cfg.values.debug

    checkpoint_path = cfg.values.checkpoint_path
    test_batch_size = cfg.values.test_batch_size


    # for reproducibility
    seed_everything(SEED)

    data_path = '/opt/ml/input/data'
    test_annot = os.path.join(data_path, 'test.json')
    test_cat = make_cat_df(test_annot, debug=True)

    test_transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    size = 256
    resize = albumentations.Resize(size, size)

    test_ds = RecycleTrashDataset(data_dir=test_annot, cat_df=test_cat, mode='test', transform=test_transform)
    test_dl = DataLoader(dataset=test_ds,
                         batch_size=test_batch_size,
                         shuffle=False,
                         num_workers=3)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC)

    model = model_module(
        encoder_name=BACKBONE,
        in_channels=3,
        classes=NUM_CLASSES
    )

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join("./ckpts", checkpoint_path)))

    model.eval()
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    cnt = 1
    with torch.no_grad():
        for step, sample in enumerate(test_dl):
            imgs = sample['image']
            file_names = sample['info']

            # inference (512 x 512)
            preds = model(imgs.to(device))
            oms = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()

            if DEBUG:
                debug_path = os.path.join('.', 'debug', 'test', BACKBONE)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)

                pred_masks = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
                for idx, file_name in enumerate(file_names):
                    pred_mask = pred_masks[idx]
                    ori_image = cv2.imread(os.path.join('.', 'input', 'data', file_name))
                    ori_image = ori_image.astype(np.float32)

                    for i in range(1, 12):
                        a_mask = (pred_mask == i)
                        cls_mask = np.zeros(ori_image.shape).astype(np.float32)
                        cls_mask[a_mask] = cls_colors[i]
                        ori_image[a_mask] = cv2.addWeighted(ori_image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

                    cv2.imwrite(os.path.join(debug_path, f"{cnt}.jpg"), ori_image)
                    cnt += 1

            # resize (256 x 256)
            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            for img, mask in zip(temp_images, oms):
                transformed = resize(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([file_name for file_name in file_names])
    print("End prediction.")

    print("Saving...")
    file_names = [y for x in file_name_list for y in x]
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    save_path = './submission'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'{BACKBONE}.csv')
    submission.to_csv(save_dir, index=False)
    print("All done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/eval_config.yml')
    parser.add_argument('--config', type=str, default='base')
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    main(cfg)