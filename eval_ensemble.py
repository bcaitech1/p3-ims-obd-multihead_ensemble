import os
import cv2
import argparse
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


def test(cfg, crf):    
    SEED = cfg.values.seed    
    BACKBONE_1 = cfg.values.backbone1
    BACKBONE_2 = cfg.values.backbone2
    MODEL_ARC_1 = cfg.values.model_arc1
    MODEL_ARC_2 = cfg.values.model_arc2
    IMAGE_SIZE = cfg.values.image_size
    NUM_CLASSES = cfg.values.num_classes

    checkpoint1 = cfg.values.checkpoint1
    checkpoint2 = cfg.values.checkpoint2
    test_batch_size = cfg.values.test_batch_size

    # for reproducibility
    seed_everything(SEED)

    data_path = '/opt/ml/input/data'
    test_annot = os.path.join(data_path, 'test.json')
    checkpoint_path = '/opt/ml/p3-ims-obd-multihead_ensemble/ckpts'

    test_transform = albumentations.Compose([
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
    
    size = 256
    resize = albumentations.Resize(size, size)
    
    test_loader = get_dataloader(data_dir='test.json', mode='test', transform=test_transform, batch_size=test_batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    aux_params=dict(
        pooling='avg',
        dropout=0.5,
        activation=None,
        classes=12
    )
    
    model_module1 = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC_1)
    model1 = model_module1(
        encoder_name=BACKBONE_1,
        in_channels=3,
        classes=NUM_CLASSES,
        aux_params = aux_params
    )
    model1 = model1.to(device)
    model1.load_state_dict(torch.load(os.path.join(checkpoint_path, checkpoint1)))
    
    
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
    
    model2 = UnetPlusPlus()
    
#     model_module2 = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC_2)
#     model2 = model_module2(
#         encoder_name=BACKBONE_2,
#         in_channels=3,
#         classes=NUM_CLASSES
#     )
    model2 = model2.to(device)
    model2.load_state_dict(torch.load(os.path.join(checkpoint_path, checkpoint2)))  
    
    
    print('Start prediction.')
    model1.eval()
    model2.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.compat.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader, desc='Test : ')):

            # Weighted soft voting
            outs1, _ = model1(torch.stack(imgs).to(device))
            outs2 = model2(torch.stack(imgs).to(device))
            
            outs  = 0.4 * outs1 + 0.6 * outs2 

            probs = F.softmax(outs, dim=1).data.cpu().numpy()
            
            if crf:                
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
    submission.to_csv(f"./submission/ensemble.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_config_file_path', type=str, default='./config/eval_config.yml')
    parser.add_argument('--eval_config', type=str, default='base')
    parser.add_argument('--crf', type=bool, default=False)
    parser.add_argument('--save_name', type=str, default='base')
    
    args = parser.parse_args()
    cfg = YamlConfigManager(args.eval_config_file_path, args.eval_config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')
    print(f'CRF : {args.crf}')
    make_submission(cfg, args.crf)