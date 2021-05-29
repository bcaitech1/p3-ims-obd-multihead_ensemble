import os
import yaml
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import cv2

from adamp import SGDP
from tqdm import tqdm
from glob import glob
from importlib import import_module
from prettyprinter import cpprint

from src.losses import DiceLoss
from src.utils import AverageMeter, YamlConfigManager, pixel_accuracy, mIoU, get_learning_rate, CosineAnnealingWarmupRestarts, YamlConfigManager, get_dataloader
from src.hrnet import get_seg_model


def presudo_train(cfg, presudo_loader):
    # Set Config
    BACKBONE = cfg.values.backbone
    BACKBONE_WEIGHT = cfg.values.backbone_weight
    MODEL_ARC = cfg.values.model_arc
    OUTPUT_DIR = cfg.values.output_dir
    NUM_CLASSES = cfg.values.num_classes

    LAMBDA = 0.75
    AUX_WEIGHT = 0.4

    SAVE_PATH = os.path.join(OUTPUT_DIR, MODEL_ARC)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Set train arguments
    num_epochs = 10
    weight_decay = cfg.values.train_args.weight_decay    
    accumulation_step = cfg.values.train_args.accumulation_step
    log_intervals = 5
    '''
    model_module = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC)    

    aux_params=dict(
        pooling='avg',
        dropout=0.5,
        activation='sigmoid',
        classes=12
    )
    
    model = model_module(
        encoder_name=BACKBONE,
        encoder_weights=BACKBONE_WEIGHT,
        in_channels=3,
        classes=NUM_CLASSES,
        aux_params=aux_params
    )
    '''
       
    config_path = './config/hrnet_config.yml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = get_seg_model(cfg)

    model.to(device)

    model.load_state_dict(torch.load('/opt/ml/vim-hjk/results/HRNet-OCR/49_epoch_58.91%.pth'))

    optimizer = SGDP(model.parameters(), lr=1e-5, weight_decay=weight_decay)
    
    auxilary_criterion = nn.CrossEntropyLoss()

    criterion = nn.CrossEntropyLoss()
    criterion_2 = DiceLoss()

    for epoch in range(num_epochs):
        model.train()

        loss_values = AverageMeter()
        mIoU_values = AverageMeter()
        accuracy = AverageMeter()

        for i, (images, masks, one_hot_label, _) in enumerate(tqdm(presudo_loader, desc=f'Training')):
            images = torch.stack(images)       
            masks = torch.stack(masks).long()
            one_hot_label = torch.stack(one_hot_label).float()

            images, masks = images.to(device), masks.to(device)
            one_hot_label = one_hot_label.to(device)

            '''
            logits, aux_logits = model(images)

            aux_loss = auxilary_criterion(aux_logits, one_hot_label)

            loss_1 = criterion(logits, masks)
            loss_2 = criterion_2(logits, masks)      

            loss = loss_1 * LAMBDA + loss_2 * (1 - LAMBDA) + aux_loss * AUX_WEIGHT            
            
            acc = pixel_accuracy(logits, masks)
            m_iou = mIoU(logits, masks)       
            '''

            logits = model(images)

            for j in range(len(logits)):
                pred = logits[j]
                ph, pw = pred.size(2), pred.size(3)
                h, w = masks.size(1), masks.size(2)
                if ph != h or pw != w:
                    pred = F.interpolate(input=pred, size=(
                        h, w), mode='bilinear', align_corners=True)
                logits[j] = pred

            aux_loss = auxilary_criterion(logits[1], masks)

            loss_1 = criterion(logits[0], masks)
            loss_2 = criterion_2(logits[0], masks)

            loss = loss_1 * LAMBDA + loss_2 * (1 - LAMBDA) + aux_loss * AUX_WEIGHT

            acc = pixel_accuracy(logits[0], masks) 
            m_iou = mIoU(logits[0], masks)

            loss_values.update(loss.item(), images.size(0))
            mIoU_values.update(m_iou.item(), images.size(0))
            accuracy.update(acc.item(), images.size(0))

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            if i % log_intervals == 0:
                tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}][{i}/{len(presudo_loader)}] || '
                           f'LR : {get_learning_rate(optimizer)[0]:.6e} ||'
                           f'Train Loss : {loss_values.val:.4f} ({loss_values.avg:.4f}) || '
                           f'Train Pixel Acc : {accuracy.val * 100.0:.3f}% ({accuracy.avg * 100.0:.4f}%) || '
                           f'Train mean IoU : {mIoU_values.val * 100.0:.3f}% ({mIoU_values.avg * 100.0:.3f}%)')
    
    print(f"Save : {os.path.join(SAVE_PATH, f'{epoch + 1}_epoch.pth')}")
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'{epoch + 1}_epoch.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='./config/config.yml')
    parser.add_argument('--config', type=str, default='base')
    args = parser.parse_args()
    
    cfg = YamlConfigManager(args.config_file_path, args.config)
    cpprint(cfg.values, sort_dict_keys=False)
    print('\n')

    import albumentations
    import albumentations.pytorch


    presudo_transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
        albumentations.pytorch.transforms.ToTensorV2()])

    presudo_loader = get_dataloader(transform=presudo_transform, mode='presudo', batch_size=cfg.values.train_args.train_batch_size, shuffle=True)

    presudo_train(cfg, presudo_loader)