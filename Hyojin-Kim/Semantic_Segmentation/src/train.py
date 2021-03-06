import os
import yaml
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import cv2

from tqdm import tqdm
from glob import glob
from adamp import AdamP
from torchsummary import summary as summary_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from importlib import import_module

from .losses import DiceLoss, LabelSmoothingLoss, FocalLoss
from .utils import AverageMeter, pixel_accuracy, mIoU, get_learning_rate, CosineAnnealingWarmupRestarts, denormalize_image
from . import lovasz_losses as L
from .model import CustomFCN8s, CustomFCN16s, CustomFCN32s
from .hrnet import get_seg_model


def train(cfg, train_loader, val_loader):
    # Set Config
    BACKBONE = cfg.values.backbone
    BACKBONE_WEIGHT = cfg.values.backbone_weight
    MODEL_ARC = cfg.values.model_arc
    OUTPUT_DIR = cfg.values.output_dir
    NUM_CLASSES = cfg.values.num_classes

    LAMBDA = 0.75
    AUX_WEIGHT = 0.4

    SAVE_PATH = os.path.join(OUTPUT_DIR, MODEL_ARC)

    class_labels ={
        0: 'Background',
        1: 'UNKNOWN',
        2: 'General trash',
        3: 'Paper',
        4: 'Paper pack',
        5: 'Metal',
        6: 'Glass',
        7: 'Plastic',
        8: 'Styrofoam',
        9: 'Plastic bag',
        10: 'Battery',
        11: 'Clothing'
    }

    pos_weight = [
        0.3040,
        0.9994,
        0.9778,
        0.9097,
        0.9930,
        0.9911,
        0.9924,
        0.9713,
        0.9851,
        0.8821,
        0.9995,
        0.9947
    ]    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pos_weight = torch.tensor(pos_weight).float().to(device)

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Set train arguments
    num_epochs = cfg.values.train_args.num_epochs    
    max_lr = cfg.values.train_args.max_lr
    min_lr = cfg.values.train_args.min_lr
    weight_decay = cfg.values.train_args.weight_decay    
    accumulation_step = cfg.values.train_args.accumulation_step
    log_intervals = cfg.values.train_args.train_batch_size * accumulation_step

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

    config_path = '/opt/ml/vim-hjk/config/hrnet_config.yml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = get_seg_model(cfg)
    model.load_state_dict(torch.load('/opt/ml/vim-hjk/results/HRNet-OCR/best/35_epoch_58.50%.pth'))
    model.to(device)
    
    # summary_(model, (3, 512, 512), train_batch_size)

    optimizer = AdamP(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    first_cycle_steps = len(train_loader) * num_epochs // 3 // accumulation_step

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=first_cycle_steps, 
        cycle_mult=1.0,
        max_lr=max_lr, 
        min_lr=min_lr, 
        warmup_steps=int(first_cycle_steps * 0.25), 
        gamma=1
    )

    auxilary_criterion = nn.CrossEntropyLoss()

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(classes=12, smoothing=0.1, weight=pos_weight)
    criterion_2 = DiceLoss()

    wandb.watch(model)

    best_score = 0.

    for epoch in range(num_epochs):
        model.train()

        loss_values = AverageMeter()
        mIoU_values = AverageMeter()
        accuracy = AverageMeter()

        for i, (images, masks, one_hot_label, _) in enumerate(tqdm(train_loader, desc=f'Training')):
            images = torch.stack(images)       
            masks = torch.stack(masks).long()
            one_hot_label = torch.stack(one_hot_label).float()

            images, masks = images.to(device), masks.to(device)

            '''
            one_hot_label = one_hot_label.to(device)

            
            logits, aux_logits = model(images)

            aux_loss = auxilary_criterion(aux_logits, one_hot_label)

            loss_1 = criterion(logits, masks)
            loss_2 = criterion_2(logits, masks)

            loss = loss_1 * LAMBDA + loss_2 * (1 - LAMBDA) + aux_loss * AUX_WEIGHT            
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

            # acc = pixel_accuracy(logits, masks)
            # m_iou = mIoU(logits, masks)       

            loss_values.update(loss.item(), images.size(0))
            mIoU_values.update(m_iou.item(), images.size(0))
            accuracy.update(acc.item(), images.size(0))

            # normalize loss to account for batch accumulation & backward pass
            loss = loss / accumulation_step
            loss.backward()

            # weights update
            if ((i + 1) % accumulation_step == 0) or (i + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            wandb.log({
                'Learning rate' : get_learning_rate(optimizer)[0],
                'Train Loss value' : loss_values.val,
                'Train Pixel Accuracy value' : accuracy.val * 100.0,
                'Train mean IoU value' : mIoU_values.val * 100.0,
            })

            if i % log_intervals == 0:
                tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}][{i}/{len(train_loader)}] || '
                           f'LR : {get_learning_rate(optimizer)[0]:.6e} ||'
                           f'Train Loss : {loss_values.val:.4f} ({loss_values.avg:.4f}) || '
                           f'Train Pixel Acc : {accuracy.val * 100.0:.3f}% ({accuracy.avg * 100.0:.4f}%) || '
                           f'Train mean IoU : {mIoU_values.val * 100.0:.3f}% ({mIoU_values.avg * 100.0:.3f}%)')
            
        
        with torch.no_grad():
            model.eval()

            val_loss_values = AverageMeter()
            val_mIoU_values = AverageMeter()
            val_accuracy = AverageMeter()            
            
            example_images = []

            for i, (images, masks, _, _) in enumerate(tqdm(val_loader, desc=f'Validation')):
                
                images = torch.stack(images)
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)

                '''
                logits, _ = model(images)

                loss_1 = criterion(logits, masks)
                loss_2 = criterion_2(logits, masks)

                loss = loss_1 * LAMBDA + loss_2 * (1 - LAMBDA)
                '''

                logits = model(images)

                for i in range(len(logits)):
                    pred = logits[i]
                    ph, pw = pred.size(2), pred.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        pred = F.interpolate(input=pred, size=(
                            h, w), mode='bilinear', align_corners=True)
                    logits[i] = pred

                aux_loss = auxilary_criterion(logits[1], masks)

                loss_1 = criterion(logits[0], masks)
                loss_2 = criterion_2(logits[0], masks)

                loss = loss_1 * LAMBDA + loss_2 * (1 - LAMBDA) + aux_loss * AUX_WEIGHT

                # acc = pixel_accuracy(logits, masks)
                # m_iou = mIoU(logits, masks)        

                acc = pixel_accuracy(logits[0], masks) 
                m_iou = mIoU(logits[0], masks)        

                val_loss_values.update(loss.item(), images.size(0))
                val_mIoU_values.update(m_iou.item(), images.size(0))
                val_accuracy.update(acc.item(), images.size(0))

                inputs_np = torch.clone(images).detach().cpu().permute(0, 2, 3, 1).numpy()
                inputs_np = denormalize_image(inputs_np, mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216))
                
                example_images.append(wandb.Image(inputs_np[0], masks={
                    "predictions" : {
                        "mask_data" : logits[0].argmax(1)[0].detach().cpu().numpy(),
                        "class_labels" : class_labels
                    },
                    "ground-truth" : {
                        "mask_data" : masks[0].detach().cpu().numpy(),
                        "class_labels" : class_labels
                    }
                }))

        wandb.log({
            'Example Image' : example_images,
            'Validation Loss average': val_loss_values.avg,
            'Validation Pixel Accuracy average' : val_accuracy.avg * 100.0,
            'Validation mean IoU average' : val_mIoU_values.avg * 100.0
        })

        tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}] || '
                   f'Val Loss : {val_loss_values.avg:.4f} || '
                   f'Val Accuracy : {val_accuracy.avg * 100.0:.3f}% || '
                   f'Val mean IoU : {val_mIoU_values.avg * 100.0:.3f}%')

        is_best = val_mIoU_values.avg >= best_score
        best_score = max(val_mIoU_values.avg, best_score)

        if is_best:
            if len(glob(SAVE_PATH + '/*.pth')) > 5:
                os.remove(glob(SAVE_PATH + '/*.pth')[-1])
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'{epoch + 1}_epoch_{best_score * 100.0:.2f}%.pth'))
        

    return best_score