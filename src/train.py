import os
import torch
import torch.nn as nn

from tqdm import tqdm
from adamp import AdamP
from torchsummary import summary as summary_
from importlib import import_module

from . import lovasz_losses as L
from .utils import AverageMeter, pixel_accuracy, mIoU
from .losses import *
from .model import CustomFCN8s, CustomFCN16s, CustomFCN32s


def train(cfg, train_loader, val_loader):
    # Set Config
    BACKBONE=cfg.values.backbone
    MODEL_ARC = cfg.values.model_arc
    OUTPUT_DIR = cfg.values.output_dir
    NUM_CLASSES = cfg.values.num_classes

    LAMBDA = 5.0

    os.makedirs(os.path.join(OUTPUT_DIR, MODEL_ARC), exist_ok=True)

    # Set train arguments
    num_epochs = cfg.values.train_args.num_epochs
    train_batch_size = cfg.values.train_args.train_batch_size
    log_intervals = cfg.values.train_args.log_intervals
    learning_rate = cfg.values.train_args.lr
    weight_decay = cfg.values.train_args.weight_decay

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC)

    model = model_module(
        encoder_name=BACKBONE,
        in_channels=3,
        classes=NUM_CLASSES
    )
    
    model.to(device)
    summary_(model, (3, 512, 512), train_batch_size)

    optimizer = AdamP(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()    

    best_score = 0.

    for epoch in range(num_epochs):
        model.train()

        loss_values = AverageMeter()
        mIoU_values = AverageMeter()
        accuracy = AverageMeter()

        for i, (images, masks, _) in enumerate(tqdm(train_loader, desc=f'Training')):
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)

            logits = model(images)

            loss_1 = criterion(logits, masks)
            loss_2 = L.lovasz_softmax(logits.argmax(dim=1), masks, ignore=0)
            loss = loss_1 + (loss_2 * LAMBDA)
            
            acc = pixel_accuracy(logits, masks)
            m_iou = mIoU(logits, masks)       

            loss_values.update(loss.item(), images.size(0))
            mIoU_values.update(m_iou.item(), images.size(0))
            accuracy.update(acc.item(), images.size(0))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_intervals == 0:
                tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}][{i}/{len(train_loader)}] || '
                           f'Train Loss : {loss_values.val:.4f} ({loss_values.avg:.4f}) || '
                           f'Train Accuracy : {accuracy.val * 100.0:.3f}% ({accuracy.avg * 100.0:.4f}%) || '
                           f'Train mean IoU : {mIoU_values.val * 100.0:.3f}% ({mIoU_values.avg * 100.0:.3f}%)')


        with torch.no_grad():
            model.eval()

            val_loss_values = AverageMeter()
            val_mIoU_values = AverageMeter()
            val_accuracy = AverageMeter()            

            for i, (images, masks, _) in enumerate(tqdm(val_loader, desc=f'Validation')):
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)

                logits = model(images)
                
                loss_1 = criterion(logits, masks)
                loss_2 = L.lovasz_softmax(logits.argmax(dim=1), masks, ignore=0)
                loss = loss_1 + (loss_2 * LAMBDA)

                acc = pixel_accuracy(logits, masks)
                m_iou = mIoU(logits, masks)                 

                val_loss_values.update(loss.item(), images.size(0))
                val_mIoU_values.update(m_iou.item(), images.size(0))
                val_accuracy.update(acc.item(), images.size(0))

        tqdm.write(f'Epoch : [{epoch + 1}/{num_epochs}] || '
                   f'Val Loss : {val_loss_values.avg:.4f} || '
                   f'Val Accuracy : {val_accuracy.avg * 100.0:.3f}% || '
                   f'Val mean IoU : {val_mIoU_values.avg * 100.0:.3f}%')

        is_best = mIoU_values.avg >= best_score
        best_score = max(mIoU_values.avg, best_score)

        if is_best:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_ARC, f'{epoch + 1}_epoch_{best_score * 100.0:.2f}%_with_val.pth'))