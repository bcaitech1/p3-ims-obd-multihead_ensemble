import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from glob import glob
from importlib import import_module
from torchsummary import summary as summary_

import segmentation_models_pytorch as smp

# opimizer
import torch.optim as optim
import torch_optimizer as _optim
from madgrad import MADGRAD
import torch.optim.lr_scheduler

# other path 
from .losses import *
from .utils import *
from .scheduler import *


def train(cfg, run_name, train_loader, val_loader):
    # Set Config
    BACKBONE = cfg.values.backbone
    BACKBONE_WEIGHT = cfg.values.backbone_weight
    MODEL_ARC = cfg.values.model_arc
    OUTPUT_DIR = cfg.values.output_dir
    NUM_CLASSES = cfg.values.num_classes
    RUN_NAME = run_name
    LAMBDA = 0.75

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
    max_lr = cfg.values.train_args.max_lr     # for cosineannealingscheduler
    min_lr = cfg.values.train_args.min_lr     # for cosineannealingscheduler
    weight_decay = cfg.values.train_args.weight_decay    

    aux_params=dict(
        pooling='avg',
        dropout=0.5,
        activation=None,
        classes=12
    )   
    
    # define model 
    model_module = getattr(import_module('segmentation_models_pytorch'), MODEL_ARC)    
    model = model_module(
        encoder_name=BACKBONE,
        encoder_weights=BACKBONE_WEIGHT,
        in_channels=3,
        classes=NUM_CLASSES,
        aux_params=aux_params
    )
    model.to(device)

    # resume
    if cfg.value.train.load_state_dict :
        model.load_state_dict(torch.load('/opt/ml/p3-ims-obd-multihead_ensemble/ckpts/madgrad/best_mIoU.pth'))
    

    # define optimizer 
    OPTIMIZER = cfg.values.optimizer
    
    if OPTIMIZER == 'MADGRAD':
        optimizer = MADGRAD(params=model.parameters(),lr=min_lr, weight_decay=weight_decay) 
        
    elif OPTIMIZER in ['AdamW' , 'Adam']:
        opt_module = getattr(import_module("torch.optim") ,OPTIMIZER) #default : Adam
        optimizer = opt_module(params=model.parameters(),lr=min_lr, weight_decay=weight_decay) 
        
    else :
        opt_module = getattr(import_module("torch_optimizer") ,OPTIMIZER)
        optimizer = opt_module(params=model.parameters(),lr=min_lr, weight_decay=weight_decay) 
    
    # define scheduler
    first_cycle_steps = len(train_loader) * num_epochs // 3
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=first_cycle_steps, 
        cycle_mult=1.0, 
        max_lr=max_lr, 
        min_lr=min_lr, 
        warmup_steps=int(first_cycle_steps * 0.25), 
        gamma=0.5
    )

    # criterion 
    CRITEROIN  = cfg.values.criterion
    criterion = criterion = create_criterion(CRITEROIN)
   

    best_loss = float("INF")
    best_mIoU = 0
    EARLY_NUM = cfg.values.early_num

    wandb.watch(model)
    for epoch in range(num_epochs):

        model.train()
        trn_losses = []
        hist = np.zeros((12, 12))

        with tqdm(train_loader, total=len(train_loader), unit='batch') as trn_bar:
            for batch, (images , masks) in enumerate(trn_bar):
                trn_bar.set_description(f"Train Epoch {epoch+1}")
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  
                
                images, masks = images.to(device), masks.to(device)
                
                preds, _ = model(images)
                loss = criterion(preds, masks)
                
                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=12)
                trn_mIoU = label_accuracy_score(hist)[2]
                trn_losses.append(loss.item())

                wandb.log({
                    'Learning rate': get_learning_rate(optimizer)[0],
                    'Train Loss value': np.mean(trn_losses),
                    'Train mean IoU value': trn_mIoU * 100.0,
                })

                trn_bar.set_postfix(trn_loss=np.mean(trn_losses), trn_mIoU=trn_mIoU)
            
        model.eval()
        val_losses = []
        hist = np.zeros((12, 12))
        
        with torch.no_grad():
            with tqdm(val_loader, total=len(val_loader), unit='batch') as val_bar:
                 for batch, (images , masks) in enumerate(val_bar):
                    val_bar.set_description(f"Valid Epoch {epoch+1}")
                    
                    images = torch.stack(images)       
                    masks = torch.stack(masks).long()  
                    images, masks = images.to(device), masks.to(device).long()

                    preds, _ = model(images)
                    loss = criterion(preds, masks)
                    val_losses.append(loss.item())

                    preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                    hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=12)
                    val_mIoU = label_accuracy_score(hist)[2]

                    val_bar.set_postfix(val_loss=np.mean(val_losses),
                                        val_mIoU=val_mIoU)

        wandb.log({
            'Valid Loss value': np.mean(val_losses),
            'Valid mean IoU value': val_mIoU * 100.0,
        })

                
        save_model(model, version=run_name, save_type='current')
        val_loss  = np.mean(val_losses)
        if best_loss > val_loss:
            best_loss = val_loss
            save_model(model, version=run_name, save_type='loss')
         
        if best_mIoU < val_mIoU:
            best_mIoU = val_mIoU
            save_model(model, version=run_name, save_type='mIoU')
        
    wandb.finish()
    return best_score