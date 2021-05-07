import os
import wandb
import logging
import argparse
import importlib
import numpy as np
import albumentations as A
from collections import defaultdict
from albumentations.pytorch import ToTensorV2

import torch.optim as optim
import segmentation_models_pytorch as smp

from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import train, valid
from src.utils import save_model
from src.dataset import SegmentationDataset, PseudoDataset
from src.models.fcn8s import FCN8s
from src.losses import *
from src.schedulers import CosineAnnealingWarmupRestarts

from torch.utils.data import DataLoader
from torchvision.models import vgg16


def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--trn_ratio', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--decay', default=1e-7, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--version', default='test', type=str)
    parser.add_argument('--model_type', default='fcn8s', type=str)
    parser.add_argument('--cutmix_beta', default=1.0, type=float)
    parser.add_argument('--use_weight', default=1, type=int)
    parser.add_argument('--use_augmix', default=1, type=int)
    parser.add_argument('--loss_type', default='DiceCELoss', type=str)


    args = parser.parse_args()
    print(args)

    wandb.init(config=args, project="[Pstage-Seg]", name=args.version, save_code=True)

    # for reproducibility
    seed_everything(args.seed)

    # define key paths
    main_path = '.'
    data_path = os.path.join(main_path, 'input', 'data')
    augmix_path = os.path.join(main_path, 'input', 'augmix_all.npy')
    if args.use_augmix:
        print("Use SeoungBaeMix\n")
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        augmix_data = np.load(augmix_path)
        augmix_data = augmix_data.item()
    else:
        augmix_data = None

    mean = np.array([0.46098186, 0.44022841, 0.41892368], dtype=np.float32)
    std  = np.array([0.21072529, 0.20763867, 0.21613272], dtype=np.float32)

    # define transform
    trn_tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.2),
        A.OneOf([
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=30, p=1.0),
        ], p=0.5),

        A.RandomGamma(p=0.3),
        A.RandomBrightness(p=0.5),
        A.OneOf([
            A.CLAHE(p=1.0),
            A.ElasticTransform(p=1, alpha=40, sigma=40 * 0.05, alpha_affine=40 * 0.03),
            A.GridDistortion(p=1),
            # A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.6),


        A.Normalize(),
        ToTensorV2()
    ])

    val_tfms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    pseudo_tfms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.Normalize(),
        ToTensorV2()
    ])


    # define train & valid dataset
    if args.trn_ratio:
        total_annot = os.path.join(data_path, 'train_all.json')
        total_cat = make_cat_df(total_annot, debug=True)
        total_ds = SegmentationDataset(data_dir=total_annot, cat_df=total_cat, mode='train', transform=None)

        trn_size = int(len(total_ds)*0.8)
        val_size = int(len(total_ds) - trn_size)
        trn_ds, val_ds = torch.utils.data.random_split(total_ds, [trn_size, val_size])
    else:
        trn_annot = os.path.join(data_path, 'train.json')
        val_annot = os.path.join(data_path, 'val.json')
        trn_cat = make_cat_df(trn_annot, debug=True)
        val_cat = make_cat_df(val_annot, debug=True)


        trn_ds = SegmentationDataset(data_dir=trn_annot, cat_df=trn_cat, mode='train', transform=trn_tfms, augmix=augmix_data)
        weight_ds = SegmentationDataset(data_dir=trn_annot, cat_df=trn_cat, mode='valid', transform=val_tfms)
        val_ds = SegmentationDataset(data_dir=val_annot, cat_df=val_cat, mode='valid', transform=val_tfms)


    pseudo_ds = PseudoDataset(data_dir='input', pseudo_csv='pseudo_result2.csv', transform=pseudo_tfms)

    # define dataloader
    trn_dl = DataLoader(dataset=trn_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)

    pseudo_dl = DataLoader(dataset=pseudo_ds,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=4
    )

    weight_dl = DataLoader(dataset=weight_ds,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)

    val_dl = DataLoader(dataset=val_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'fcn8s':
        backbone = vgg16(pretrained=True)
        model = FCN8s(backbone=backbone)
    elif args.model_type == "unet":
        model = smp.Unet(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=12,  # model output channels (number of classes in your dataset)
        )
    elif args.model_type == "unet_pp":
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=12,  # model output channels (number of classes in your dataset)
        )
    elif args.model_type == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=12
        )
    elif args.model_type == 'torch_deeplab':
        import torchvision
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, aux_loss=False)
        classifier = list(model.classifier)
        classifier[4] = nn.Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
        model.classifier = nn.Sequential(*classifier)
    elif args.model_type == 'hrnet_ocr':
        import yaml
        from src.models.hrnet_seg import get_seg_model

        config_path = './src/configs/hrnet_seg_ocr.yaml'
        with open(config_path) as f:
            cfg = yaml.load(f)
        model = get_seg_model(cfg)
    elif args.model_type == 'hrnet_ocr_tp':
        import yaml
        from src.models.hrnet_seg_transpose import get_seg_model

        config_path = './src/configs/hrnet_seg_ocr.yaml'
        with open(config_path) as f:
            cfg = yaml.load(f)
        model = get_seg_model(cfg)

    model = model.to(device)
    wandb.watch(model)

    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    first_cycle_steps = (len(trn_dl) + len(pseudo_dl)) * args.epochs // 3
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.0001,
        min_lr=0.000001,
        warmup_steps=int(first_cycle_steps * 0.25),
        gamma=0.5
    )
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(trn_dl), epochs=args.epochs, anneal_strategy='cos')
    # scheduler = None

    normedWeights = None
    if args.use_weight:
        annot_cnts = defaultdict(int)
        for sample in weight_dl:
            masks = sample['mask']

            for cls in range(12):
                mask = masks == cls
                annot_cnts[cls] += mask.sum()

        annot_list = [val for _, val in sorted(annot_cnts.items(), key=lambda x: x[0])]
        normedWeights = [1 - (x / sum(annot_list)) for x in annot_list]
        normedWeights = torch.FloatTensor(normedWeights).to(device)
        print(normedWeights, "\n")

    # criterion = nn.CrossEntropyLoss()
    # criterion = DiceCELoss(weight=normedWeights)
    criterion = OhMyLoss(weight=normedWeights)
    # criterion = AmazingCELoss(weight=normedWeights)
    # criterion = IoULoss()
    # criterion = FocalLoss()

    logger = logging.getLogger("Segmentation")
    logger.setLevel(logging.INFO)
    logger_dir = f'./logs/'
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    file_handler = logging.FileHandler(os.path.join(logger_dir, f'{args.version}.log'))
    logger.addHandler(file_handler)


    best_loss = float("INF")
    best_mIoU = 0
    for epoch in range(args.epochs):
        train(epoch, model, trn_dl, pseudo_dl, criterion, optimizer, scheduler, logger, device, augmix_data=augmix_data)
        val_loss, val_mIoU = valid(epoch, model, val_dl, criterion, logger, device, debug=True)


        save_model(model, version=args.version, save_type='current')
        if best_loss > val_loss:
            logger.info(f"Best loss {best_loss:.5f} -> {val_loss:.5f}")
            best_loss = val_loss
            save_model(model, version=args.version, save_type='loss')

        if best_mIoU < val_mIoU:
            logger.info(f"Best mIoU {best_mIoU:.5f} -> {val_mIoU:.5f}")
            best_mIoU = val_mIoU
            save_model(model, version=args.version, save_type='mIoU')


if __name__ == "__main__":
    main()
