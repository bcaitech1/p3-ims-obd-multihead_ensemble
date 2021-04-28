import os
import cv2
import json
import random
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


num_cls = 12
cmap = plt.get_cmap("rainbow")
colors = [cmap(i) for i in np.linspace(0, 1, num_cls+2)]
colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
cls_colors = {k: colors[k] for k in range(num_cls+1)}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def make_cat_df(train_annot_path, debug=False):
    # Read annotations
    with open(train_annot_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    if debug:
        print('Number of super categories:', nr_super_cats)
        print('Number of categories:', nr_cats)
        print('Number of annotations:', nr_annotations)
        print('Number of images:', nr_images)
        print()

    cat_histogram = np.zeros(nr_cats, dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']] += 1

    cat_df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    cat_df = cat_df.sort_values('Number of annotations', 0, False)
    sorted_df = pd.DataFrame(["Backgroud"], columns=["Categories"])
    sorted_df = sorted_df.append(cat_df.sort_index(), ignore_index=True)

    return sorted_df


# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def save_model(model, version, save_type='loss'):
    save_path = os.path.join(f'./ckpts/{version}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'best_{save_type}.pth')
    torch.save(model.state_dict(), save_dir)


def rand_bbox(size, lam, mask):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)

    # uniform
    hs, ws = (mask > 0).nonzero(as_tuple=True)
    try:
        hmin, hmax = hs.min().item(), hs.max().item()
        wmin, wmax = ws.min().item(), ws.max().item()

        cut_w = np.int((wmax - wmin) * cut_rat)
        cut_h = np.int((hmax - hmin) * cut_rat)

        cx = np.random.randint(wmin, wmax)
        cy = np.random.randint(hmin, hmax)
    except Exception as e:
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, scheduler, logger, device, beta=0.0, debug=True):
    cnt = 1

    model.train()
    trn_losses = []
    hist = np.zeros((12, 12))
    logger.info(f"\nTrain on Epoch {epoch+1}")
    with tqdm(trn_dl, total=len(trn_dl), unit='batch') as trn_bar:
        for batch, sample in enumerate(trn_bar):
            trn_bar.set_description(f"Train Epoch {epoch+1}")

            optimizer.zero_grad()
            images, masks = sample['image'], sample['mask']
            images, masks = images.to(device), masks.to(device).long()

            if beta > 0:
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                for image_idx, rand_idx in enumerate(rand_index):
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam, masks[rand_idx])
                    images[image_idx, :, bbx1:bbx2, bby1:bby2] = images[rand_idx, :, bbx1:bbx2, bby1:bby2]
                    masks[image_idx, bbx1:bbx2, bby1:bby2] = masks[rand_idx, bbx1:bbx2, bby1:bby2]


            aux_preds = None
            preds = model(images)
            if isinstance(preds, collections.OrderedDict):
                preds = preds['out']
            elif isinstance(preds, list):
                aux_preds, preds = preds
                ph, pw = preds.size(2), preds.size(3)
                h, w = masks.size(1), masks.size(2)
                if ph != h or pw != w:
                    preds = F.interpolate(input=preds, size=(
                        h, w), mode='bilinear', align_corners=True)
                    aux_preds = F.interpolate(input=aux_preds, size=(
                        h, w), mode='bilinear', align_corners=True)


            loss = criterion(preds, masks)
            if aux_preds is not None:
                loss += 0.4 * criterion(aux_preds, masks)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=12)
            trn_mIoU = label_accuracy_score(hist)[2]

            trn_losses.append(loss.item())
            if (batch+1) % (int(len(trn_dl)//10)) == 0:
                logger.info(f'Train Epoch {epoch+1} ==>  Batch [{str(batch+1).zfill(len(str(len(trn_dl))))}/{len(trn_dl)}]  |  Loss: {np.mean(trn_losses):.5f}  |  mIoU: {trn_mIoU:.5f}')

            trn_bar.set_postfix(trn_loss=np.mean(trn_losses),
                                trn_mIoU=trn_mIoU)


    model.eval()
    val_losses = []
    hist = np.zeros((12, 12))
    logger.info(f"\nValid on Epoch {epoch+1}")
    with torch.no_grad():
        with tqdm(val_dl, total=len(val_dl), unit='batch') as val_bar:
            for batch, sample in enumerate(val_bar):
                val_bar.set_description(f"Valid Epoch {epoch+1}")

                images, masks = sample['image'], sample['mask']
                images, masks = images.to(device), masks.to(device).long()

                preds = model(images)
                if isinstance(preds, collections.OrderedDict):
                    preds = preds['out']
                elif isinstance(preds, list):
                    _, preds = preds
                    ph, pw = preds.size(2), preds.size(3)
                    h, w = masks.size(1), masks.size(2)
                    if ph != h or pw != w:
                        preds = F.interpolate(input=preds, size=(
                            h, w), mode='bilinear', align_corners=True)

                loss = criterion(preds, masks)
                val_losses.append(loss.item())

                if debug:
                    debug_path = os.path.join('.', 'debug', 'valid')
                    if not os.path.exists(debug_path):
                        os.makedirs(debug_path)

                    file_names = sample['info']
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

                preds = torch.argmax(preds, dim=1).detach().cpu().numpy()
                hist = add_hist(hist, masks.detach().cpu().numpy(), preds, n_class=12)
                val_mIoU = label_accuracy_score(hist)[2]

                if (batch + 1) % (int(len(trn_dl) // 10)) == 0:
                    logger.info(
                        f'Valid Epoch {epoch+1} ==>  Batch [{str(batch+1).zfill(len(str(len(val_dl))))}/{len(val_dl)}]  |  Loss: {np.mean(val_losses):.5f}  |  mIoU: {val_mIoU:.5f}')

                val_bar.set_postfix(val_loss=np.mean(val_losses),
                                    val_mIoU=val_mIoU)

    return np.mean(trn_losses), trn_mIoU, np.mean(val_losses), val_mIoU