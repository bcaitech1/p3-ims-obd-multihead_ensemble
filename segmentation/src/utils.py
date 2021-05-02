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
from albumentations.core.transforms_interface import DualTransform

from torch.cuda.amp import GradScaler, autocast  

num_cls = 12
cmap = plt.get_cmap("rainbow")
colors = [cmap(i) for i in np.linspace(0, 1, num_cls+2)]
colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
cls_colors = {k: colors[k] for k in range(num_cls+1)}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def Cutout(n_holes , length , imgs, label):
    h = imgs.size(2)
    w = imgs.size(3)
    for idx, img in enumerate(imgs):
        mask = np.ones((h, w), np.float32)
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            mask[y1: y2, x1: x2] = np.where(label[idx][y1: y2, x1: x2] > 0 , 1, mask[y1: y2, x1: x2]) ## background 아니면 살림

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)

        imgs[idx] = img * mask
            
    return imgs

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    # cut_rat = np.sqrt(1. - lam)
    # cut_w = np.int(W * cut_rat)
    # cut_h = np.int(H * cut_rat)

    # # uniform
    # cx = np.random.randint(W)
    # cy = np.random.randint(H)

    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbx1 , bby1, bbx2 , bby2 = 0, 0, int(W/2),int(H)
    return bbx1, bby1, bbx2, bby2

import time
def train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, logger, device, scheduler=None, cutout = 0, cutmix = 0 , augmix_data = None ,  debug=False):
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

            # if augmix_data is not None:
            #     images, masks = augmix_search(augmix_data.item(), images.numpy().astype(np.float32), masks.numpy())

            r = np.random.rand(1)
            if cutout and cutout > r:
                images = Cutout(8, 10 ,images, masks) # holes  , size

            if cutmix and cutmix > r:
                # generate mixed sample
                lam = np.random.beta(1., 1.)
                rand_index = torch.randperm(images.size()[0]).to(device)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                masks[:, bbx1:bbx2, bby1:bby2] = masks[rand_index, bbx1:bbx2, bby1:bby2]

            images, masks = images.to(device), masks.to(device).long()

            preds = model(images)
            loss = criterion(preds, masks)

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




class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')



import random
import matplotlib.pyplot as plt
import sys
def hide_patch(img):
    # get width and height of the image
    s = img.shape
    wd = s[0]
    ht = s[1]

    # possible grid size, 0 means no hiding
    grid_sizes=[0,16,32,44,56]

    # hiding probability
    hide_prob = 0.5
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if(grid_size > 0):
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 if(random.random() <=  hide_prob):
                       img[x:x_end,y:y_end,:]=0

    return img