import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp


import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import torch
import torch.nn.functional as F



num_cls = 12
cmap = plt.get_cmap("rainbow")
# colors = [cmap(i) for i in np.linspace(0, 1, num_cls+2)]
# colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
# cls_colors = {k: colors[k] for k in range(num_cls+1)}
cls_colors = {
    0: (255.0, 0.0, 127.5),
    1 : (255, 29, 29),
    2 : (236, 127, 43),
    3 : (241, 229 , 15),
    4 : (124, 241, 15),
    5 : (31,62,48),
    6 : (52,226,220),
    7 : (20,96,247),
    8 : (8,19,62),
    9 : (213, 192, 231),
    10 : (75,0,135),
    11 : (255,0,187)
  }

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

def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])

def dense_crf(img, output_probs):
    MAX_ITER = 50
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


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

def get_mIoU(logits, labels, smooth=1e-10, num_classes=12):
    with torch.no_grad():
        #logits = F.softmax(logits, dim=1)
        logits = torch.argmax(logits, dim=1)
        labels = torch.argmax(labels, dim=1)
        logits = logits.contiguous().view(-1)
        labels = labels.contiguous().view(-1)

        iou_per_class = []
        # loop per pixel class / ignored class : 0(unlabeled)
        for clas in range(0, num_classes): 
            true_class = logits == clas
            true_label = labels == clas

            # no exist label in this loop
            if true_label.long().sum().item() == 0: 
                iou_per_class.append(np.nan)
            else:
                intersect = (true_class & true_label).sum().float().item()
                union = (true_class | true_label).sum().float().item()
                iou_per_class.append((intersect+smooth)/(union+smooth))

        return np.nanmean(iou_per_class)

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


def train_valid(epoch, model, trn_dl, val_dl, criterion, optimizer, scheduler, logger, device, debug=True):
    cnt = 1

    model.train()
    trn_mIoU = []
    trn_losses = []
    logger.info(f"\nTrain on Epoch {epoch+1}")
    with tqdm(trn_dl, total=len(trn_dl), unit='batch') as trn_bar:
        for batch, sample in enumerate(trn_bar):
            trn_bar.set_description(f"Train Epoch {epoch+1}")

            optimizer.zero_grad()
            images, masks = sample['image'], sample['mask']


            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            mIoU = get_mIoU(preds, masks, num_classes=12)
            
            trn_mIoU.append(mIoU)

            trn_losses.append(loss.item())
            if (batch+1) % (int(len(trn_dl)//10)) == 0:
                logger.info(f'Train Epoch {epoch+1} ==>  Batch [{str(batch+1).zfill(len(str(len(trn_dl))))}/{len(trn_dl)}]  |  Loss: {np.mean(trn_losses):.5f}  |  mIoU: {np.mean(trn_mIoU):.5f}')

            trn_bar.set_postfix(trn_loss=np.mean(trn_losses),
                                trn_mIoU=np.mean(trn_mIoU))


    model.eval()
    val_mIoU = []
    val_losses = []
    n_class = 12
    hist = np.zeros((n_class, n_class))
    logger.info(f"\nValid on Epoch {epoch+1}")
    with torch.no_grad():
        with tqdm(val_dl, total=len(val_dl), unit='batch') as val_bar:
            for batch, sample in enumerate(val_bar):
                val_bar.set_description(f"Valid Epoch {epoch+1}")

                images, masks = sample['image'], sample['mask']
                images, masks = images.to(device), masks.to(device)

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

                        # 256으로 학습할때만
                        ori_image = cv2.resize(ori_image, (256,256))
                        for i in range(1, 12):
                            a_mask = (pred_mask == i)
                            cls_mask = np.zeros(ori_image.shape).astype(np.float32)
                            cls_mask[a_mask] = cls_colors[i]
                            ori_image[a_mask] = cv2.addWeighted(ori_image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

                        cv2.imwrite(os.path.join(debug_path, f"{cnt}.jpg"), ori_image)
                        cnt += 1

                hist = add_hist(hist, torch.argmax(masks, dim=1).detach().cpu().numpy(), torch.argmax(preds, dim=1).detach().cpu().numpy(), n_class=n_class)

        acc, acc_cls, val_mIoU, fwavacc = label_accuracy_score(hist)
        print(f'Valid Epoch {epoch+1} ==>  Batch [{str(batch+1).zfill(len(str(len(val_dl))))}/{len(val_dl)}]  |  Loss: {np.mean(val_losses):.5f}  |  mIoU: {val_mIoU:.5f}')
        logger.info(
                        f'Valid Epoch {epoch+1} ==>  Batch [{str(batch+1).zfill(len(str(len(val_dl))))}/{len(val_dl)}]  |  Loss: {np.mean(val_losses):.5f}  |  mIoU: {val_mIoU:.5f}')
    return np.mean(trn_losses), np.mean(trn_mIoU), np.mean(val_losses), np.mean(val_mIoU)