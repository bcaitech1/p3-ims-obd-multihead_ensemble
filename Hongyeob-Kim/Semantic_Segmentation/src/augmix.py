import json
import random
import numpy as np
import torch
import cv2
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt


def imbalence_aug(tr_img, tr_mask, classdict):
    '''
    input : tr_img, tr_mask, classdict -> tr_img, tr_masks from train_loader
    classdict = {
      1: [],4: [],5: [],6: [],8: [],10: [],11: []
  }
  classdict-> dictionary with low number classes
    return : None -> save .npy file
    '''
    temp_images = tr_img
    temp_masks = tr_mask
    for i in range(len(temp_masks)):
        mask = temp_masks[i].numpy().astype(np.uint8)
        img = temp_images[i].permute([1, 2, 0]).numpy().astype(np.float32)
        mask[mask == 3] = 0
        mask[mask == 2] = 0
        mask[mask == 7] = 0
        mask[mask == 9] = 0
        class_type = np.unique(mask)
        if len(class_type) == 1:
            continue
        mask3d = np.dstack([mask] * 3)
        res = np.where(mask3d, 0, img)
        res1 = cv2.bitwise_and(img, img, mask=mask)
        for j in class_type:
            if j == 0:
                continue
            temp_mask = copy.deepcopy(mask)
            temp_mask[temp_mask != j] = 0
            temp = copy.deepcopy(res1)
            temp = cv2.bitwise_and(temp, temp, mask=temp_mask)
            classdict[i].append(temp)
    np.save('input/augmix.npy', classdict)


import random


def augmix_search(augmix_data, images, masks):
    '''
    input : augmix_data.item() -> data loaded from aug.npy
            images, masks -> from train_loader
    return images, masks -> augmixed
    '''
    tr_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
    ])

    # temp_dict = {
    #     0: 1, 1: 4, 2: 5, 3: 6, 4: 10, 5: 11
    # }
    # num = [1, 4, 5, 6, 10, 11]
    temp_dict = {
        0: 4, 1: 5, 2: 6, 3: 10, 4: 11
    }
    num = [4, 5, 6, 10, 11]

    temp, search, img = [], [], []
    augmix_img, augmix_mask = [], []
    for idx, i in enumerate(num):
        search.append(np.random.randint(len(augmix_data[i])))
        temp.append(augmix_data[i][search[idx]])
    # idx = random.sample([i for i in range(6)], 3)
    idx = random.sample([i for i in range(5)], 3)
    for i in range(3):
        img.append(temp[idx[i]])
    mask = np.zeros((3, 512, 512))

    for i in range(3):
        mask[i][img[i][:, :, 0] != 0] = temp_dict[idx[i]]
        #         img[i] = img[i] * 255
        img[i] = img[i].astype(np.uint8)

        transformed = tr_transform(image=img[i], mask=mask[i])
        src_imgs = transformed['image'].astype(np.uint8)
        src_mask = transformed['mask'].astype(np.uint8)

        rows, cols, _ = src_imgs.shape
        if np.sum(src_imgs != 0) // 3 < 3000:
            rs = np.random.uniform(low=1.5, high=2.0)
        elif np.sum(src_imgs != 0) // 3 < 500:
            rs = np.random.uniform(low=8.0, high=9.0)
        else:
            rs = np.random.uniform(low=0.3, high=0.65)

        h_list, w_list = np.where(src_mask != 0)
        # max_h, min_h = np.max(h_list), np.min(h_list)
        # max_w, min_w = np.max(w_list), np.min(w_list)
        # limit_h = min(512 - max_h, min_h)
        # limit_w = min(512 - max_w, min_w)

        # xs = np.random.uniform(0, limit_h)
        # ys = np.random.uniform(0, limit_w)
        xs = np.random.uniform(0, 20)
        ys = np.random.uniform(0, 20)

        M = np.float32([[1 * rs, 0, xs],
                        [0, 1 * rs, ys]])

        dst_imgs = cv2.warpAffine(src_imgs, M, (rows, cols))
        dst_mask = cv2.warpAffine(src_mask, M, (rows, cols))

        augmix_img.append(dst_imgs.astype(np.float32))
        augmix_mask.append(dst_mask.astype(np.float32))
    for i in range(3):
        images[i][augmix_img[i] != 0] = augmix_img[i][augmix_img[i] != 0]
        masks[i][augmix_mask[i] != 0] = augmix_mask[i][augmix_mask[i] != 0]
    return images, masks