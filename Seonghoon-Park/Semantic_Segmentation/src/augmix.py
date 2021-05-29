import json
import random
import numpy as np
import torch
import cv2
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
        img = temp_images[i].permute([1, 2, 0]).numpy().astype(np.float64)
        mask[mask == 3] = 0
        mask[mask == 2] = 0
        mask[mask == 7] = 0
        mask[mask == 9] = 0
        class_type = np.unique(mask)
        if len(class_type) == 1:
            continue
        mask3d = np.dstack([mask]*3)
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
    np.save('augmix.npy', classdict)


def augmix_search(augmix_data, images, masks):
    '''
    input : augmix_data.item() -> data loaded from aug.npy
            images, masks -> from train_loader

    return images, masks -> augmixed
    '''
    
    masks = masks.argmax(dim=1).float()
    
    tr_transform = A.Compose([
        A.Resize(256,256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=30, p=0.5),
        ToTensorV2()
    ])
    temp_dict = {
        0: 1, 1: 4, 2: 5, 3: 6, 4: 8, 5: 10, 6: 11
    }
    num = [1, 4, 5, 6, 8, 10, 11]
    temp, search, img = [], [], []
    augmix_img, augmix_mask = [], []
    for idx, i in enumerate(num):
        search.append(np.random.randint(len(augmix_data[i])))
        temp.append(augmix_data[i][search[idx]])
    idx = random.sample([i for i in range(7)], 3)
    for i in range(3):
        img.append(temp[idx[i]])
    mask = np.zeros((3, 512, 512))
    for i in range(3):
        mask[i][img[i][:, :, 0] != 0] = temp_dict[idx[i]]
        transformed = tr_transform(image=img[i], mask=mask[i])
        augmix_img.append(transformed['image'].float())
        augmix_mask.append(transformed['mask'].float())

    for i in range(3):
        images[i][augmix_img[i] != 0] = augmix_img[i][augmix_img[i] != 0]
        masks[i][augmix_mask[i] != 0] = augmix_mask[i][augmix_mask[i] != 0]
        
    return images, masks
