import os
import cv2
import numpy as np
import random
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset


dataset_path = '/opt/ml/input/data'
category_names = ['Background', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                    'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"

class RecycleTrashDataset(Dataset):
    """Some Information about RecycleTrashDataset"""
    def __init__(self, data_dir, mode='train', transform=None, augmix=None, augmix_prob=0):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.augmix = augmix
        self.augmix_prob = augmix_prob
        self.coco = COCO(os.path.join(dataset_path, data_dir))
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
            masks = masks.astype(np.float32)

            # one-hot
            target = np.unique(masks).astype(int)
            one_hot_label = np.sum(np.eye(12)[target], axis=0)

            # img, mask 다 np , 512x512
            if self.augmix is not None :
              r = np.random.rand(1)
              if r < self.augmix_prob :
                  images , masks = self.augmix_search(images , masks)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, torch.Tensor(one_hot_label), image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


    def augmix_search(self, images, masks):
        # image 3, 512, 512 ,mask: 512, 512 (둘 다 numpy)
        temp_dict = {
        0: 4, 1: 5, 2: 6, 3: 10, 4: 11
        }
        num = [ 4, 5, 6, 10, 11]
        
        label = random.choice(num)  # ex) 4
        idx = np.random.randint(len(self.augmix[label]))
        augmix_img = self.augmix[label][idx]
        
        augmix_mask = np.zeros((512, 512))
        augmix_mask[augmix_img[:, :, 0] != 0] = label     # augmix img가 있는 만큼 label로 mask를 채워줌
        
        images[augmix_img != 0] = augmix_img[augmix_img != 0]
        masks[augmix_mask!= 0] = augmix_mask[augmix_mask != 0]
        
        return images, masks


import pandas as pd
import albumentations as A

class PseudoDataset(Dataset):
    def __init__(self, data_dir='/opt/ml/input/data/', pseudo_csv='presudo_result.csv', transform=None):
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, pseudo_csv))
        self.tfms = A.Resize(512, 512, interpolation= cv2.INTER_NEAREST)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, img_path)
        mask = self.df.iloc[idx, 1]
        mask = np.array(list(map(int, mask.split())))
        mask = mask.reshape(256, 256)
        mask = self.tfms(image=mask)['image']

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # one-hot
        target = np.unique(mask).astype(int)
        one_hot_label = np.sum(np.eye(12)[target], axis=0)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            images = transformed["image"]
            masks = transformed["mask"]

        return images, masks, torch.Tensor(one_hot_label), img_path


class AllDataset(Dataset):
    def __init__(self, data_dir, mode='train', augmix_prob=0.15, augmix=None, transform=None):
        self.presudo_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.461, 0.440, 0.419), std=(0.211, 0.208, 0.216)),
            A.pytorch.transforms.ToTensorV2()
        ])
        self.presudo_ds = PseudoDataset(transform=self.presudo_transform)
        self.seg_ds = RecycleTrashDataset(data_dir=data_dir, mode=mode, augmix=augmix, augmix_prob=augmix_prob, transform=transform)
        

    def __len__(self):
        return len(self.presudo_ds) + len(self.seg_ds)


    def __getitem__(self, idx):
        if idx >= len(self.seg_ds):
            return self.presudo_ds.__getitem__(idx - len(self.seg_ds))
        else:
            return self.seg_ds.__getitem__(idx)