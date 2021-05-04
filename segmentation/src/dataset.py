import os
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

import torch
import random
from torch.utils.data import Dataset
import albumentations as A


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class SegmentationDataset(Dataset):
    """COCO format"""

    def __init__(self, data_dir, cat_df, mode='train', num_cls=12, transform=None, augmix=None, prob=0.15):
        super().__init__()
        self.mode = mode
        self.num_cls = num_cls
        self.transform = transform
        self.coco = COCO(data_dir)
        self.ds_path = f'{os.sep}'.join(data_dir.split(os.sep)[:-1])
        self.category_names = list(cat_df.Categories)

        self.augmix = augmix
        self.prob = prob
        self.augmix_tfms = A.Compose([
            A.ShiftScaleRotate(scale_limit=0, rotate_limit=[-30, 30], shift_limit=0.3, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.GridDistortion(p=0.5)
        ])

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.ds_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if (self.mode in ('train', 'valid')):
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
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
            masks = masks.astype(np.float32)

            if self.augmix is not None:
                r = np.random.rand(1)
                if r[0] < self.prob:
                    images, masks = self.augmix_search(images, masks)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return {
                'image': images,
                'mask' : masks,
                'info' : image_infos['file_name']
            }

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return {
                'image': images,
                'info': image_infos['file_name']
            }

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


    def augmix_search(self, images, masks):
        temp_dict = {
            0: 4, 1: 5, 2: 6, 3: 10, 4: 11
        }
        num = [4, 5, 6, 10, 11]

        label = random.choice(num)
        idx = np.random.randint(len(self.augmix[label]))
        augmix_img = self.augmix[label][idx]

        augmix_mask = np.zeros((512, 512))
        augmix_mask[augmix_img[:, :, 0] != 0] = label  # augmix img가 있는 만큼 label로 mask를 채워줌

        transformed = self.augmix_tfms(image=augmix_img, mask=augmix_mask)
        augmix_img = transformed['image'].astype(np.uint8)
        augmix_mask = transformed['mask'].astype(np.uint8)

        images[augmix_img != 0] = augmix_img[augmix_img != 0]
        masks[augmix_mask != 0] = augmix_mask[augmix_mask != 0]

        return images, masks


class PseudoDataset(Dataset):
    def __init__(self, data_dir, pseudo_csv, transform=None):
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, pseudo_csv))
        self.tfms = A.Resize(512, 512, interpolation= cv2.INTER_NEAREST)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, 'data', img_path)
        mask = self.df.iloc[idx, 1]
        mask = np.array(list(map(int, mask.split())))
        mask = mask.reshape(256, 256)
        mask = self.tfms(image=mask)['image']


        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            images = transformed["image"]
            masks = transformed["mask"]

        return {
            'image': images,
            'mask': masks,
            'info': img_path
        }



if __name__ == "__main__":
    # from utils import make_cat_df
    # import matplotlib.pyplot as plt
    # from albumentations.pytorch import ToTensorV2
    #
    # data_path = '../input/data'
    # train_annot_path = os.path.join(data_path, "train.json")
    # augmix_path = os.path.join(data_path, '..', 'augmix.npy')
    # cat_df = make_cat_df(train_annot_path, debug=True)
    #
    # np_load_old = np.load
    # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # augmix_data = np.load(augmix_path)
    # augmix_data = augmix_data.item()
    #
    # tfms = ToTensorV2()
    # ds = SegmentationDataset(train_annot_path, cat_df, mode='train', num_cls=12, transform=tfms, augmix=augmix_data, prob=1.0)
    #
    # num_cls = 12
    # cmap = plt.get_cmap("rainbow")
    # colors = [cmap(i) for i in np.linspace(0, 1, num_cls+2)]
    # colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    # cls_colors = {k: colors[k] for k in range(num_cls+1)}
    #
    # for idx, sample in enumerate(iter(ds)):
    #     if idx == 30: break
    #
    #     image = sample['image']
    #     image = image.permute(1, 2, 0).detach().cpu().numpy()
    #     # mask = sample['mask'].detach().cpu().numpy()
    #
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("image", image.astype(np.uint8))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    cls_colors = {
        0: (255.0, 0.0, 127.5),
        1: (255, 29, 29),
        2: (236, 127, 43),
        3: (241, 229, 15),
        4: (124, 241, 15),
        5: (31, 62, 48),
        6: (52, 226, 220),
        7: (20, 96, 247),
        8: (8, 19, 62),
        9: (213, 192, 231),
        10: (75, 0, 135),
        11: (255, 0, 187)
    }

    from albumentations.pytorch import ToTensorV2
    ds = PseudoDataset(data_dir='../input', pseudo_csv='pseudo_result.csv', transform=ToTensorV2())
    for idx, sample in enumerate(iter(ds)):
        image = sample['image']
        image = image.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        mask = sample['mask'].detach().cpu().numpy()

        for i in range(1, 12):
            a_mask = (mask == i)
            cls_mask = np.zeros(image.shape).astype(np.float32)
            cls_mask[a_mask] = cls_colors[i]
            image[a_mask] = cv2.addWeighted(image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image.astype(np.uint8))
        cv2.waitKey()
        cv2.destroyAllWindows()
