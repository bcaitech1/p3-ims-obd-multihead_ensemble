import os
import cv2
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import random
import albumentations as A 
import cv2
import pandas as pd
def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class SegmentationDataset(Dataset):

    def __init__(self, data_dir, cat_df, mode='train', psudo_label='./presudo_result.csv', num_cls=12, transform=None, augmix=None, image_list=None,args=None):
        super().__init__()
        self.mode = mode
        self.num_cls = num_cls
        self.transform = transform
        self.coco = COCO(data_dir)
        self.ds_path = f'{os.sep}'.join(data_dir.split(os.sep)[:-1])
        self.category_names = list(cat_df.Categories)
        self.augmix=augmix
        self.coco_test=COCO('./input/data/test.json')
        self.image_list=image_list
        self.psudo_label_tfms=A.Resize(512,512, interpolation=cv2.INTER_NEAREST)
        self.df=pd.read_csv(psudo_label)
        self.args=args

    def __getitem__(self, index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        images = cv2.imread(os.path.join(
            self.ds_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
#############################test 배경넣기 
        prob=np.random.rand(1)
        assert images.shape[2]==3,'image error'
        if (self.mode in ('train', 'val')):
            img_path = self.df.iloc[(index % len(self.df)), 0]
            mask = self.df.iloc[(index % len(self.df)), 1]
            mask = np.array(list(map(int, mask.split())))
            mask = mask.reshape(256, 256)
            mask = self.psudo_label_tfms(image=mask)['image']
            psudo_label_img = cv2.imread(os.path.join('./input/data',img_path))
            psudo_label_img = cv2.cvtColor(psudo_label_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            # Load the categories in a vairable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            # masks: size 가 (height*width)인 2d
            # 각각의 pixel 값에는 'category id +1 " 할당
            # Backgroud =0
            masks = np.zeros((image_infos['height'], image_infos['width']))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(
                    anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)
            if self.augmix:
              r=np.random.rand(1) 
              if r<0.5:
                images,masks =self.augmix_search(images,masks)
            prob=np.random.rand(1)
            if self.mode=='train':
              if self.args.use_only_test:
                images=psudo_label_img
                masks=mask.astype(np.float32)
              if prob>=0.3:
                images=psudo_label_img
                masks=mask.astype(np.float32)
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed['image']
                masks = transformed['mask']
            return {
                'image': images,
                'mask': masks,
                'info':image_infos['file_name']
            }

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed['image']
            return {
                'image': images,
                'info':image_infos['file_name']
            }

    def __len__(self):
        return len(self.coco.getImgIds())

    def augmix_search(self, images, masks):
      # image 3, 512, 512 ,mask: 512, 512 (둘 다 numpy)
      tfms=A.Compose([
        A.GridDistortion(p=0.3),
        A.Rotate(limit=60,p=1.0),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5)
      ]
      )
      temp_dict = {
              0: 4, 1: 5, 2: 6, 3: 10, 4: 11
      }
      num = [4, 5, 6, 10, 11]

      label = random.choice(num)  # ex) 4
      idx = np.random.randint(len(self.augmix[label]))
      augmix_img = self.augmix[label][idx]

      augmix_mask = np.zeros((512, 512))
      # augmix img가 있는 만큼 label로 mask를 채워줌
      augmix_mask[augmix_img[:, :, 0] != 0] = label
      ################################################## 새로 추가한 transform을 적용해보자 
      transformed=tfms(image=augmix_img, mask=augmix_mask)
      augmix_img = transformed['image']
      augmix_mask = transformed['mask']
      ####################################################
      images[augmix_img != 0] = augmix_img[augmix_img != 0]
      masks[augmix_mask != 0] = augmix_mask[augmix_mask != 0]

      return images, masks

if __name__ == '__main__':
    from utils import make_cat_df
    data_path = '../input/data'
    train_annot_path = os.path.join(data_path, 'train.json')
    cat_df = make_cat_df(train_annot_path, debug=True)
    df = SegmentationDataset(train_annot_path, cat_df,
                             mode='train', transform=None)

    for i in range(10):
        sample = next(iter(ds))
        image = sample['image']
        mask = sample['mask']
        image_info = sample['image_info']
        print(image.shape)
        print(mask.shape)
        print(image_info)
