import os
import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset

from src.utils import * 

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class SegmentationDataset(Dataset):
    """COCO format"""

    def __init__(self, data_dir, cat_df, mode='train', transform=None , augmix = None , prob = 0):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.ds_path = f'{os.sep}'.join(data_dir.split(os.sep)[:-1])
        self.category_names = list(cat_df.Categories)
        self.augmix = augmix
        self.prob = prob

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.ds_path, image_infos['file_name']))
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.uint8)

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

            ## transform 들어가기 전에 해주자.
            # img, mask 다 np , 512x512
            if self.augmix is not None :
              r = np.random.rand(1)
              if r < self.prob :
                  images , masks = self.augmix_search(images , masks)
            
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


if __name__ == "__main__":
  from src.utils import make_cat_df
  import matplotlib.pyplot as plt
  from albumentations.pytorch import ToTensorV2
  import numpy as np

  # 본인에 맞게 경로 설정
  np_load_old = np.load
  np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
  augmix_data = np.load('/content/drive/MyDrive/code/augmix.npy')

  data_path = '/content/input'
  train_annot_path = os.path.join(data_path, "train.json")
  cat_df = make_cat_df(train_annot_path, debug=True)
  tfms = ToTensorV2()
  ds = SegmentationDataset(train_annot_path, cat_df, mode='train', transform=tfms, augmix = augmix_data.item() , prob = 1)

  num_cls = 12
  cmap = plt.get_cmap("rainbow")
  colors = [cmap(i) for i in np.linspace(0, 1, num_cls+2)]
  colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
  cls_colors = {k: colors[k] for k in range(num_cls+1)}

  for idx, sample in enumerate(iter(ds)):
    image = sample['image']
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    mask = sample['mask']
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(image.astype(np.uint8))
    break
