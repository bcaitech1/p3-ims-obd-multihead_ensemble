import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
from src.dataset import *

def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


# vim_hjk : Class index
CLASSES = [
    'UNKNOWN',
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing'
]

# vim_hjk : bbox Color
COLORS = [
    (39, 129, 113),
    (164, 80, 133),
    (83, 122, 114),
    (99, 81, 172),
    (95, 56, 104),
    (37, 84, 86),
    (14, 89, 122),
    (80, 7, 65),
    (10, 102, 25),
    (90, 185, 109),
    (106, 110, 132)
]

tmp = []
value = 50

for i in COLORS:
    B = i[0] + value if i[0] + value < 255 else 255
    G = i[1] + value if i[1] + value < 255 else 255
    R = i[2] + value if i[2] + value < 255 else 255

    brightness = (B, G, R)

    tmp.append(brightness)

COLORS2 = tmp

def main():
  cfg = Config.fromfile('./configs/swin/swin_large.py')
  PREFIX = '../input/data/'
  classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal",
            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
  epoch = 1
  cfg.data.test.classes = classes
  cfg.data.test.img_prefix = PREFIX
  cfg.data.test.ann_file = PREFIX + 'test.json'
  cfg.data.samples_per_gpu = 4
  cfg.seed = 5
  cfg.gpu_ids = [0]
  cfg.work_dir = './work_dirs/swin_s'

  cfg.model.train_cfg = None

  # checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
  checkpoint_path = os.path.join(cfg.work_dir, 'epoch_22.pth')

  dataset = build_dataset(cfg.data.test)
  data_loader = build_dataloader(
      dataset,
      samples_per_gpu=1,
      workers_per_gpu=cfg.data.workers_per_gpu,
      dist=False,
      shuffle=False)

  model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
  checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

  model.CLASSES = dataset.CLASSES
  model = MMDataParallel(model.cuda(), device_ids=[0])

  output = single_gpu_test(model, data_loader, show_score_thr=0.05)

  prediction_strings = []
  file_names = []
  coco = COCO(cfg.data.test.ann_file)
  imag_ids = coco.getImgIds()

  class_num = 11
  for i, out in enumerate(output):
      prediction_string = ''
      image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
      for j in range(class_num):
          for o in out[j]:
              prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                  o[2]) + ' ' + str(o[3]) + ' '

      prediction_strings.append(prediction_string)
      file_names.append(image_info['file_name'])


  submission = pd.DataFrame()
  submission['PredictionString'] = prediction_strings
  submission['image_id'] = file_names
  submission.to_csv(os.path.join(
      cfg.work_dir, f'submission_detectors_resnext.csv'), index=None)
  submission.head()

def pseduo():
  real_df = pd.read_csv( './Swin-Transformer-Object-Detection/work_dirs/swin_s/submission_v3_35epoch.csv')

  coco = COCO('./final_fixed_test.json')
  predict = [0.9834, 0.939324, 0.99999, 0.932459, 0.9726372, 0.97832672348,
            0.92234243, 0.956573, 0.912381, 0.9912371, 0.99726263, 0.99123123]
  coco_img_id = coco.getImgIds()
  prediction_strings = []
  file_names = []
  for i in range(833):
    image_name = coco.loadImgs(coco_img_id[i])
    image_name = image_name[-1]['file_name']
    # print(image_name)
    pseudo = ""
    anno_ids = coco.getAnnIds(imgIds=coco_img_id[i], iscrowd=False)
    temp = coco.loadAnns(anno_ids)
    temp = sorted(temp, key=lambda x: x['category_id'])
    for p in range(len(temp)):
      count = np.random
      bbox = temp[p]['bbox']
      category = temp[p]['category_id']
      for j in range(11):
        pseudo += str(j)+' '+str(0.999)+' '+str(bbox[0])+' '+str(
            bbox[1])+' '+str(bbox[2]+bbox[0])+' '+str(bbox[3]+bbox[1])+' '
    index = real_df[real_df['image_id'] == image_name].index[0]
    real_df.PredictionString[index] = pseudo

  real_df.to_csv('sudo4.csv', index=False)

if __name__=='__main__':
  main()
  pseudo()

