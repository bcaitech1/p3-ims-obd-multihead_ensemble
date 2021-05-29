import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test, train_detector
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
cfg = Config.fromfile('./configs/swin/swin_large.py')

PREFIX = '../input/data/'


# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = [PREFIX + 'train_all.json', PREFIX+'fixed_test.json']

cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'

cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'

cfg.data.samples_per_gpu = 4

cfg.seed = 777
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/swin_s'


cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

# checkpoint path
# checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
model = build_detector(cfg.model)
datasets = [build_dataset(cfg.data.train)]
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
