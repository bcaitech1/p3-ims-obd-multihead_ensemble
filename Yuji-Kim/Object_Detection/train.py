from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./configs/trash/detectors_resnext.py')

PREFIX = '../../input/data/'

cfg.data.samples_per_gpu = 8

cfg.seed=5
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/detectors_v6'


model = build_detector(cfg.model)
datasets = [build_dataset(cfg.data.train)]
train_detector(model, datasets[0], cfg, distributed=False, validate=True)