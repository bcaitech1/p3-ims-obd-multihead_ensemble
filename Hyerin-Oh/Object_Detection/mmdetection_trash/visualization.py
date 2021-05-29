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
from tqdm import tqdm
from pandas import DataFrame
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pkl', required=False, default=None)
parser.add_argument('--cfg', required=False, default='/opt/ml/code/mmdetection_trash/configs/trash/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_trash.py')
parser.add_argument('--work_dir', required=False, default='/opt/ml/code/mmdetection_trash/work_dirs/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_trash')
parser.add_argument('--pth_name', required=False, default='epoch_36.pth')
parser.add_argument('--set_name', required=False, default='val')
parser.add_argument('--csv', required=False, default='False')

args = parser.parse_args()

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

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


class RecycleTrashDataset(Dataset):
    """Some Information about RecycleTrashDataset"""
    def __init__(self, root_dir='/opt/ml/input/data', set_name='val', transform=None):
        super(RecycleTrashDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.transform = transform
        
        self.load_classes()
        
    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
        
        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key        
            
    def load_image(self, image_idx):
        image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img.astype(np.uint8)
    
    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]
    
    def label_to_coco_label(self, label):
        return self.coco_labels[label]
    
    def num_classes(self):
        return 11
    
    def load_annotations(self, image_idx):
        # get GT annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_idx], iscrowd=False)
        annotations = np.zeros((0, 5))
        
        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations
        
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)
            
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        
        return annotations

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annotation = self.load_annotations(idx)
        if self.transform:
            transformed = self.transform(image=img, bboxes=annotation)
            img = transformed['image']
            annotation = transformed['bboxes']
            
            return {'img' : img, 'annot' : annotation}
        return {'img' : img, 'annot' : annotation}

    def __len__(self):
        return len(self.image_ids)


PREFIX = '../../input/data/'

if args.pkl == None:
    # config file 들고오기
    cfg = Config.fromfile(args.cfg)

    # dataset 바꾸기
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = PREFIX
    cfg.data.train.ann_file = PREFIX + 'train.json'
    cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = PREFIX
    cfg.data.val.ann_file = PREFIX + 'val.json'
    cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = PREFIX
    cfg.data.test.ann_file = PREFIX + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.samples_per_gpu = 4

    cfg.seed=777
    cfg.gpu_ids = [0]
    cfg.work_dir = args.work_dir

    # cfg.model.roi_head.bbox_head.num_classes = 11

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, args.pth_name)

    if args.set_name == 'val':        
        dataset = build_dataset(cfg.data.val)
        val_dataset = RecycleTrashDataset()
        coco = COCO(cfg.data.val.ann_file)
    else:
        dataset = build_dataset(cfg.data.test)
        coco = COCO(cfg.data.test.ann_file)

    
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

else:
    if args.set_name == 'val':
        val_dataset = RecycleTrashDataset()
        coco = COCO('../../input/data/val.json')        
    else:
        coco = COCO('../../input/data/test.json')
    
    output = pd.read_pickle(args.pkl)


save_path = f'./{args.set_name}_visualization'
os.makedirs(save_path, exist_ok=True)

prediction_strings = []
img_list = []
file_names = []
annot_list = []
gt_annot_list = []

class_num = 11
for i, out in enumerate(output):
    prediction_string = ''
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    
    img_path = os.path.join(PREFIX, image_info['file_name'])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_list.append(img)
    bbox_list = []

    if args.set_name == 'val':
        sample = val_dataset.__getitem__(i)
        gt_annot_list.append(sample['annot'])

    for j in range(class_num):
        for o in out[j]:
            bbox_list.append([o[0], o[1], o[2], o[3], j])
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
    annot_list.append(bbox_list)
    prediction_strings.append(prediction_string)
    file_names.append(image_info['file_name'])


if args.csv == 'True':
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(args.csv, index=None)
    print(submission.head())


for img, fn, annot, gt_annot in tqdm(zip(img_list, file_names, annot_list, gt_annot_list)):
    output_img = img.copy()    
    file_name = fn.replace('/', '_')
    
    if args.set_name == 'val':
        gt_img = img.copy()
        for bbox, gt_bbox in zip(annot, gt_annot):
            bbox = np.int64(np.array(bbox))
            gt_bbox = np.int64(np.array(gt_bbox))

            label = int(bbox[-1])
            gt_label = int(gt_bbox[-1])
            
            xmin, ymin, xmax, ymax = bbox[:4]
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox[:4]
            
            color = COLORS[label]
            gt_color = COLORS[gt_label]
            
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(gt_img, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), gt_color, 2)    
            text_size = cv2.getTextSize(classes[label], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]    
            gt_text_size = cv2.getTextSize(classes[gt_label], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 2, ymin + text_size[1] + 6), color, -1)
            cv2.rectangle(gt_img, (gt_xmin, gt_ymin), (gt_xmin + text_size[0] + 2, gt_ymin + text_size[1] + 6), gt_color, -1)
            cv2.putText(
                output_img, classes[label],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_ITALIC, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                gt_img, classes[gt_label],
                (gt_xmin, gt_ymin + gt_text_size[1] + 4), cv2.FONT_ITALIC, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )
        
        result_img = np.hstack((gt_img, img, output_img))
    else:
        for bbox in annot:
            bbox = np.int64(np.array(bbox))
            label = int(bbox[-1])

            xmin, ymin, xmax, ymax = bbox[:4]

            color = COLORS[label]
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)        
            text_size = cv2.getTextSize(classes[label], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 2, ymin + text_size[1] + 6), color, -1)
            cv2.putText(
                output_img, classes[label],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_ITALIC, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )        
        result_img = np.hstack((img, output_img))

    cv2.imwrite(os.path.join(save_path, file_name), result_img)
