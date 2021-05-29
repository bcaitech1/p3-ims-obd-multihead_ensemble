import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO

class RecycleTrashDataset(Dataset):
    """
    CoCo Dataset
    
    load_classes:loading 11classes 

    load_image:from coco loadimgs load images

    coco_label_to_label: convert label to target label

    load_annotations: from coco_dataset load annotations

    
    """

    def __init__(self, root_dir='/opt/ml/input/data', set_name='train', transform=None):
        super(RecycleTrashDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

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
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_idx], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

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

            return {'img': img, 'annot': annotation}
        return {'img': img, 'annot': annotation}

    def __len__(self):
        return len(self.image_ids)
