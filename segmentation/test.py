import os
import cv2
import argparse
import numpy as np
import pandas as pd
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import cls_colors
from src.dataset import SegmentationDataset
from src.models import *

from importlib import import_module

def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    # Hyperparameter
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--postfix', default='effib3_unet_v1', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--num_workers', default='3', type=int)
    # Model_Type
    parser.add_argument('--model_type', default='FCN8s', type=str , help = 'ex) Unet , Deeplabv3, FCN8s ,SegNet')
    # Path 
    parser.add_argument('--data_path', default='/opt/ml/input/data', type=str)
    parser.add_argument('--ckpt', default='segnet_v1/best_mIoU.pth', type=str)
    args = parser.parse_args()
    print(args)

    # for reproducibility
    seed_everything(args.seed)


    test_annot = os.path.join(args.data_path, 'test.json')
    test_cat = make_cat_df(test_annot, debug=True)

    test_tfms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    size = 256
    resize = A.Resize(size, size)

    test_ds = SegmentationDataset(data_dir=test_annot, cat_df=test_cat, mode='test', transform=test_tfms)
    test_dl = DataLoader(dataset=test_ds,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_module = getattr(import_module("src.models"), args.model_type)
    model = model_module(num_classes = 12).to(device)

    model.load_state_dict(torch.load(os.path.join("./ckpts", args.ckpt)))

    model.eval()
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    cnt = 1
    with torch.no_grad():
        for step, sample in enumerate(test_dl):
            imgs = sample['image']
            file_names = sample['info']

            # inference (512 x 512)
            preds = model(imgs.to(device))
            oms = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()

            if args.debug:
                debug_path = os.path.join('.', 'debug', 'test', args.postfix)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)

                pred_masks = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
                for idx, file_name in enumerate(file_names):
                    pred_mask = pred_masks[idx]
                    ori_image = cv2.imread(os.path.join(args.data_path, file_name))
                    ori_image = ori_image.astype(np.float32)

                    for i in range(1, 12):
                        a_mask = (pred_mask == i)
                        cls_mask = np.zeros(ori_image.shape).astype(np.float32)
                        cls_mask[a_mask] = cls_colors[i]
                        ori_image[a_mask] = cv2.addWeighted(ori_image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

                    cv2.imwrite(os.path.join(debug_path, f"{cnt}.jpg"), ori_image)
                    cnt += 1

            # resize (256 x 256)
            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            for img, mask in zip(temp_images, oms):
                transformed = resize(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([file_name for file_name in file_names])
    print("End prediction.")

    print("Saving...")
    file_names = [y for x in file_name_list for y in x]
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    save_path = './submission'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'{args.postfix}.csv')
    submission.to_csv(save_dir, index=False)
    print(f"Saved in {save_dir}")
    print("All done.")

if __name__ == '__main__':
    main()