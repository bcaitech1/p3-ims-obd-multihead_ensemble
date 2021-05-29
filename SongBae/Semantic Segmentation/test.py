import os
import cv2
import argparse
import numpy as np
import pandas as pd
import albumentations as A
import multiprocessing as mp 
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from src.models import UnetPlusPlus
from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import cls_colors,dense_crf,dense_crf_wrapper
from src.dataset import SegmentationDataset
from src.models import *
import copy
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--postfix', default='effib3_unet_v1', type=str)
    parser.add_argument('--ckpt', default='unet/best_loss.pth', type=str)
    parser.add_argument('--model_type', default='unet', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--crf',default=0,type=int)

    args = parser.parse_args()
    print(args)

    # for reproducibility
    seed_everything(args)

    main_path = '.'
    data_path = os.path.join(main_path, 'input', 'data')
    test_annot = os.path.join(data_path, 'test.json')
    test_cat = make_cat_df(test_annot, debug=True)

    test_tfms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    size = 256
    resize = A.Resize(size, size)

    test_ds = SegmentationDataset(
        data_dir=test_annot, cat_df=test_cat, mode='test', transform=test_tfms)
    test_dl = DataLoader(dataset=test_ds,
                         batch_size=32,
                         shuffle=False,
                         num_workers=3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == "fcn8s":
        backbone = vgg16(pretrained=False)
        model = FCN8s(backbone)
    elif args.model_type == 'unet':
        model = smp.Unet(
            # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=3,
            # model output channels (number of classes in your dataset)
            classes=12,
        )
    elif args.model_type == 'deeplabv3':
      model = smp.DeepLabV3(
      encoder_name='efficientnet-b3',
      encoder_weights='imagenet',
      in_channels=3,
      classes=12
      )
      
    elif args.model_type == 'hrnet_ocr':
      import yaml
      from src.models import get_seg_model
      config_path = './src/configs/hrnet_seg.yaml'
      with open(config_path) as f:
        cfg = yaml.load(f)
      model = get_seg_model(cfg)
      model1=get_seg_model(cfg)
      model2=UnetPlusPlus()
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join("./ckpts",'use_only_test/pseudo_best.pth')))
    model.eval()
    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)
    cnt = 1
    tar_size = 512
    with torch.no_grad():
        for step, sample in tqdm(enumerate(test_dl)):
            imgs = sample['image']
            file_names = sample['info']

            # inference (512 x 512)
            pred1 = model(imgs.to(device))
            pred2=model1(imgs.to(device))
            pred3=model2(imgs.to(device))
            pred3=F.interpolate(input=pred3, size=(512,512),mode='bilinear',align_corners=True)
            if isinstance(pred1, list):
                pred1 = pred1[0]
                ph, pw = pred1.size(2), pred1.size(3)
                if ph != tar_size or pw != tar_size:
                    pred1 = F.interpolate(input=pred1, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
            if isinstance(pred2, list):
                pred2 = pred2[0]
                ph, pw = pred2.size(2), pred2.size(3)
                if ph != tar_size or pw != tar_size:
                    pred2 = F.interpolate(input=pred2, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
            #preds=pred1+pred2+pred3
            preds=pred2+pred1
            if args.crf:
              preds=F.softmax(preds,dim=1).data.cpu().numpy()
              pool=mp.Pool(mp.cpu_count())
              imgs=imgs.data.cpu().numpy().astype(np.uint8).transpose(0,2,3,1)
              preds=pool.map(dense_crf_wrapper,zip(imgs,preds))
              pool.close()
              oms=np.argmax(preds,axis=1)
            else:
              oms = torch.argmax(preds.squeeze(), dim=1).detach().cpu().numpy()
          
            if args.debug:
                debug_path = os.path.join('.', 'debug', 'test', args.postfix)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)

                  
                for idx, file_name in enumerate(file_names):
                    pred_mask = pred_masks[idx]
                    ori_image = cv2.imread(os.path.join(
                        '.', 'input', 'data', file_name))
                    ori_image = ori_image.astype(np.float32)

                    for i in range(1, 12):
                        a_mask = (pred_mask == i)
                        cls_mask = np.zeros(ori_image.shape).astype(np.float32)
                        cls_mask[a_mask] = cls_colors[i]
                        ori_image[a_mask] = cv2.addWeighted(
                            ori_image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

                    cv2.imwrite(os.path.join(
                        debug_path, f"{file_name}{np.unique(cls_mask)}.jpg"), ori_image)
                    cnt += 1

            # resize (256 x 256)
            temp_mask = []
            if args.crf:temp_images=imgs
            else:temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()

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
    submission = pd.read_csv(
        './sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(
                str(e) for e in string.tolist())},
            ignore_index=True)

    save_path = './submission'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'{args.postfix}.csv')
    submission.to_csv(save_dir, index=False)
    print("All done.")


if __name__ == '__main__':
    main()
