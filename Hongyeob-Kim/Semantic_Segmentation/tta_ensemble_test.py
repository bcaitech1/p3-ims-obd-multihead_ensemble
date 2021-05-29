import ttach as tta
import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
import multiprocessing as mp
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16

from src.utils import seed_everything
from src.utils import make_cat_df
from src.utils import cls_colors
from src.dataset import SegmentationDataset
from src.models.fcn8s import FCN8s

import torch.nn.functional as F


def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])

def dense_crf(img, output_probs):
    MAX_ITER = 50
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def main():
    parser = argparse.ArgumentParser(description="MultiHead Ensemble Team")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--postfix', required=True, type=str)
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--debug', default=0, type=int)

    args = parser.parse_args()
    print(args)

    model_types = args.model_type.split()
    model_ckpts = args.ckpt.split()

    # for reproducibility
    seed_everything(args.seed)

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

    test_ds = SegmentationDataset(data_dir=test_annot, cat_df=test_cat, mode='test', transform=test_tfms)
    test_dl = DataLoader(dataset=test_ds,
                         batch_size=32,
                         shuffle=False,
                         num_workers=3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import yaml
    from src.models.hrnet_seg import get_seg_model

    models = []
    tta_tfms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90([0, 90]),
        ]
    )

    for model_type, model_ckpt in zip(model_types, model_ckpts):
        print(model_type, model_ckpt, "\n")

        if model_type == 'hrnet_ocr':
            config_path = './src/configs/hrnet_seg_ocr.yaml'
            with open(config_path) as f:
                cfg = yaml.load(f)
                cfg['MODEL']['PRETRAINED'] = ''
            model = get_seg_model(cfg, test=True)

        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join("./ckpts", model_ckpt)))
        tta_model = tta.SegmentationTTAWrapper(model, tta_tfms, merge_mode='mean')
        tta_model.eval()
        models.append(tta_model)




    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    cnt = 1
    tar_size = 512
    print("Start Inference.")
    torch.multiprocessing.set_start_method('spawn')
    with torch.no_grad():
        for step, sample in tqdm(enumerate(test_dl), total=len(test_dl)):
            imgs = sample['image']
            file_names = sample['info']

            # inference (512 x 512)
            final_probs = 0
            for model in models:
                preds = model(imgs.to(device))
                ph, pw = preds.size(2), preds.size(3)
                if ph != tar_size or pw != tar_size:
                    preds = F.interpolate(input=preds, size=(
                        tar_size, tar_size), mode='bilinear', align_corners=True)
                probs = F.softmax(preds, dim=1).detach().cpu().numpy()

                pool = mp.Pool(mp.cpu_count())
                images = imgs.detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                if images.shape[1] != tar_size or images.shape[2] != tar_size:
                    images = np.stack([resize(image=im)['image'] for im in images], axis=0)

                final_probs += np.array(pool.map(dense_crf_wrapper, zip(images, probs))) / len(models)
                pool.close()

            oms = np.argmax(final_probs.squeeze(), axis=1)

            if args.debug:
                debug_path = os.path.join('.', 'debug', 'test', args.postfix)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)

                for idx, file_name in enumerate(file_names):
                    pred_mask = oms[idx]
                    ph, pw = pred_mask.shape
                    ori_image = cv2.imread(os.path.join('.', 'input', 'data', file_name))
                    if ori_image.shape[0] != ph or ori_image.shape[1] != pw:
                        ori_image = cv2.resize(ori_image, (ph, pw))
                    ori_image = ori_image.astype(np.float32)

                    for i in range(1, 12):
                        a_mask = (pred_mask == i)
                        cls_mask = np.zeros(ori_image.shape).astype(np.float32)
                        cls_mask[a_mask] = list(cls_colors[i])[-1::-1]
                        ori_image[a_mask] = cv2.addWeighted(ori_image[a_mask], 0.2, cls_mask[a_mask], 0.8, gamma=0.0)

                    label = np.unique(pred_mask)
                    cv2.imwrite(os.path.join(debug_path, f"{file_name.replace(os.sep, '_')}_{label}.jpg"), ori_image)
                    cnt += 1

            # resize (256 x 256)
            temp_mask = []
            temp_images = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
            for img, mask in zip(temp_images, oms):
                if mask.shape[0] != 256 or mask.shape[1] != 256:
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
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    save_path = './submission'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'{args.postfix}.csv')
    submission.to_csv(save_dir, index=False)
    print("All done.")

if __name__ == '__main__':
    main()
