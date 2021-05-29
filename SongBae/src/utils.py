import tqdm
import warnings
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from packaging import version
import torch
import matplotlib.pyplot as plt
import cv2
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import wandb
from collections import OrderedDict
import torch.nn.functional as F
import pandas as pd 
num_cls = 12
cmap = plt.get_cmap("rainbow")
colors = [cmap(i) for i in np.linspace(0, 1, num_cls+2)]
colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
cls_colors = cls_colors = {
    0: (255.0, 0.0, 127.5),
    1: (255, 29, 29),
    2: (236, 127, 43),
    3: (241, 229, 15),
    4: (124, 241, 15),
    5: (31, 62, 48),
    6: (52, 226, 220),
    7: (20, 96, 247),
    8: (8, 19, 62),
    9: (213, 192, 231),
    10: (75, 0, 135),
    11: (255, 0, 187)
}

def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])


def dense_crf(img, output_probs):
    MAX_ITER = 50
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 3

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(
        sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def imbalence_aug(tr_img, tr_mask, classdict):
    temp_images = tr_img
    temp_masks = tr_mask
    for i in range(len(temp_masks)):
        mask = temp_masks[i].numpy().astype(np.uint8)
        img = temp_images[i].permute([1, 2, 0]).numpy().astype(np.float64)
        mask[mask == 3] = 0
        mask[mask == 2] = 0
        mask[mask == 7] = 0
        mask[mask == 9] = 0
        class_type = np.unique(mask)
        if len(class_type) == 1:
            break
        mask3d = np.dstack([mask]*3)
        res = np.where(mask3d, 0, img)
        res1 = cv2.bitwise_and(img, img, mask=mask)
        for j in class_type:
            if j == 0:
                continue
            temp_mask = copy.deepcopy(mask)
            temp_mask[temp_mask != j] = 0
            temp = copy.deepcopy(res1)
            temp = cv2.bitwise_and(temp, temp, mask=temp_mask)
            classdict[i].append(temp)
    np.save('augmix.npy', classdict)


def seed_everything(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_cat_df(train_annot_path, debug=False):
    # Read annotations
    with open(train_annot_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    nr_cats = len(categories)
    nr_annotations = len(anns)
    nr_images = len(imgs)

    # Load categories and super categories
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1

    if debug:
        print('Number of super categories:', nr_super_cats)
        print('Number of categories:', nr_cats)
        print('Number of annotations:', nr_annotations)
        print('Number of images:', nr_images)
        print()

    cat_histogram = np.zeros(nr_cats, dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']] += 1

    cat_df = pd.DataFrame(
        {'Categories': cat_names, 'Number of annotations': cat_histogram})
    cat_df = cat_df.sort_values('Number of annotations', 0, False)
    sorted_df = pd.DataFrame(["Backgroud"], columns=["Categories"])
    sorted_df = sorted_df.append(cat_df.sort_index(), ignore_index=True)
    return sorted_df

 # 무슨 뜻일까?


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class*label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist)/hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist)/(hist.sum(axis=1)+hist.sum(axis=0)-np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1)/hist.sum()
    fwavacc = (freq[freq > 0]*iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def save_model(model, version, save_type='loss'):
    save_path = os.path.join(f'./ckpts/{version}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_dir = os.path.join(save_path, f'best_{save_type}.pth')
    torch.save(model.state_dict(), save_dir)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


class_labels = {
    0: 'Background',
    1: 'UNKNOWN',
    2: 'General trash',
    3: 'Paper',
    4: 'Paper pack',
    5: 'Metal',
    6: 'Glass',
    7: 'Plastic',
    8: 'Styrofoam',
    9: 'Plastic bag',
    10: 'Battery',
    11: 'Clothing'
}



def train_valid(epochs, model, train_dl, valid_dl, criterion, optimizer, logger, device, scheduler, args, augmix_data, use_augmix):
    now_dl = None
    best_score=0.0
    best_loss=100
    wandb.init()
    wandb.config.update(args)
    wandb.watch(model)
    for epoch in range(epochs):
      example_images=[]
      for phase in ['train', 'valid']:
          if args.use_only_test and phase=='valid':
            continue
          run_mIoU, run_loss = [], []
          if phase == 'train':
              model.train()
              now_dl = train_dl
          else:
              model.eval()
              now_dl = valid_dl
          logger.info(f'\n{phase} on Epoch {epoch+1}')
          with torch.set_grad_enabled(phase == 'train'):
              with tqdm(now_dl, total=len(now_dl), unit='batch') as now_bar:
                  for batch, sample in enumerate(now_bar):
                      now_bar.set_description(f'{phase} Epoch {epoch}')
                      optimizer.zero_grad()

                      # 3개의 이미지에 더해주자
                      images, masks = sample['image'], sample['mask']
                      file_names=sample['info']
                      images, masks = images.to(device), masks.to(device).long()
                      ####################################################
                      preds = model(images)
                      if isinstance(preds, OrderedDict):
                        preds = preds['out']
                      elif isinstance(preds, list):
                        for i in range(len(preds)):
                          pred = preds[i]
                          ph, pw = pred.size(2), pred.size(3)
                          h, w = masks.size(1), masks.size(2)
                          if ph != h or pw != w:
                              pred = F.interpolate(input=pred, size=(
                            h, w), mode='bilinear', align_corners=True)
                          preds[i] = pred
                      if isinstance(preds, list):
                        loss=0
                        for i in range(len(preds)):
                          loss+=criterion(preds[i],masks)*(1/2**i)
                        preds=preds[0]
                      else :
                        loss=criterion(preds,masks)
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()
                      if phase=='valid':
                        input_np = cv2.imread(os.path.join('.', 'input', 'data', file_names[0]))
                        example_images.append(wandb.Image(input_np, masks={
                            "predictions": {
                                "mask_data": preds.argmax(1)[0].detach().cpu().numpy(),
                                "class_labels": class_labels
                            },
                            "ground-truth": {
                                "mask_data": masks[0].detach().cpu().numpy(),
                                "class_labels": class_labels
                            }
                        }))
                      preds = torch.argmax(
                          preds.squeeze(), dim=1).detach().cpu().numpy()
                      mIoU = label_accuracy_score(
                          masks.detach().cpu().numpy(), preds, n_class=12)[2]
                      run_mIoU.append(mIoU)
                      run_loss.append(loss.item())

                      if (batch+1) % (int(len(now_dl)//10)) == 0:
                          logger.info(
                              f'{phase} Epoch {epoch+1} ==> Batch [{str(batch+1).zfill(len(str(len(now_dl))))}/{len(now_dl)}] |  Loss: {np.mean(run_loss):.5f}  |  mIoU: {np.mean(run_mIoU):.5f}')
                      now_bar.set_postfix(run_loss=np.mean(run_loss),
                                          run_mIoU=np.mean(run_mIoU))
                  if phase == 'valid':
                      scheduler.step(np.mean(run_loss))
                  if phase == 'valid' and best_score < np.mean(run_mIoU):
                      best_score = np.mean(run_mIoU)
                      save_model(model, args.version)
                      print('best_model_saved')
                  if phase=='valid' and best_loss>np.mean(run_loss):
                    best_loss=np.mean(run_loss)
                    save_model(model,'best_loss_model.pth')
                    print('best_model_loss saved')
                  if phase=='train' and args.use_only_test:
                    save_model(model,'use_only_test.pth')
                  if phase=='valid':
                    wandb.log({
                      'Example Image': example_images,
                    'Valid_Loss': np.mean(run_loss),
                    'Valid_MIoU':np.mean(run_mIoU)
                    })
                  if phase=='train':
                    wandb.log({
                    'Learning rate': get_learning_rate(optimizer)[0],
                    'Train_Loss': np.mean(run_loss),
                    'Train_MIoU':np.mean(run_mIoU)
                    })


tfms_to_small = A.Compose([
    A.Resize(256, 256),
    A.PadIfNeeded(512, 512, border_mode=0),
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=0.5)
])
tfms_to_big = A.Compose([
    A.CropNonEmptyMaskIfExists(300, 300),
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),
    A.Resize(512, 512)
])
tfms = A.Compose([
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=0.5),
])
warnings.filterwarnings('ignore')
classdict = {
    1: [],
    4: [],
    5: [],
    6: [],
    10: [],
    11: []
}

'''
making npy.file for new augmentation

input: basic image

output: augmented image seperated by class

'''
for idx, (imgs, masks, image_infos) in enumerate(train_loader):
    if idx == 200:
        break
    image_infos = image_infos[0]
    temp_images = imgs
    temp_masks = masks
    for i in range(len(temp_masks)):

        mask = temp_masks[i].numpy().astype(np.uint8)
        img = temp_images[i].permute([1, 2, 0]).numpy().astype(np.uint8)
        mask[mask == 3] = 0
        mask[mask == 2] = 0
        mask[mask == 7] = 0
        mask[mask == 8] = 0
        mask[mask == 9] = 0
        mask[mask == 1] = 0
        class_type = np.unique(mask)
        if len(class_type) == 1:
            continue
        mask3d = np.dstack([mask]*3)
        res = np.where(mask3d, 0, img)
        res1 = cv2.bitwise_and(img, img, mask=mask)
        for j in class_type:
          if j == 0:
              continue
          temp_mask = copy.deepcopy(mask)
          temp_mask[temp_mask != j] = 0
          if (np.sum(temp_mask != 0)) < 400:
              continue
          temp = copy.deepcopy(res1)
          temp = cv2.bitwise_and(temp, temp, mask=temp_mask)
          if np.sum(temp != 0) > 20000:
            transformed = tfms_to_small(image=temp, mask=temp_mask)
            mask = transformed['mask']
            temp = transformed['image']
          elif np.sum(temp != 0) < 2000:
            transformed = tfms_to_big(image=temp, mask=temp_mask)
            mask = transformed['mask']
            temp = transformed['image']
          else:
            transformed = tfms(image=temp, mask=temp_mask)
            mask = transformed['mask']
            temp = transformed['image']
            fig, axes = plt.subplots(1, 2, dpi=200)
            axes[0].imshow(res)
            axes[1].imshow(temp)
          classdict[j].append(temp)
for i in classdict:
  print(len(i))
