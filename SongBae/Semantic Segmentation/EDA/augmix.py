import copy
import warnings
import tqdm
import numpy as np
import albumentations as A
import cv2

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

np.save(classdict, "augmix.npy")
for i in classdict:
    print(len(i))
