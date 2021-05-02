import os
import argparse
import pandas as pd
import numpy as np
import pickle as pickle
import multiprocessing as mp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from albumentations import *
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from src.dataset import *
from src.models import *
from src.utils import dense_crf_wrapper


def inference(args):
    def collate_fn(batch):
        return tuple(zip(*batch))

    
    test_path = dataset_path + '/test.json'

    test_transform = A.Compose([
                            A.Normalize(),
                            A.Resize(256, 256),
                            ToTensorV2()
                            ])

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = FCN8s(num_classes=12)
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b3",
                            encoder_weights="imagenet",
                            in_channels=3,
                            classes=12,
                            )

    model = model.to(device)

    model_path = os.path.join(args.saved_dir, args.model_name + ".pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).float().to(device))

            if args.is_crf:
                probs = F.softmax(outs, dim=1).data.cpu().numpy()

                pool = mp.Pool(mp.cpu_count())
                images = torch.stack(imgs).data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                probs = np.array(pool.map(dense_crf_wrapper, zip(images, probs)))
                pool.close()
                oms = np.argmax(probs, axis=1)
            else:
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    
    print("End prediction.")

    print("Make submission...")

    if not os.path.exists(args.submission_dir):
        os.makedirs(args.submission_dir)
    
    file_names = [y for x in file_name_list for y in x]
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    
    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(args.submission_dir, args.model_name + "csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--saved_dir", type=str, default="./checkpoints/exp4")
    parser.add_argument("--submission_dir", type=str, default="./submission")
    parser.add_argument("--model_name", type=str, default="deeplabv3p")
    parser.add_argument("--is_crf", type=bool, default=True)

    args = parser.parse_args()
    
    inference(args)