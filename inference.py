from torch.utils.data import DataLoader
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os
from tqdm import tqdm
from src.dataset import *
from src.models import *

from albumentations import *
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def inference(args):
    def collate_fn(batch):
        return tuple(zip(*batch))

    
    test_path = dataset_path + '/test.json'

    test_transform = A.Compose([
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
    model = smp.DeepLabV3(encoder_name="efficientnet-b0",
            encoder_depth=5,
            encoder_weights="imagenet",
            in_channels=3,
            classes=args.num_classes,
            )
    model = model.to(device)

    model_path = os.path.join(args.saved_dir, args.file_name)
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
            outs = model(torch.stack(imgs).to(device))
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
    submission.to_csv(os.path.join(args.submission_dir, args.submission_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--saved_dir", type=str, default="./checkpoints/exp3")
    parser.add_argument("--file_name", type=str, default="deeplabv3.pt")
    parser.add_argument("--submission_dir", type=str, default="./submission")
    parser.add_argument("--submission_name", type=str, default="deeplabv3.csv")
    parser.add_argument("--num_classes", type=int, default=12)

    args = parser.parse_args()
    
    inference(args)