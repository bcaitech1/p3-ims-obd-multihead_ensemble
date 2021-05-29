import os
import cv2
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--f', required=False, default='csv/submission_ensemble_all_FUCKJH.csv')
args = parser.parse_args()

root_dir = '/opt/ml/input/data/'
save_path = '/opt/ml/code/mmdetection_trash/submission_image'
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(f'/opt/ml/code/mmdetection_trash/{args.f}')

image_id = df.image_id
prediction_string = df.PredictionString

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

for i in tqdm(range(837)):    
    img_path = os.path.join(root_dir, image_id[i])
    # print(img_path)
    img = cv2.imread(img_path)
    annot = prediction_string[i].split(' ')
    annot = [annot[i * 6:(i + 1) * 6] for i in range((len(annot) + 6 - 1) // 6 )] 
    annot.remove([''])
    output_img = img.copy()
    for bbox in annot:
        if 'nan' in bbox:
            continue
        bbox = list(map(float, bbox))
        bbox = np.array(bbox).astype(np.int64)
        label = int(bbox[0])
        xmin, ymin, xmax, ymax = bbox[2:6]

        color = COLORS[label]
        try:
            cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
        except:
            print(a)
            print((xmin, ymin), (xmax, ymax))        
        text_size = cv2.getTextSize(classes[label], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 2, ymin + text_size[1] + 6), color, -1)
        cv2.putText(
            output_img, classes[label],
            (xmin, ymin + text_size[1] + 4), cv2.FONT_ITALIC, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA)        
    result_img = np.concatenate((img, output_img), axis=1)
    
    cv2.imwrite(os.path.join(save_path, image_id[i].replace('/', '-')), result_img)