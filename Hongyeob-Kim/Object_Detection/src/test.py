import os
from pycocotools.coco import COCO
import pandas as pd

from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg_path', default='./configs/swin/swin_detection.py', type=str)
	# parser.add_argument('--cfg_path', required=True)
	parser.add_argument('--seed', default=43, type=int)
	parser.add_argument('--ckpt', required=True)
	parser.add_argument('--name', required=True)
	parser.add_argument('--img_prefix', default='../input/data', type=str)
	parser.add_argument('--train_json', default='train.json', type=str)
	parser.add_argument('--valid_json', default='val.json', type=str)
	parser.add_argument('--test_json', default='test.json', type=str)
	parser.add_argument('--img_res', default=512, type=int)
	parser.add_argument('--exp_name', default='swin_test', type=str)
	args = parser.parse_args()


	cfg = Config.fromfile(args.cfg_path)
	cfg.data.test.classes = classes
	cfg.data.test.img_prefix = args.img_prefix
	cfg.data.test.ann_file = os.path.join(args.img_prefix, args.test_json)

	cfg.gpu_ids = [0]
	cfg.work_dir = f'./work_dirs/{args.exp_name}'

	checkpoint_path = os.path.join(cfg.work_dir, f'{args.ckpt}.pth')

	print("\n", cfg.data.test)
	dataset = build_dataset(cfg.data.test)
	data_loader = build_dataloader(
		dataset,
		samples_per_gpu=1,
		workers_per_gpu=cfg.data.workers_per_gpu,
		dist=False,
		shuffle=False)

	print(cfg)
	model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

	print("Loading ckpt.")
	checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
	print("Done.\n")

	model.CLASSES = classes
	model = MMDataParallel(model.cuda(), device_ids=[0])

	print("Testing....")
	output = single_gpu_test(model, data_loader, show_score_thr=0.05)
	print("Done.")

	prediction_strings = []
	file_names = []
	coco = COCO(cfg.data.test.ann_file)
	imag_ids = coco.getImgIds()

	class_num = 11
	for i, out in enumerate(output):
		prediction_string = ''
		image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
		for j in range(class_num):
			for o in out[j]:
				prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(o[2]) + ' ' + str(o[3]) + ' '

		prediction_strings.append(prediction_string)
		file_names.append(image_info['file_name'])

	submission = pd.DataFrame()
	submission['PredictionString'] = prediction_strings
	submission['image_id'] = file_names
	submission.to_csv(os.path.join(cfg.work_dir, f'{args.name}.csv'), index=None)
	print("Finished.")



if __name__ == "__main__":
	main()