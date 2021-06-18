import os
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg_path', default='./configs/swin/swin_detection.py', type=str)
	# parser.add_argument('--cfg_path', required=True)
	parser.add_argument('--seed', default=43, type=int)
	parser.add_argument('--img_prefix', default='../input/data', type=str)
	parser.add_argument('--train_json', default='train.json', type=str)
	parser.add_argument('--valid_json', default='val.json', type=str)
	parser.add_argument('--img_res', default=512, type=int)
	parser.add_argument('--exp_name', default='swin_test', type=str)
	parser.add_argument('--batch_size', default=8, type=int)
	parser.add_argument('--cfg_options',
	                    nargs='+',
                        action=DictAction)
	# parser.add_argument('--exp_name', required=True)
	args = parser.parse_args()


	cfg = Config.fromfile(args.cfg_path)
	if args.cfg_options is not None:
		cfg.merge_from_dict(args.cfg_options)

	cfg.data.train.classes = classes
	cfg.data.train.img_prefix = args.img_prefix
	cfg.data.train.ann_file = [os.path.join(args.img_prefix, args.train_json), os.path.join(args.img_prefix, "final_fixed_test.json")]

	cfg.data.val.classes = classes
	cfg.data.val.img_prefix = args.img_prefix
	cfg.data.val.ann_file = [os.path.join(args.img_prefix, args.valid_json), os.path.join(args.img_prefix, "final_fixed_test.json")]


	cfg.seed = args.seed
	cfg.data.samples_per_gpu = args.batch_size
	cfg.gpu_ids = [0]
	cfg.work_dir = f'./work_dirs/{args.exp_name}'

	model = build_detector(cfg.model)
	datasets = [build_dataset(cfg.data.train)]
	train_detector(model, datasets[0], cfg, distributed=False, validate=True)


if __name__ == "__main__":
	main()