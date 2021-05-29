python src/train.py --cfg_path './src/configs/swin/swin_detection_strong_fpn.py' \
                    --exp_name 'swin_test' --img_prefix './input/data' \
                    --batch_size 1 \
                    --img_res 256 \
                    --cfg_options model.pretrained='./src/pretrained/swin_base_patch4_window12_384_22kto1k.pth'