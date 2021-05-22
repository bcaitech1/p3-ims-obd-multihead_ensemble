python src/train.py --cfg_path './src/configs/swin/swin_detection_strong_fpn.py' \
                    --exp_name 'htc_swin_v5' --img_prefix './input/data' \
                    --batch_size 4 \
                    --img_res 512 \
                    --cfg_options model.pretrained='./src/pretrained/swin_base_patch4_window12_384_22kto1k.pth'