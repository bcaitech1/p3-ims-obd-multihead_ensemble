python src/test.py --cfg_path './src/configs/swin/swin_detection_strong_fpn.py' \
                    --exp_name 'htc_swin_v9' --img_prefix './input/data' \
                    --img_res 512 \
                    --ckpt "epoch_40" \
                    --name "pseudo_cascade_rcnn_swin_v9"