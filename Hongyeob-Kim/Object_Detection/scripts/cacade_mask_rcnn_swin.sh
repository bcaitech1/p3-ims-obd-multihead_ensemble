python src/train.py --cfg_path './src/configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py' \
                    --exp_name 'cascade_mask_rcnn_swin_base' --img_prefix './input/data' \
                    --img_res 512 \
                    --cfg_options model.pretrained='./src/pretrained/swin_base_patch4_window12_384_22kto1k.pth'