python src/test.py --cfg_path './src/configs/swin/swin_detection.py' \
                    --exp_name 'swin_B_384' --img_prefix './input/data' \
                    --batch_size 8 \
                    --img_res 512 \
                    --ckpt "epoch_36" \
                    --name "swin_test"