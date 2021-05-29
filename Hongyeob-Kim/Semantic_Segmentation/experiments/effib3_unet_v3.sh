python train_eval.py --seed 42 --batch_size 16 --lr 1e-4 --trn_ratio 0.0 \
                     --decay 1e-6 --epochs 20 --cutmix_beta 1.0 \
                     --version "effib3_unet_v3" --model_type 'unet'


# 컷믹스
# IoULoss