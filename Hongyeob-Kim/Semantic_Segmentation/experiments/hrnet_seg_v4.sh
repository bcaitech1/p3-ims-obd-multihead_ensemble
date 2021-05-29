python train_eval.py --seed 42 --batch_size 16 --lr 1e-4 --trn_ratio 0.0 \
                     --decay 1e-6 --epochs 25 --cutmix_beta 0.0 --use_weight 1 \
                     --version "hrnet_ocr_v4" --model_type 'hrnet_ocr'
# Weighted DiceCELoss
