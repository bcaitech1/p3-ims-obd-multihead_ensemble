python augmix_train.py --seed 42 --batch_size 16 --lr 1e-4 --trn_ratio 0.0 \
                     --decay 1e-6 --epochs 50 \
                     --cutmix_beta 0.0 --use_weight 1 --use_augmix 1 \
                     --version "hrnet_ocr_v12" --model_type 'hrnet_ocr'

# Weighted DiceCELoss(ce 0.7, dice 0.3)
# Hflip, VFlip, OneOf(Rotate90, Rotate30), OneOf(CLAHE(0.5), ElasticTransform, OpticalDistortion, GridDistortion)