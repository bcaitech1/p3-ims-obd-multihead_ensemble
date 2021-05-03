python augmix_train.py --seed 42 --batch_size 10 --lr 1e-4 --trn_ratio 0.0 \
                     --decay 1e-6 --epochs 40 \
                     --cutmix_beta 0.0 --use_weight 1 --use_augmix 1 \
                     --version "hrnet_ocr_tp_v2" --model_type 'hrnet_ocr_tp'

# Weighted DiceCELoss(ce 0.7, dice 0.3)
# Hflip, VFlip(0.3), OneOf(Rotate90, Rotate30), RandomBrightness, OneOf(CLAHE, ElasticTransform, OpticalDistortion, GridDistortion),