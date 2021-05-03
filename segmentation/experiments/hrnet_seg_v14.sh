python augmix_train.py --seed 42 --batch_size 16 --lr 1e-4 --trn_ratio 0.0 \
                     --decay 1e-6 --epochs 50 \
                     --cutmix_beta 0.0 --use_weight 1 --use_augmix 1 \
                     --version "hrnet_ocr_v14" --model_type 'hrnet_ocr'

# Weighted FocalCELoss(ce 0.7, fc 0.3)
# Hflip, VFlip(0.3), OneOf(Rotate90, Rotate30), RandomBrightness, OneOf(CLAHE, ElasticTransform, OpticalDistortion, GridDistortion),
# SongBae Mix w/ prob 0.15 -> 배치가 16일때 확률 상으로는 2개 씩 학습가능
# 기존 prob 0.5로 생각해보면 8개씩 들어감... -> 너무 많이 들어가서 학습초기에 높은 mIoU이지만 후반으로 갈수록 달리는거 같음