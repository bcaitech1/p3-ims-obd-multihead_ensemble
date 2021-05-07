python pseudo_train.py --seed 42 --batch_size 16 --lr 1e-4 --trn_ratio 0.0 \
                       --decay 1e-6 --epochs 50 \
                       --cutmix_beta 0.0 --use_weight 1 --use_augmix 1 \
                       --version "pseudo_hrnet_ocr_v3" --model_type 'hrnet_ocr' \
                       --loss_type "OhMyLoss"

# Weighted DiceCELoss(ce 0.8, fc 0.2)

#        A.HorizontalFlip(p=0.5),
#        A.OneOf([
#            A.RandomRotate90(p=1.0),
#            A.Rotate(limit=30, p=1.0),
#        ], p=0.5),
#
#        A.RandomGamma(p=0.3),
#        A.RandomBrightness(p=0.5),
#        A.OneOf([
#            A.CLAHE(p=1.0),
#            A.ElasticTransform(p=1, alpha=40, sigma=40 * 0.05, alpha_affine=40 * 0.03),
#            A.GridDistortion(p=1),
#            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
#        ], p=0.6),

# SongBae Mix w/ prob 0.15 -> 배치가 16일때 확률 상으로는 2개 씩 학습가능
# 기존 prob 0.5로 생각해보면 8개씩 들어감... -> 너무 많이 들어가서 학습초기에 높은 mIoU이지만 후반으로 갈수록 달리는거 같음