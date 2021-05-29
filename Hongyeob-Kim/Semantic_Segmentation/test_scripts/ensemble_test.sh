python ensemble_test.py --seed 42 --batch_size 32 --postfix "pseudo_hrnet_ocr_[v1,v2]" \
               --model_type "hrnet_ocr hrnet_ocr" --ckpt "pseudo_hrnet_ocr_v2/best_mIoU.pth pseudo_hrnet_ocr_v1/best_mIoU.pth" \
               --debug 1