dataset_type = 'CocoDataset'
data_root = './input/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 2)),
    # dict(type='MinIoURandomCrop',
    #                   min_ious=(0.5, 0.6, 0.7, 0.8, 0.9),
    #                   min_crop_size=0.4),
    dict(
        type="AutoAugment",
        policies=[
                    [
                        dict(
                            type="RandomCrop",
                            crop_type="absolute_range",
                            crop_size=(384, 384),
                            allow_negative_crop=True,
                        ),
                        dict(
                            type="Resize",
                            img_scale=(512, 512),
                            multiscale_mode="value",
                            override=True,
                            keep_ratio=True,
                        )
                    ]
                ]
    ),
    #########################################
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])#, 'gt_masks']),
]


fixed_test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])#, 'gt_masks']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(384, 512), (416, 512), (512, 512)],
        # img_scale=(512,512),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# train_ori = dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file= data_root + 'train.json',
#         img_prefix=data_root,
#         pipeline=train_pipeline
# )
#
# train_test = dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file= data_root + 'fixed_test.json',
#         img_prefix=data_root,
#         pipeline=train_pipeline
# )


data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file= [data_root + 'train.json', data_root + 'fixed_test.json'],
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file= [data_root + 'val.json', data_root + 'fixed_test.json'],
        img_prefix=data_root,
        separate_eval=True,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))