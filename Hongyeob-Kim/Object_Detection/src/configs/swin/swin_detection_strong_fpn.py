_base_ = [
    '../_base_/models/htc_swin_more_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05)
                 # paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                 #                                 'relative_position_bias_table': dict(decay_mult=0.),
                 #                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0/10,
#     min_lr_ratio=4e-6
# )
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    by_epoch=False,
    periods=[7860, 7860, 7860],
    restart_weights=[1, 0.5, 0.25],
    min_lr=4e-6)


runner = dict(type='EpochBasedRunner', max_epochs=40)