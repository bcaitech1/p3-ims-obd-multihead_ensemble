# optimizer
lr = 0.00004
optimizer = dict(type='Adam', lr=lr, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',    
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    by_epoch=False,
    periods=[7860, 7860, 7860],
    restart_weights=[1, 0.7, 0.5],
    min_lr=5e-6)
runner = dict(type='EpochBasedRunner', max_epochs=36)
