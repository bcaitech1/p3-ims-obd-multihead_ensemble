# _base_ = "../detectors/detectors_htc_r50_1x_coco.py"
# _base_ = 'dataset.py'
_base_ = '../detectors/detectors_htc_r50_1x_coco.py'

epoch = 50

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='P-stage3-detection',
                name='detectors-v2')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
workflow = [('train', 1)]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',    
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    by_epoch=False,
    periods=[7860, 7860, 7860, 7860, 7860],
    restart_weights=[1, 0.7, 0.5, 0.3, 0.1],
    min_lr=5e-5)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epoch)
# evaluation = dict(interval=1, metric=['bbox'])
evaluation = dict(
    interval=1,
    save_best='bbox_mAP_50',
    metric=['bbox']
)

# max-depth 3
checkpoint_config = dict(max_keep_ckpts=3, interval=1)

