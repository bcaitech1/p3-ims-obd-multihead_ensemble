_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

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
                name='faster-rcnn')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=1e-3, weight_decay=5e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=30)
# evaluation = dict(interval=1, metric=['bbox'])
evaluation = dict(
    interval=1,
    save_best='bbox_mAP_50',
    metric=['bbox']
)

# max-depth 3
checkpoint_config = dict(max_keep_ckpts=3, interval=1)