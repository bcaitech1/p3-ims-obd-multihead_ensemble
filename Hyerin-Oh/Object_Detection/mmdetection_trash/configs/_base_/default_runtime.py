checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='P-stage3-detection',
                name='Swin-Large')
        )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
