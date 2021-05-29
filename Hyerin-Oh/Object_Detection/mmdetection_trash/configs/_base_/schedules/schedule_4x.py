# optimizer = dict(type='MADGRAD', lr=1e-4, momentum=0.9,paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))

lr = 0.00005  # max learning rate
optimizer = dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))


# runtime settings
total_epochs = 60

 # cosine anealing scheduler
lr_config = dict(policy='CosineAnnealing',warmup='linear',warmup_iters=3000,
                    warmup_ratio=0.0001, min_lr_ratio=1e-7)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=60)