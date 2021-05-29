# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.00005  # max learning rate
optimizer = dict(type='Adam', lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(policy='step', warmup=None, step=[24, 32])
'''
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=4e-6)
'''

# runtime settings
total_epochs = 24

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',    
    warmup_iters=1000,
    warmup_ratio=1.0 / 9,
    by_epoch=False,
    periods=[10356, 10356],
    restart_weights=[1, 0.35],
    min_lr=5e-6)