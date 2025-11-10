# dataset settings
dataset_type = 'DepthDataset'
data_root = 'path/to/your/dataset'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile'),
    dict(
        type='ResizeToMultiple',
        size_divisor=14,
        scale=(518, 518),
        keep_ratio=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_gt']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeToMultiple',
        size_divisor=14,
        scale=(518, 518),
        keep_ratio=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        depth_dir='depth/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        depth_dir='depth/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/testing',
        depth_dir='depth/testing',
        pipeline=test_pipeline))