# dataset settings
dataset_type = 'TianchiDataset'  # 上一步中你定义的数据集的名字
data_root = 'data/tianchi_aug'  # 数据集存储路径
iamge_scale_t = (768, 768)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 数据集的均值和标准差，空引用默认的，也可以网上搜代码计算
crop_size = (512, 512)  # 数据增强时裁剪的大小
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='RandomMosaic', prob=1, img_scale=iamge_scale_t),
    dict(type='Resize', img_scale=iamge_scale_t,
         ratio_range=(0.5, 2.0)),  # img_scale图像尺寸
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCutOut', prob=0.5, n_holes = 3, cutout_shape = [(4, 4), (4, 8), (8, 4),(8, 8), (16, 8), (8, 16),
                                            (16, 16), (16, 32), (32, 16) ]),
    dict(type='RandomRotate', prob = 0.5, degree=90),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),  # img_scale图像尺寸
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



train_dataset_mosaic = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',  # 训练图像路径
        ann_dir='annotations/training',  # 训练mask路径
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations')
        ],
    ),
    pipeline=train_pipeline
)
train_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',  # 训练图像路径
        ann_dir='annotations/training',  # 训练mask路径
        pipeline=train_pipeline)


data = dict(
    train=train_dataset_mosaic,
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',  # 验证图像路径
        ann_dir='annotations/validation',  # 验证mask路径
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/img',  # 测试图像路径
        ann_dir=None,  # 测试mask路径
        pipeline=test_pipeline)
)
