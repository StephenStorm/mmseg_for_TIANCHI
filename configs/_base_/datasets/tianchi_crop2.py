# dataset settings
dataset_type = 'TianchiDataset'  # 上一步中你定义的数据集的名字
# data_root = 'data/tianchi_aug'  # 数据集存储路径
data_root = 'data/new_tianchi_data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 数据集的均值和标准差，空引用默认的，也可以网上搜代码计算
img_scale = (1024, 1024)  # 数据增强时裁剪的大小
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='LoadAnnotationsTianchi'),
    dict(type='Resize', img_scale=img_scale,
        #  ratio_range=(1.0, 1.0)),
         ratio_range=(0.5, 2.0)),  # img_scale图像尺寸
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
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
        img_scale=img_scale,  # img_scale图像尺寸
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



data = dict(
    samples_per_gpu=2,  # batch_size
    workers_per_gpu=4,  # nums gpu
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train_val',  # 训练图像路径
        ann_dir='annotations/train_val',  # 训练mask路径
        pipeline=train_pipeline),
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

# data = dict(
#     samples_per_gpu=16,  # batch_size
#     workers_per_gpu=4,  # nums gpu
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='train_images',  # 训练图像路径
#         ann_dir='train_labels',  # 训练mask路径
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='val_images',  # 验证图像路径
#         ann_dir='val_labels',  # 验证mask路径
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='val_images',  # 测试图像路径
#         ann_dir=None,  # 测试mask路径
#         pipeline=test_pipeline)
# )
