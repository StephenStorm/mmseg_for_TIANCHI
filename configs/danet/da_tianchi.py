_base_ = [
    '../_base_/models/danet_r50-d8.py',   #这个是网络的骨架，使用单卡记得去骨架模型里将SyncBN改成BN
    '../_base_/datasets/tianchi.py',   #换成自己定义的数据集
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_my.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2)) #换成自己的分类类别数

