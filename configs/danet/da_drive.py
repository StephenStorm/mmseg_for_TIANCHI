_base_ = [
    '../_base_/models/danet_r50-d8.py',
    '../_base_/datasets/drive.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
# runner = dict(type='IterBasedRunner', max_iters=20000)
