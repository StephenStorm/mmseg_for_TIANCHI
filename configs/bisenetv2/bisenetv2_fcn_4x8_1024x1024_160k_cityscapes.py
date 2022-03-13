_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/tianchi_Aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_my.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.05)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
)
