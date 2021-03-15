_base_ = [
    './models/universenet50_2008s.py',
    '../_base_/datasets/coco_detection_mini_mstrain_320_640.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=16)

# lr=0.01 for total batch size 16 (1 GPU  * 16 samples_per_gpu)
# lr=0.04 for total batch size 64 (4 GPUs * 16 samples_per_gpu)
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)
