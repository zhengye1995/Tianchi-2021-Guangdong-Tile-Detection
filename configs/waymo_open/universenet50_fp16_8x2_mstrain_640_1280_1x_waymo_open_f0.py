_base_ = [
    '../_base_/models/universenet50.py',
    '../_base_/datasets/waymo_open_2d_detection_f0_mstrain_640_1280.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=3))

data = dict(samples_per_gpu=2)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)

load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.06/universenet50_fp16_8x2_mstrain_480_960_2x_coco_20200523_epoch_24-726c5c93.pth'  # noqa
