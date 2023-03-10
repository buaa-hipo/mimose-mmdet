_base_ = './retinanet_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
    memory_threshold=14,
    memory_buffer=6,
    dc=dict(
        enable=True,
        warmup_iters=30,
    )
)

data = dict(
    samples_per_gpu=6
)