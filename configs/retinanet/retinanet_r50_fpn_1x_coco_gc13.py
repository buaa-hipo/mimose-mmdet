_base_ = './retinanet_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(depth=50, with_cp=True))

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
    memory_threshold=13,
)
