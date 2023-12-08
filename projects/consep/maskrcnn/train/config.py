
# _base_ = [
#     "/root/autodl-tmp/pannuke_app/src/models/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
# ]
_base_ = [
    # "/root/autodl-tmp/pannuke_app/src/models/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
    "/root/mmlab/mmdetection-main/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
]
data_root = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/"

# consep
dataset_type = "CocoDataset"
# data_root = "/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/"
# data_root = "/home/pannuke_app/train/datasets/CoNSeP/"
metainfo = {
    "classes": ("Inflammatory", "Healthy_epithelial", "Epithelial", "Spindle-shaped"),
    "palette": [(120, 120, 60), (20, 120, 160), (72, 100, 60), (111, 67, 60)],
}

backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(800, 800)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    # dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/train_annotations.json",
        data_prefix=dict(img="train/imgs/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
        # indices=2,
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="test/test_annotations.json",
        data_prefix=dict(img="test/imgs/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        # indices=1,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "test/test_annotations.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
    outfile_prefix="work_dirs/val",
)
test_evaluator = val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "test/test_annotations.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
    outfile_prefix="work_dirs/test",
)


model = dict(
    roi_head=dict(bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4))
)


evaluation = dict(interval=1, metric="segm", options={"maxDets": [100, 300, 1000]})

# load_from = "work-dir/epch_6.pth"
# resume = True
# if osp.exists("/home/pannuke_app/"):

# if osp.exists("/root/autodl-tmp"):
#     load_from = "/root/autodl-tmp/pannuke_app/train/mask_rcnn/consep/model_data/old/epoch_13.pth"
#     resume = False
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=500, val_interval=10)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
    # dict(type="WandbVisBackend"),
]

visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    # save_dir="/root/autodl-tmp/pannuke_app/train/mask_rcnn/consep/model_data",
)

load_from = "/root/tf-logs/consep_maskrcnn/epoch_30.pth"
resume = False

optim_wrapper = dict(
    type='AmpOptimWrapper',
    # optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0001),
    loss_scale='dynamic')

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.03, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=500,
#         by_epoch=True,
#         milestones=[80, 110],
#         gamma=0.1)
# ]
param_scheduler = None
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
#     loss_scale='dynamic')
auto_scale_lr = dict(enable=False, base_batch_size=8)