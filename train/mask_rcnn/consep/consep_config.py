_base_ = [
    # "/root/autodl-tmp/archive/v2/models/mask_rcnn/file/config/mask-rcnn_r50_fpn_1x_coco.py"
    "/home/pannuke_app/src/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
]

# consep
dataset_type = "CocoDataset"
data_root = "/home/pannuke_app/train/datasets/CoNSeP/"
metainfo = {
    "classes": ("Inflammatory", "Healthy_epithelial", "Epithelial", "Spindle-shaped"),
    "palette": [(120, 120, 60), (20, 120, 160), (72, 100, 60), (111, 67, 60)],
}

backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomCrop", crop_size=(256, 256)),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="Train/annotations.json",
        data_prefix=dict(img="Train/Images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="Test/annotations.json",
        data_prefix=dict(img="Test/Images/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# autodl-tmp/archive/datasets/monusac/patched/coco_format/annotations/test/instances_test.json
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "Test/annotations.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator


model = dict(
    roi_head=dict(bbox_head=dict(num_classes=4), mask_head=dict(num_classes=4))
)
