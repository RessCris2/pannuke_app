
_base_ = [
    # "/root/autodl-tmp/pannuke_app/src/models/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
      "/root/mmlab/mmdetection-main/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
]
data_root = "/root/autodl-tmp/pannuke_app/datasets/processed/PanNuke/"

# consep
dataset_type = "CocoDataset"
# data_root = "/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/"
# data_root = "/home/pannuke_app/train/datasets/CoNSeP/"
metainfo = {
    "classes": ("Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"),
    "palette": [(200, 10, 60), (120, 120, 60), (20, 120, 160), (72, 100, 60), (111, 67, 60)],
}

backend_args = None
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(256, 256)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    # dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=64,
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
    batch_size=64,
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
        indices=3,
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
    roi_head=dict(bbox_head=dict(num_classes=5), mask_head=dict(num_classes=5))
)


evaluation = dict(interval=10, metric="segm", options={"maxDets": [100, 300, 1000]})

# if osp.exists("/home/pannuke_app/"):

# if osp.exists("/root/autodl-tmp"):
load_from = None
resume = False
