# _base_ = ["../../../src/unet/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py"]
# _base_ = ["/home/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py"]
_base_ = [
    "/root/autodl-tmp/pannuke_app/src/models/unet/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py"
]
# -----------------------------------------------------------------------------
# dataset settings
# data_root = "/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/"

# consep
dataset_type = "CoNSePDataset"
data_root = "/root/autodl-tmp/pannuke_app/projects/monusac/training_data"
# data_root = "/home/pannuke_app/train/datasets/CoNSeP/"
metainfo = {
    # "classes": ("Inflammatory", "Healthy_epithelial", "Epithelial", "Spindle-shaped"),
    "classes": ("Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"),
    "palette": [(120, 120, 60), (20, 120, 160), (72, 100, 60), (111, 67, 60)],
}

img_suffix = ".png"
seg_map_suffix = ".png"

img_scale = (256, 256)
crop_size = (256, 256)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=img_scale, ratio_range=(0.8, 1.3), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(256, 256), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        # type='RepeatDataset',
        # times=40000,
        dataset=dict(
            metainfo=metainfo,
            type=dataset_type,
            data_root=data_root,
            reduce_zero_label=True,
            data_prefix=dict(img_path="train/imgs", seg_map_path="train/seg_mask"),
            pipeline=train_pipeline,
        )
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(img_path="test/imgs", seg_map_path="test/seg_mask"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator


# -----------------------------------------------------------------------------
model = dict(decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
