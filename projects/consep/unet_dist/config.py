# _base_ = ["../../../src/unet/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py"]
_base_ = [
    "/root/autodl-tmp/pannuke_app/src/models/unet_dist/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py"
]
data_root = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP"

# _base_ = [
#     "/root/mmlab/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-40k_hrf-256x256.py"
# ]
# data_root = "/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/"
# -----------------------------------------------------------------------------
# dataset settings
# data_root = "/root/autodl-tmp/pannuke_app/train/datasets/CoNSeP/"

# consep
dataset_type = "HRFDataset"


metainfo = {
    "classes": ("Inflammatory", "Healthy_epithelial", "Epithelial", "Spindle-shaped"),
    "palette": [(120, 120, 60), (20, 120, 160), (72, 100, 60), (111, 67, 60)],
}

img_suffix = ".png"
seg_map_suffix = ".png"

img_scale = (2336, 3504)
crop_size = (256, 256)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
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
            type=dataset_type,
            data_root=data_root,
            reduce_zero_label=True,
            data_prefix=dict(img_path="train/imgs", seg_map_path="train/dist_map"),
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
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(img_path="test/imgs", seg_map_path="test/dist_map"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"]) # miou 和 dist_map 怎么算损失？
test_evaluator = val_evaluator


# -----------------------------------------------------------------------------
model = dict(
    decode_head=dict(
        num_classes=4,
        # loss_decode=dict(type="MyLoss"),
    ),
    auxiliary_head=dict(
        num_classes=1,
        loss_decode=dict(type="MyLoss", loss_weight=0.8),  ## 这个效果可能好吗？真诡异。
    ),
)
