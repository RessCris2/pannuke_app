crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/home/pannuke_app/train/datasets/CoNSeP'
dataset_type = 'RepeatDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    2336,
    3504,
)
img_suffix = '.png'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
metainfo = dict(
    classes=(
        'Inflammatory',
        'Healthy_epithelial',
        'Epithelial',
        'Spindle-shaped',
    ),
    palette=[
        (
            120,
            120,
            60,
        ),
        (
            20,
            120,
            160,
        ),
        (
            72,
            100,
            60,
        ),
        (
            111,
            67,
            60,
        ),
    ])
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=4,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=4,
        num_convs=1,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        170,
        170,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
seg_map_suffix = '.png'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='Test/images', seg_map_path='Test/seg_mask'),
        data_root='/home/pannuke_app/train/datasets/CoNSeP',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=3,
    dataset=dict(
        dataset=dict(
            data_prefix=dict(
                img_path='Train/Images', seg_map_path='Train/seg_mask'),
            data_root='/home/pannuke_app/train/datasets/CoNSeP',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.5,
                        2.0,
                    ),
                    scale=(
                        2336,
                        3504,
                    ),
                    type='RandomResize'),
                dict(
                    cat_max_ratio=0.75,
                    crop_size=(
                        256,
                        256,
                    ),
                    type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs'),
            ],
            type='RepeatDataset'),
        times=40000,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2336,
            3504,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        256,
        256,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='Test/images', seg_map_path='Test/seg_mask'),
        data_root='/home/pannuke_app/train/datasets/CoNSeP',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/pannuke_app/train/unet/consep/work-dir'
