default_scope = "mmdet"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=1),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # visualization=dict(  # 用户可视化验证和测试结果
    #     type="DetVisualizationHook",
    #     draw=True,
    #     interval=1,
    #     # test_out_dir="/root/autodl-tmp/pannuke_app/train/mask_rcnn/consep/model_data",
    #     score_thr=0.05,
    # ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    # dict(type="TensorboardVisBackend"),
    # dict(type="WandbVisBackend"),
]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    # save_dir="/root/autodl-tmp/pannuke_app/train/mask_rcnn/consep/model_data",
)

log_processor = dict(type="LogProcessor", window_size=1, by_epoch=True)

log_level = "DEBUG"
load_from = None
resume = False
