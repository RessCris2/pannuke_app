from mmdet.apis import DetInferencer

# from mmengine.runner import Runner  # , build_dataloader
from mmdet.registry import DATASETS, RUNNERS
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

# Setup a checkpoint file to load
checkpoint = "/root/autodl-tmp/pannuke_app/predict/maskrcnn/epoch_1.pth"
config_path = "/root/autodl-tmp/pannuke_app/predict/maskrcnn/consep_config.py"
# Initialize the DetInferencer
inferencer = DetInferencer(model=config_path, weights=checkpoint)


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

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

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
        ann_file="test/test_annotations.json",
        data_prefix=dict(img="test/imgs/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        indices=3,
    ),
)

# runner = Runner(default_scope="mmdet")
init_default_scope("mmdet")
# dataloader = Runner.build_dataloader(dataloader=val_dataloader)
# for idx, data_batch in enumerate(dataloader):
#     print(idx)
#     print(data_batch)

#     # 这明显是不可行的，因为 data_batch 已经是经过预处理过的数据，而 inferencer 会重新进行预处理。
#     # 所以这里就只能是，要么用 test 的流程做评估，要么就是用 predict 的流程做预测。如果要用 predict 就不可能用内置的方法去做评估。
#     # 考虑把评估的代码简化写
#     result = inferencer(
#         data_batch,
#         no_save_pred=False,
#         out_dir="./",
#         return_datasample=True,
#     )
#     break
cfg = Config.fromfile(config_path)
cfg.load_from = checkpoint
cfg.work_dir = "./work_dirs"
runner = RUNNERS.build(cfg)
# start testing
runner.test()
