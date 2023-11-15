"""对 hovernet 模型进行预测
"""
import sys

sys.path.append("/root/autodl-tmp/pannuke_app/hover")
from hover.infer.tile import InferManager

model_path = (
    "/root/autodl-tmp/pannuke_app/projects/consep/hovernet/train/model_data/epoch=1.tar"
)

pred_dir = "/root/autodl-tmp/pannuke_app/datasets/processed/CoNSeP/test/imgs"


# --gpu='1' \
# --nr_types=6 \
# --type_info_path=type_info.json \
# --batch_size=64 \
# --model_mode=fast \
# --model_path=../pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
# --nr_inference_workers=8 \
# --nr_post_proc_workers=16 \

infer = InferManager(
    input_dir=pred_dir,
    output_dir="/root/autodl-tmp/pannuke_app/projects/consep/hovernet/predict/pred_data",
    gpu="1",
    nr_types=5,
    type_info_path="/root/autodl-tmp/pannuke_app/hover/type_info.json",
    batch_size=4,
    model_mode="original",
    model_path=model_path,
    nr_inference_workers=8,
    nr_post_proc_workers=8,
)

infer.process_file_list(
    output_dir="/root/autodl-tmp/pannuke_app/projects/consep/hovernet/predict/pred_data"
)
