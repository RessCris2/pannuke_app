"""对模型预测结果，应用后处理算法，得到最后的分割结果
"""

import pathlib
import sys
from os.path import join as opj

import cv2
import numpy as np
from tqdm import tqdm

from ..core.utils import find_files, load_img, loads_model, rm_n_mkdir
from .dataloader import data_aug
from .dist_net import DIST
from .post_proc import post_process


def predict(model, img):
    img = data_aug()(img).unsqueeze(0)
    mp = model(img).detach().numpy().squeeze()
    mp = np.clip(mp, 0, 255)
    pred = post_process(mp)  # p1, p2 取默认值
    return pred


def predict_batch(model, dataset_name):
    """
    通过 predict_loader 进行数据预处理; predict 每一张图片并保存为 mat.
    每一张图片的 pred 为 (256, 256);
    pred 的结果是 inst_map; 需要获取 type_map? no, 没有type part; 就只是保存为 inst 即可 npy

    这个地方代码可以调整为 cuda 预测, 加快预测速度...
    """
    model_name = "dist"
    img_dir = (
        "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/images/test".format(
            dataset_name
        )
    )
    file_list = find_files(img_dir, ext="jpg")
    save_dir = "/root/autodl-tmp/archive/datasets/{}/patched/infer/{}".format(
        dataset_name, model_name
    )
    rm_n_mkdir(save_dir)
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        basename = pathlib.Path(file).stem
        img = load_img(file)
        pred = predict(model, img)
        np.save(opj(save_dir, "{}.npy".format(basename)), pred)


if __name__ == "__main__":
    save_path = "/root/autodl-tmp/archive/v2/model_data/dist/pannuke/202305301616/202305301622/202305301627/202305301632/202305301637/202305301643/202305301648/epoch_30.pth"
    model = DIST(num_features=6)
    model = loads_model(model, save_path)
    # 只对一张图片进行评估
    # img = load_img("/root/autodl-tmp/archive/datasets/pannuke/patched/coco_format/images/test/0.jpg")
    # pred = predict(model, img)
    dataset_name = "pannuke"
    predict_batch(model, dataset_name)
