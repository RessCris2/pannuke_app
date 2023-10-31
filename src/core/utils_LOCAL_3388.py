# %load_ext autoreload
# %autoreload 2
import os
import torch
import numpy as np
from skimage.morphology import erosion, disk
import cv2
import shutil
import openslide
from PIL import Image
import time
from os.path import join as opj
import glob
import logging
from functools import wraps
import yaml


def read_yaml_to_dict(
    yaml_path: str,
):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


# utils
def add_contours(rgb_image, contour, ds=2):
    """
    Adds contours to images.
    The image has to be a binary image
    """
    rgb = rgb_image.copy()
    contour[contour > 0] = 1
    boundery = contour - erosion(contour, disk(ds))
    rgb[boundery > 0] = np.array([0, 0, 0])
    return rgb


def loads_model(model, path, map_location="cpu", mode="eval"):
    # model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location=map_location))

    if mode == "evel":
        model.eval()  # 用于预测时使用
    else:
        model.train()  # 用于继续训练时使用
    return model


def load_img(path):
    """
    cv2 读取图片
    to_rgb: 转换为 rgb
    flags=0, 表示读取灰度图
    """
    if path.endswith("npy"):
        img = np.load(path)

    else:
        if path.endswith("png"):
            # 为 seg_mask 文件
            flags = 0
            to_rgb = False
        else:
            flags = None
            to_rgb = True

        if flags is not None:
            img = cv2.imread(path, flags=flags)
        else:
            img = cv2.imread(path)

        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=False)


def svs_to_tif(path):
    img = openslide.OpenSlide(path)
    cv2.imwrite(
        path.replace("svs", "tif"),
        np.array(img.read_region((0, 0), 0, img.level_dimensions[0])),
    )


def save_img(path, img):
    if path.endswith("npy"):
        np.save(path, img)
    elif path.endswith("png"):
        Image.fromarray(img.astype(np.uint8)).save(path, "PNG")
    elif path.endswith("jpg"):
        Image.fromarray(img.astype(np.uint8)).save(path, "JPEG")


def get_curtime():
    time_str = time.strftime("%Y%m%d%H%M", time.localtime())
    return time_str


def find_files(path, ext, exs=None):
    file_list = glob.glob(opj(path, "*.{}".format(ext)))
    if ext == "jpg":
        if exs is not None:
            exs = np.load(exs, allow_pickle=True)
            for ex in exs:
                try:
                    # 尝试移除，如果不存在，就跳过
                    file_list.remove(ex)
                except Exception:
                    continue
    return file_list


def get_logger(log_file_name, log_dir, level=logging.INFO, when="D", back_count=10):
    """
    params:
        log_file_name: 日志名称, 同时也会和 log_dir, 拼接为最终的log文件名称
        log_dir: 需要写入的目录文件夹，如果没有创建会自动创建
        level:  需要显示的日志等级：
            logging.DEBUG < logging.INFO < logging.WARNING < logging.ERROR
            < logging.CRITICAL
            when: 文件新起一个log 间隔时间:S:秒 M:分 H:小时 D:天 W:每星期（interval==0时代表星期一）
              midnight: 每天凌晨
        back_count: 备份文件的个数，若超过该值，就会自动删除
    return:
     logger

    使用方法： get_logger 相当于只是指定输出位置和格式, 具体在什么位置输出, 还是由 logging.info 指定
    import logging
    dirname = os.path.dirname(__file__)
    logger = get_logger('process_deepwalk_data.log', os.path.join(dirname, 'logs'),
                level=logging.INFO)
    logging.info("prdc ----{0}----{1}----数据量为{2}".format(date, prov, prdc.shape[0]))
    logging.info("user ----{0}----{1}----数据量为{2}".format(date, prov, user.shape[0]))
    logging.info("interactions ----{0}----{1}----数据量为{2}".format(date, prov,
            interactions.shape[0]))
    logging.info("testdata ----{0}----{1}----数据量为{2}".format(date, prov,
        testdata.shape[0]))
    """
    logger = logging.getLogger(log_file_name)
    logger.setLevel(level)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = os.path.join(log_dir, log_file_name)

    # 创建格式器
    formatter = logging.Formatter(
        "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )

    # 创建处理器：ch为控制台处理器，fh为文件处理器
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # 输出到文件
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path, when=when, backupCount=back_count, encoding="utf-8"
    )
    fh.setLevel(level)
    # 设置日志输出格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将处理器，添加至日志器中
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def fn_time(func):
    """耗时装饰器"""

    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        dur = time.time() - start_time
        print("{}的执行时间：{}s".format(func.__name__, np.round(dur, 4)))
        return res

    return inner


# 强数据处理相关？
def fetch_inst_centroid_maskrcnn(pred_masks):
    """maskrcnn 的预测结果中包含了 bboxes, label, masks 这些"""
    # inst_id_list = pred_id_list  # exlcude background
    inst_id_list = list(range(len(pred_masks)))  # 直接就取 pred_masks 的第一维度
    inst_centroid_list = []
    for inst_id in inst_id_list:
        inst_mapp = pred_masks[inst_id]  # 获取每个inst_id 对应的 mask
        # TODO: chane format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(inst_mapp)
        #     rmin, cmin, rmax, cmax= pred_bboxes[inst_id]
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_mappp = inst_mapp[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map = inst_mappp.astype(np.uint8)

        inst_moment = cv2.moments(inst_map)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


def transfer_inst_format(true_inst, true_type_map):
    """将 inst [256, 256]
    转换为
    dict(
    scores:
    labels:
    bboxes:
    masks: (N, 256, 256)
    )
    的格式

    如果传入 true_type_map: 也就是 type_map 的结果, 则进一步转化 labels
    """
    true = {}
    # true_inst_id = np.unique(true_inst)[1:]
    true_inst_id = list(np.unique(true_inst))
    if 0 in true_inst_id:
        true_inst_id.remove(0)
    try:
        true_masks = np.stack(
            [(true_inst == inst_id).astype(int) for inst_id in true_inst_id]
        )
    except Exception as e:
        print("something wrong", e)
        raise ValueError
    true_bboxes = np.stack(
        [
            get_bounding_box((true_inst == inst_id).astype(int))
            for inst_id in true_inst_id
        ]
    )
    true_scores = np.array([0.99] * len(true_inst_id))
    # fake one!
    if true_type_map is None:
        true_labels = np.array([1] * len(true_inst_id))  # 这部分要处理
    else:
        true_labels = (
            np.array(
                [
                    np.unique(true_type_map[true_inst == inst_id])[0]
                    for inst_id in true_inst_id
                ]
            )
            - 1
        )  # 对 type_map取值，待测试; 有很大bug的样子

    true_centroids = fetch_inst_centroid_maskrcnn(true_masks)

    true.update({"bboxes": true_bboxes})
    true.update({"scores": true_scores})
    true.update({"masks": true_masks})
    true.update({"labels": true_labels})
    true.update({"centroids": true_centroids})
    return true


def load_gt(img_path, is_class=True):
    """获取相应 img_path 的 gt 标签"""
    type_mask = load_img(img_path.replace("images", "seg_mask").replace("jpg", "png"))
    inst_mask = load_img(img_path.replace("images", "inst").replace("jpg", "npy"))
    if is_class:
        true = transfer_inst_format(inst_mask, type_mask)
    else:
        true = transfer_inst_format(inst_mask, true_type_map=None)
    return true
