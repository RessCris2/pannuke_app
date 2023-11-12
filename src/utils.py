import os
import shutil
import glob
import logging
import time
from functools import wraps
from os.path import join as opj

import cv2
import numpy as np
import openslide
from PIL import Image


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
                    # attempt to remove, if not present, skip
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
