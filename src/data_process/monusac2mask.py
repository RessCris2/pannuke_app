import glob
import os
import os.path as osp
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
# import openslide
import openslide
from skimage import draw
from tqdm import tqdm

sys.path.append("/root/autodl-tmp/pannuke_app/src")
from utils import rm_n_mkdir, svs_to_tif


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


def read_xml(xml_file):
    name_to_id = {'Epithelial':1,
                  'Lymphocyte':2,
                  'Neutrophil':3,
                    'Macrophage':4,

    }
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_path = xml_file.replace("xml", "tif")
    img_shape = cv2.imread(img_path).shape[:2]

    inst_mask = np.zeros(shape=img_shape)
    type_mask = np.zeros(shape=img_shape)

    inst_id = 0
    for k in root:
        # type_id = int(k.attrib["Id"])
        type_name = k[0][0].attrib['Name']
        if type_name not in name_to_id.keys():
            continue
        type_id = name_to_id[type_name]


        # cnt=0
        for x in k[1][1:]:
            # print(cnt)
            # cnt = cnt+1
            regions = []
            vertices = x[1]
            coords = np.zeros((len(vertices), 2))
            for i, vertex in enumerate(vertices):
                coords[i][0] = vertex.attrib["X"]
                coords[i][1] = vertex.attrib["Y"]
            regions.append(coords)
            if len(regions[0]) < 4:
                continue
            # poly = Polygon(regions[0])

            vertex_row_coords = regions[0][:, 0]
            vertex_col_coords = regions[0][:, 1]
            fill_row_coords, fill_col_coords = draw.polygon(
                vertex_col_coords, vertex_row_coords, inst_mask.shape
            )
            inst_id += 1
            inst_mask[fill_row_coords, fill_col_coords] = inst_id  # int(x.attrib["Id"])
            type_mask[fill_row_coords, fill_col_coords] = type_id

            # print(inst_id, type_id)
    # np.save(inst_path, inst_mask)
    # np.save(type_path, type_mask)
    return inst_mask, type_mask


def convert(
    data_path,
    dest_path,
):
    """对文件夹中的 xml 文件批量处理"""
    dirs = glob.glob(osp.join(data_path, "*"))  # Creating a glob object
    # 每个 病人还有n张图片，要生成 n张 img 吗？
    rm_n_mkdir(osp.join(dest_path, "inst"))
    rm_n_mkdir(osp.join(dest_path, "seg_mask"))
    rm_n_mkdir(osp.join(dest_path, "imgs"))

    for _, d in tqdm(enumerate(dirs)):
        filename = d.split("/")[-1]
        print(filename)
        xml_files = glob.glob(osp.join(d, "*.xml"))
        # 每个病人/--生成一个文件夹;每张图片 mask 为文件
        for idx, xml in enumerate(xml_files):
            basename = osp.basename(xml).strip(".xml")
            # 每个xml file 生成对应 inst_mask, type_mask
            inst_path = osp.join(dest_path, "inst", "{0}.npy".format(basename))
            type_path = osp.join(dest_path, "seg_mask", "{0}.png".format(basename))
            img_path = osp.join(dest_path, "imgs", "{0}.png".format(basename))

            # tif 转换为 png 格式
            img_file = xml.replace("xml", "tif")
            if not os.path.exists(img_file):
                svs_to_tif(img_file.replace("tif", "svs"))
            cv2.imwrite(img_path, cv2.imread(img_file))

            inst_mask, type_mask = read_xml(xml)
            np.save(inst_path, inst_mask)
            # np.save(type_path, type_mask)
            cv2.imwrite(type_path, type_mask)


if __name__ == "__main__":
    # data_path = "/home/pannuke_pre/datasets/original/MoNuSAC/MoNuSAC Testing Data and Annotations"  # Path to read data from
    # dest_path = "/home/pannuke_pre/datasets/original/MoNuSAC/test" # Path to save binary masks corresponding to xml files
    data_path = "/root/autodl-tmp/pannuke_app/datasets/raw/MoNuSAC/train"
    dest_path = "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train"
    convert(
        data_path,
        dest_path,
    )
