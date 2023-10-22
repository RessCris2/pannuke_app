"""绘制每个模型的预测结果到图上
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import seaborn as sns
import sys
sys.path.append("/root/autodl-tmp/viax")
from src.core import infer_base
from src.hovernet.misc import viz_utils # , random_colors
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def add_legend(inst_rng_colors):
    """inst_rng_colors 为 0,1 之间的值
    """
    # 定义一组颜色（RGB 格式）
    colors_rgb = inst_rng_colors  # 分别表示红色、绿色、蓝色、紫色
    # 创建一个空白图形
    fig, ax = plt.subplots()
    # 在图形中添加示例矩形，每个矩形对应一个颜色
    legend_elements = [Patch(facecolor=color, label=f'colors {i+1}') for i, color in enumerate(colors_rgb)]
    # 创建图例并将示例矩形添加到图例中
    ax.legend(handles=legend_elements)
    # 可选：将图例放置在合适的位置
    ax.legend(handles=legend_elements, loc='upper right')

    # 显示图形
    plt.show()


def plot_original():
    x = Image.open('/root/autodl-tmp/viax/datasets/consep/images/train/train_1_000.jpg')
    x.save('consep.png',format='png', dpi=(600,600), compress_level=0)

def plot_gt_overlay(img_path="/root/autodl-tmp/viax/datasets/consep/images/train/train_1_000.jpg"):
    """关键是指定特定类别对应的颜色
    """
    input_image =  cv2.imread(img_path)
    inst_map = np.load(img_path.replace('jpg', 'npy').replace('images', 'inst')) #"/root/autodl-tmp/viax/datasets/consep/inst/train/train_1_000.npy")
    type_mask = cv2.imread(img_path.replace('jpg', 'png').replace('images', 'seg_mask')) # "/root/autodl-tmp/viax/datasets/consep/seg_mask/train/train_1_000.png")

    num_classes = 4
    float_colors = viz_utils.random_colors(num_classes) # 这个颜色话 legend
    inst_rng_colors = np.array(float_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)
    colors = [(x).tolist() for x in inst_rng_colors]
    overlay = viz_utils.visualize_instances_map(input_image, inst_map, type_mask, type_colour=colors)

    add_legend(float_colors)
    
    x = Image.fromarray(overlay)
    x.save('consep_gt.png',format='png', dpi=(600,600), compress_level=0) # 记得修改保存地址
    return overlay, colors, float_colors


def plot_pred_overlay(img_path):
    """关注pred的格式"""
    input_image = cv2.imread(img_path)
    dataset_name = "consep"
    model_name = "seg_unet"
    model_path_dict={
            "consep": {
                    "seg_unet":{
                            "model_path" : "/root/autodl-tmp/viax/train/model_data/consep/seg_unet/202309050053/model_25.pth"
                        },
            },
        }
    type_map, inst_map = infer_base.seg_predict_oneimg_for_plot(img_path, dataset_name,  model_name, model_path_dict)

    num_classes = 5
    float_colors = viz_utils.random_colors(num_classes) # 这个颜色话 legend
    inst_rng_colors = np.array(float_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)
    colors = [(x).tolist() for x in inst_rng_colors]
    overlay = viz_utils.visualize_instances_map(input_image, inst_map, type_map, type_colour=colors)

    add_legend(float_colors)
    
    x = Image.fromarray(overlay)
    x.save('consep_unet_pred.png',format='png', dpi=(600,600), compress_level=0) # 记得修改保存地址
    return overlay, colors, float_colors





if __name__ == "__main__":
    img_path = "/root/autodl-tmp/viax/datasets/consep/images/train/train_1_000.jpg"
    plot_pred_overlay(img_path)
    # plot_gt_overlay(img_path)
