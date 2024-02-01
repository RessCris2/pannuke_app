"""记录每个实验的结果数据路径？
/root/autodl-tmp/pannuke_app/projects/patched_monusac/hovernet/evaluation/hovernet.csv
对原始数据集进行统计
现在需要知道每张图片中细胞核的数量和面积均值？
"""
from pycocotools.coco import COCO
from pycocotools import cocoeval
import pandas as pd
import numpy as np

def load_coco(ann_path):
    """加载数据集 json 文件
    """
    coco = COCO(ann_path)
    return coco


def desc_dataset(coco, dataset, flag):
    """针对一个数据集，返回以下统计值
    - img 名字
    - 细胞核数量
    - 细胞核的平均大小
    - 细胞核的面积和
    - 平均数量=数量/面积和
    - 细胞核面积占比
    
    """
    data = []

    for _, img in coco.imgs.items():
        annIds = coco.getAnnIds(imgIds=img['id'])
        cnt = len(annIds)

        if cnt > 0:
            anns = coco.loadAnns(annIds)
            area = np.mean([ann['area'] for ann in anns])
            area_sum = np.sum([ann['area'] for ann in anns])
            avg_cnt = cnt / area_sum
            area_busy = area_sum / (img['width'] * img['height'])

            data.append({
                'imgs': img['file_name'],
                'cnts': cnt,
                'areas': area,
                'avg_cnts': avg_cnt,
                'area_sums': area_sum,
                'area_busy_rate': area_busy
            })
            
            

    # 创建 DataFrame
    df = pd.DataFrame.from_records(data)
    df['dataset']= dataset
    df['flag'] = flag
    return df

def get_desc_res():
    dataset_names =['consep', 'monusac','pannuke']
    datasets = ['CoNSeP','MoNuSAC','PanNuke']
    flags=['train', 'test']

    dfs = []

    # 在循环外部加载COCO数据集
    for dataset in datasets:
        for flag in flags:
            coco = load_coco(f"{root_dir}/datasets/processed/{dataset}/{flag}/{flag}_annotations.json")
            df = desc_dataset(coco, dataset, flag)
            dfs.append(df)

    # 使用 extend 避免创建新的 DataFrame
    res = pd.concat(dfs, ignore_index=True)
    return res


## 给三个指标绘图
res = pd.read_csv("data/desc_res.csv")
sns.boxplot(data=res, x='dataset',y='area_busy_rate')
plt.ylabel('The proportion of nucleus area per image')
plt.xlabel(None)

# pannuke 细胞核数量密度中等
sns.boxplot(data=res,x='dataset', y='avg_cnts')
plt.ylim((0, 0.01))
plt.xlabel(None)

# pannuke 细胞核平均面积中等，consep则偏小
sns.boxplot(data=res,x='dataset', y='areas')
plt.ylim((0, 3000))
plt.xlabel(None)