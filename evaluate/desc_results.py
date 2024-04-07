"""记录每个实验的结果数据路径？
/root/autodl-tmp/pannuke_app/projects/patched_monusac/hovernet/evaluation/hovernet.csv
对原始数据集进行统计
现在需要知道每张图片中细胞核的数量和面积均值？
"""
from pycocotools.coco import COCO
from pycocotools import cocoeval
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_stat():
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
    
    
def segment_df(df, columns):
    """将columns分别划分为三段
    df_segmented = segment_df(df, columns=['areas','avg_cnts','area_busy_rate'])
    """
    for col in columns:
        quantile_bin_labels = [f'{col}_Q1', f'{col}_Q2', f'{col}_Q3']
        df[f'{col}_segmented'] = pd.qcut(df[col], 3, labels=quantile_bin_labels)
    return df


import matplotlib.pyplot as plt

# Define a function to plot the distribution comparison for a segmented metric
def plot_segmented_distribution(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, hue='dataset', palette='Set2')
    plt.title(title)
    plt.xlabel(column.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()

#     # Plot the distribution comparison for 'areas_segmented'
#     plot_segmented_distribution(df, 'areas_segmented', 'Distribution of Area Segmentation Across Datasets')

#     # Plot the distribution comparison for 'avg_cnts_segmented'
#     plot_segmented_distribution(df, 'avg_cnts_segmented', 'Distribution of Average Counts Segmentation Across Datasets')

#     # Plot the distribution comparison for 'area_busy_rate_segmented'
#     plot_segmented_distribution(df, 'area_busy_rate_segmented', 'Distribution of Area Busy Rate Segmentation Across Datasets')


# Function to plot all three metrics for each dataset in one figure with subplots
def plot_all_metrics_per_dataset(df, metrics, dataset,save_path):
    plt.figure(figsize=(18, 6))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        subset = df[df['dataset'] == dataset]
        order = sorted(subset[metric].unique())  # Sort the x labels
        sns.countplot(data=subset, x=metric, order=order, palette='Set2')
        plt.title(f'{dataset} - {metric.replace("_", " ").title()}')
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    plt.save(save_path)

# # Metrics to plot
# metrics = ['areas_segmented', 'avg_cnts_segmented', 'area_busy_rate_segmented']

# # Plot for each dataset
# for dataset in df['dataset'].unique():
#     plot_all_metrics_per_dataset(df, metrics, dataset)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_metrics_for_all_datasets(df, metrics, datasets, save_path):
    # 设置整个大图的尺寸
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(18, 6 * len(datasets)), dpi=300)

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            subset = df[df['dataset'] == dataset]
            order = sorted(subset[metric].unique())  # 排序标签
            # 绘制子图
            sns.countplot(data=subset, x=metric, order=order, palette='Set2', ax=axes[i, j])
            axes[i, j].set_title(f'{dataset} - {metric.replace("_", " ").title()}')
#             axes[i, j].set_xlabel(metric.replace('_', ' ').title())
            axes[i, j].set_ylabel('Count')
            axes[i, j].set_xlabel(None)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# # Metrics and datasets
# metrics = ['areas_segmented', 'avg_cnts_segmented', 'area_busy_rate_segmented']
# datasets = df['dataset'].unique()

# # Save path for the combined figure
# save_path = 'all_datasets_q_dist.pdf'

# # Generate and save the combined plot
# plot_all_metrics_for_all_datasets(df, metrics, datasets, save_path)

def get_desc_cate():
    import pandas as pd
    dataset_names =['consep', 'monusac','pannuke']
    datasets = ['CoNSeP','MoNuSAC','PanNuke']
    flags=['train', 'test']
    # 准备数据列表
    data = []
    for dataset in datasets:
        for flag in flags:
            coco = load_coco(f"{root_dir}/datasets/processed/{dataset}/{flag}/{flag}_annotations.json")
            df = desc_dataset(coco, dataset, flag)

            # 假设 coco 是您已经加载的 COCO 数据集对象
            # cates 是一个字典，包含了类别的信息
            cates = coco.cats  # 获取类别信息



            # 遍历所有类别，获取每个类别的注解数量
            for cate_id, cate_info in cates.items():
                cate_name = cate_info['name']  # 获取类别名称
                ann_ids = coco.getAnnIds(catIds=cate_id)  # 获取当前类别的所有注解ID
                cnt = len(ann_ids)  # 计算注解数量
                data.append([dataset, flag, cate_name, cnt])  # 将数据添加到列表中

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['dataset','flag', 'cate', 'cnt'])

    # 显示DataFrame
    print(df.head())  # 显示前几行以验证结果

    
def gen_eval_df():
    dfs = []
    for index, row in config.iterrows():
        path = row['paths']
        if path == 'None':
            continue
        path = os.path.join('/root', path) if not path.startswith("/root") else path
        df = pd.read_csv(path)
        df.rename({'Unnamed: 0':'pic_name', '0':'map','1':'map50'}, axis=1, inplace=True)
        df['projects'] = row['projects']
        df['map_dataset']= row['map_dataset']
        df['models']= row['models']
        dfs.append(df)
    eval_df = pd.concat(dfs)
    


def result_compare_fig_1():
    """dataset 为横轴，不同模型表现为纵轴。
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 假设你的数据存储在一个名为 eval_df 的 DataFrame 中

    # 创建一个柱状图
    plt.figure(figsize=(10, 6))

    # 使用 seaborn 来绘制柱状图，x 轴是数据集，y 轴是 map 分数，hue 是模型
    sns.barplot(x='map_dataset', y='map', hue='models', data=eval_df)

    # 添加标题和标签
    plt.title('Comparison of MAP Scores by Dataset and Model')
    plt.xlabel('Dataset')
    plt.ylabel('MAP Score')

    # 旋转 x 轴标签以提高可读性
    plt.xticks(rotation=45)

    # 显示图例
    plt.legend(title='Model', loc='upper right')

    # 保存图表为图片文件
    # plt.savefig('/mnt/data/map_comparison.png')
    plt.tight_layout()
    # 显示图表
    plt.show()
