from scipy.spatial.distance import cdist
import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import seaborn as sns

def find_edge(per_inst):
    contours, hierarchy = cv2.findContours(per_inst.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    edge = contours[0]
    try:
      assert len(edge) > 4
    except:
        edge=None
    return edge

def compute_inst(inst_label_path):
    """对每个inst计算 area, dist 
       然后取平均得到每个image 的结果
       输入是
       inst: inst 格式的标签数据
       inst_ids: inst_id 所有取值
    """
    inst = np.load(inst_label_path)
    inst_ids = list(np.unique(inst))[1:]
    cnt = 0
    
    edges = []
    areas = []
    for inst_id in inst_ids:
        per_inst = np.where(inst == inst_id, 1, 0)
        area = per_inst.sum()
        edge = find_edge(per_inst)

        # 过滤小面积或者找不到edge的情况
        if area < 30 or edge is None:
          continue
        else:
          cnt += 1
          areas.append(area)
          edges.append(edge.squeeze(1))

    n = len(edges)
    if n == 0:
       # 没有实例的情况
       return 0,0,0
    dist_matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = cdist(edges[i],edges[j],metric='euclidean')
            # 得到每个实例与其他实例的最小距离
            dist_matrix[i,j] = dist_matrix[j,i] = d.min() 
        dist_matrix[i,i] = 1000

    # 得到每个实例与最近的实例的值,再求平均
    try:
      dist = dist_matrix.min(axis=0).mean()
    except:
      print('xxx')
    area = np.mean(areas)
    return dist, area, cnt

def compute_image(dir_root):
#     dir_root = "/root/autodl-tmp/datasets/consep/inst/test"
    files = glob.glob(f"{dir_root}/*")
    # df = pd.DataFrame(columns=['file_name','dist','area','cnt'])
    df = []
    for inst_label_path in files:
        file_name = pathlib.Path(inst_label_path).stem
        print(file_name)
        dist, area, cnt = compute_inst(inst_label_path)
        df.append((file_name, dist, area, cnt))
    res = pd.DataFrame(df, columns=['file_name','dist','area','cnt'])
    res['filter'] = res['dist'] + res['area'] + res['cnt']
    return res.query("filter > 0")[['file_name','dist','area','cnt']]
        


if  __name__ == "__main__":
    dataset_name = 'consep'
    flag = 'train'
    dir_root = f"/root/autodl-tmp/viax/datasets/{dataset_name}/inst/{flag}"
    res = compute_image(dir_root)
    print('xxx')
    res.to_csv(f"root/autodl-tmp/viax/visualize/scales/{dataset_name}_{flag}")