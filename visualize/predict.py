import sys
sys.path.append("/root/autodl-tmp/viax")
from src.core import infer_base
from src.core.eval_config import model_path_dict
import numpy as np

# seg 
img_path = "/root/autodl-tmp/viax/datasets/consep/images/train/train_1_000.jpg"
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
print("xxx")

# plot_gt_overlay()
# dist: 只有 inst_map, type_map 也许可以初始化为全1，看后续处理
inst_map = infer_base.dist_predict_oneimg_for_plot(img_path, dataset_name,  model_name, model_path_dict)
type_map = np.ones_like(inst_map)

# hover
dataset_name = "consep"
model_name = "hovernet"
model_path_dict={
        "consep": {
                "hovernet":{
                        "model_path" : "/root/autodl-tmp/archive/v2/model_data/hovernet/consep/202305292134/01/net_epoch=50.tar"
                    },
        },
        
    }

pred, _ = infer_base.hovernet_predict_dir(img_path, dataset_name,  model_name, model_path_dict)
## predict 的结果需要转换为 type_map, inst_map


# maskrcnn

pred, _ = infer_base.mask_rcnn_predict_oneimg(img_path, dataset_name,  model_name, model_path_dict)
## predict 的结果需要转换为 type_map, inst_map
