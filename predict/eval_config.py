# 还要考虑不同函数的传参
model_path_dict={
        "pannuke": {
                "mask_rcnn": { "config_file":'/root/autodl-tmp/archive/v2/models/mask_rcnn/pannuke_maskrcnn/work_dir/mask-rcnn_r50_fpn_1x_coco.py',
                              "checkpoint_file":'/root/autodl-tmp/archive/v2/models/mask_rcnn/pannuke_maskrcnn/work_dir/epoch_12.pth'
                            },
                "mmseg_fcn":{
                            "config_file" : "/root/autodl-tmp/archive/v2/models/mmseg_fcn/pannuke/pannuke_work_dir/pannuke_config.py",
                            "checkpoint_file" : "/root/autodl-tmp/archive/v2/models/mmseg_fcn/pannuke/pannuke_work_dir/iter_40000.pth"
                            },
                "seg_unet":{
                            "model_path" : "/root/autodl-tmp/archive/v2/model_data/seg_unet/pannuke/model_27.pth"
                            },
                "dist": {"save_path" : 
                             "/root/autodl-tmp/archive/v2/model_data/dist/pannuke/epoch_65.pth"},
                "hovernet":{'model_path':
                            '/root/autodl-tmp/archive/v2/model_data/hovernet/pannuke/202305231527/net_epoch=50.tar'
                            }
        },
        "consep": {
                "mask_rcnn": { "config_file" : '/root/autodl-tmp/viax/train/mask_rcnn/consep/consep_config.py',
                            "checkpoint_file" : '/root/autodl-tmp/viax/train/model_data/consep/mask_rcnn/epoch_12.pth'
                            },
                "mmseg_fcn":{
                        "config_file" : "/root/autodl-tmp/archive/v2/models/mmseg_fcn/consep/consep_work_dir/consep_config.py",
                        "checkpoint_file" : "/root/autodl-tmp/archive/v2/models/mmseg_fcn/consep/consep_work_dir/iter_40000.pth"
                    },
                "seg_unet":{
                        "model_path" : "/root/autodl-tmp/archive/v2/model_data/seg_unet/consep/model_10.pth"
                    },
                "dist": {"save_path" : "/root/autodl-tmp/archive/v2/model_data/dist/consep/202305301943/epoch_45.pth"},
                "hovernet":{'model_path':
                            '/root/autodl-tmp/archive/v2/model_data/hovernet/consep/202305292134/01/net_epoch=50.tar'
                            }
        },
        "monusac": {
                "mask_rcnn": { "config_file":'/root/autodl-tmp/archive/v2/model_data/mask_rcnn/monusac/config_monusac.py',
                            "checkpoint_file":'/root/autodl-tmp/archive/v2/model_data/mask_rcnn/monusac/epoch_12.pth'
                            },
                "mmseg_fcn":{
                    "config_file" : "/root/autodl-tmp/archive/v2/models/mmseg_fcn/monusac/monusac_work_dir/monusac_config.py",
                    "checkpoint_file" : "/root/autodl-tmp/archive/v2/models/mmseg_fcn/monusac/monusac_work_dir/iter_40000.pth"
                    },
                "seg_unet":{
                    "model_path" : "/root/autodl-tmp/archive/v2/model_data/seg_unet/monusac/model_2.pth"
                    },
                "dist": {"save_path" : "/root/autodl-tmp/archive/v2/model_data/dist/monusac/202305301817/epoch_95.pth"},
                 "hovernet":{'model_path':
                            '/root/autodl-tmp/archive/v2/model_data/hovernet/monusac/202305240326/01/net_epoch=41.tar'}
        },
    }


# tasks = [
#  {'dataset_name': 'consep','model_name': 'mask_rcnn','predict_fn': 'mask_rcnn_predict_dir','is_enable': 0},
#  {'dataset_name': 'consep','model_name': 'seg_unet','predict_fn': 'seg_predict_dir','is_enable': 0},
#  {'dataset_name': 'consep','model_name': 'mmseg_fcn','predict_fn': 'mmseg_predict_dir','is_enable': 0},
#  {'dataset_name': 'consep','model_name': 'dist','predict_fn': 'dist_predict_dir','is_enable': 0},
#  {'dataset_name': 'consep','model_name': 'hovernet','predict_fn': '——','is_enable': 0},
#  {'dataset_name': 'monusac','model_name': 'mask_rcnn','predict_fn': 'mask_rcnn_predict_dir','is_enable': 0},
#  {'dataset_name': 'monusac','model_name': 'seg_unet','predict_fn': 'seg_predict_dir','is_enable': 0},
#  {'dataset_name': 'monusac', 'model_name': 'mmseg_fcn', 'predict_fn': 'mmseg_predict_dir', 'is_enable': 0},
#  {'dataset_name': 'monusac','model_name': 'dist','predict_fn': 'dist_predict_dir','is_enable': 0},
#  {'dataset_name': 'monusac','model_name': 'hovernet','predict_fn': '——','is_enable': 0},
#  {'dataset_name': 'pannuke', 'model_name': 'mask_rcnn', 'predict_fn': 'mask_rcnn_predict_dir', 'is_enable': 1},
#  {'dataset_name': 'pannuke', 'model_name': 'seg_unet', 'predict_fn': 'seg_predict_dir', 'is_enable': 1},
#  {'dataset_name': 'pannuke', 'model_name': 'mmseg_fcn', 'predict_fn': 'mmseg_predict_dir', 'is_enable': 1},
#  {'dataset_name': 'pannuke', 'model_name': 'dist', 'predict_fn': 'dist_predict_dir', 'is_enable': 1},
#  {'dataset_name': 'pannuke', 'model_name': 'hovernet', 'predict_fn': '——', 'is_enable': 0}
# ]

tasks = [
#  {'dataset_name': 'consep','model_name': 'mask_rcnn'},
#  {'dataset_name': 'consep','model_name': 'seg_unet',},
#  {'dataset_name': 'consep','model_name': 'mmseg_fcn'},
#  {'dataset_name': 'consep','model_name': 'dist'},
#  {'dataset_name': 'consep','model_name': 'hovernet','predict_fn': '——'},
#  {'dataset_name': 'monusac','model_name': 'mask_rcnn'},
#  {'dataset_name': 'monusac','model_name': 'seg_unet'},
#  {'dataset_name': 'monusac', 'model_name': 'mmseg_fcn'},
#  {'dataset_name': 'monusac','model_name': 'dist'},
#  {'dataset_name': 'monusac','model_name': 'hovernet'},
#  {'dataset_name': 'pannuke', 'model_name': 'mask_rcnn' },
#  {'dataset_name': 'pannuke', 'model_name': 'seg_unet' },
#  {'dataset_name': 'pannuke', 'model_name': 'mmseg_fcn'},
#  {'dataset_name': 'pannuke', 'model_name': 'hovernet' },
 {'dataset_name': 'pannuke','model_name': 'dist'}
]