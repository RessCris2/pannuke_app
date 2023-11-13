_base_ = [
    "mask-rcnn_r50_fpn.py",
    "coco_instance.py",
    "schedule_1x.py",
    "default_runtime.py",
]

# classes = ('Neoplastic','Inflammatory','Connective','Dead','Epithelial')
# palette = [
#         (220, 20, 60),(20, 20, 160),(120, 120, 60),(20, 120, 160),
#     ]
# visualization = _base_.default_hooks.visualization
# visualization.update(dict(draw=True, show=False))
