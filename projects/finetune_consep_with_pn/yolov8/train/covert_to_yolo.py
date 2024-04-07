from ultralytics.data.converter import convert_coco

convert_coco(
    "/root/autodl-tmp/pannuke_app/projects/patched_consep/training_data/train",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)


convert_coco(
    "/root/autodl-tmp/pannuke_app/projects/patched_consep/training_data/test",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)


# root_dir = ""
# cp  /root/autodl-tmp/pannuke_app/projects/patched_consep/training_data/train/imgs/*
# /root/autodl-tmp/pannuke_app/projects/finetune_consep_with_pn/yolov8/train/coco_converted/images/train


# /root/autodl-tmp/pannuke_app/projects/finetune_consep_with_pn/yolov8/train/coco_converted/images/test

