from ultralytics.data.converter import convert_coco

convert_coco(
    "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/traing_data/train/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)


convert_coco(
    "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/traing_data/test/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)
