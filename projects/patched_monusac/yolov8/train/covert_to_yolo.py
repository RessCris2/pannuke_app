from ultralytics.data.converter import convert_coco

convert_coco(
    "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/train/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)


convert_coco(
    "/root/autodl-tmp/pannuke_app/projects/monusac/training_data/test/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)
