from ultralytics.data.converter import convert_coco

convert_coco(
    "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/train/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)


convert_coco(
    "/root/autodl-tmp/pannuke_app/datasets/processed/MoNuSAC/test/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)
