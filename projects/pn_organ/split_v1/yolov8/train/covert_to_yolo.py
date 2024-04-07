from ultralytics.data.converter import convert_coco

root_dir = "/root/autodl-tmp/pannuke_app/projects/pn_organ/split_v1/training_data"
convert_coco(
    f"{root_dir}/train/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)


convert_coco(
    f"{root_dir}/test/",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False,
)

""""
mkdir split_v1/yolov8/train/coco_converted/labels/test/
mkdir split_v1/yolov8/train/coco_converted/labels/train/
mv split_v1/yolov8/train/coco_converted2/labels/test_annotations/* split_v1/yolov8/train/coco_converted/labels/test/
mv split_v1/yolov8/train/coco_converted/labels/train_annotations/* split_v1/yolov8/train/coco_converted/labels/train/
rm -rf mv split_v1/yolov8/train/coco_converted2/
rm -rf  split_v1/yolov8/train/coco_converted/labels/train_annotations/

mkdir split_v1/yolov8/train/coco_converted/images/train/
mkdir split_v1/yolov8/train/coco_converted/images/test/
cp -r split_v1/training_data/train/imgs/* split_v1/yolov8/train/coco_converted/images/train/
cp -r split_v1/training_data/test/imgs/* split_v1/yolov8/train/coco_converted/images/test/
"""