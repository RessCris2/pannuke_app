import datetime
import json
import os

import numpy as np

# sys.path.append("/root/autodl-tmp/archive/metrics")
from .data_transformer import get_transformer
from .pycococreatortools import create_annotation_info, create_image_info


def convert_to_coco(dataset_name, data_dir, save_path, test_mode=False):
    """
    Convert experimental datasets to COCO format.

    Args:
        dataset_name:


    Returns:

    """
    # save_path = os.path.join(save_dir, "annotations.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get the corresponding class for the given dataset and initialize the class with the dataset dir.
    transformer = get_transformer(dataset_name)(data_dir)

    CATEGORIES = transformer.category
    INFO = {
        "description": "Dataset in coco format",
        "url": "https://github.com/waspinator/pycococreator",
        "version": "0.1.0",
        "year": 2023,
        "contributor": "weifeiouyang",
        "date_created": datetime.datetime.utcnow().isoformat(" "),
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        }
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    segmentation_id = 1

    exs = []
    # get path of imgs
    images_files = transformer.imgs
    for image_filename in images_files:
        print(image_filename)
        image = transformer.load_img(image_filename)
        image_info = create_image_info(
            image_id, os.path.basename(image_filename), image.shape[:2]
        )
        coco_output["images"].append(image_info)

        # get ann for this img.
        # basename = pathlib.Path(image_filename).stem
        # inst = load_img(opj(INST_DIR, "{}.npy".format(basename)))
        # ann_path = transformer.load_ann_path(image_filename)
        try:
            ann = transformer.load_ann_for_patch(image_filename)
        except:
            ann = transformer.load_ann(image_filename)

        # ann = transformer.load_ann(image_filename)

        inst = ann[:, :, 0]
        inst_ids = np.unique(inst)[1:]
        type_mask = ann[:, :, 1]
        # type_mask = load_img(opj(TYPE_DIR, "{}.png".format(basename)))

        for inst_id in inst_ids:
            # if inst_id == 78:
            #     pass
            binary_mask = np.where(inst == inst_id, 1, 0)
            try:
                # 去除0之后的那个id; 有的图片是只有0，这种图片就直接跳过？
                class_id = np.unique(np.where(inst == inst_id, type_mask, 0))
                if len(class_id) > 2:
                    # raise "you have more than one type in one instance!"
                    ex = [image_filename, inst_id, class_id]
                    exs.append(ex)
                    print(ex)
                # else:
                class_id = class_id[1]
                category_info = {"id": int(class_id), "is_crowd": 0}
                annotation_info = create_annotation_info(
                    segmentation_id,
                    image_id,
                    category_info,
                    binary_mask,
                    binary_mask.shape,
                    tolerance=2,
                )
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
            except Exception as e:
                print("error", e)
                continue

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    # test mode will not save result
    if not test_mode:
        with open(save_path, "w") as output_json_file:
            json.dump(coco_output, output_json_file)
