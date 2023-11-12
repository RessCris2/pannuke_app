"""
    pannuke dataset 的mask 转变为 inst 标注的数据; seg_mask
"""
import os
import os.path as osp

import cv2
import numpy as np


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def pn2inst_type(pn_mask_path, save_dir, prefix="train"):
    """
    np.max(mask0) = 3512; 数据实际处理过程中，发现 inst id 有重复的情况，所以需要重新标注 inst id;
    处理方式为每一层加 i * 10000, i = 1, 2, 3, 4, 5, 6; 之后再对每层本该为 0 的地方，赋值为 0;
    """
    # path = osp.join(pn_dir, "masks.npy")
    inst_dir = osp.join(save_dir, "inst")
    seg_mask_dir = osp.join(save_dir, "seg_mask")
    os.makedirs(inst_dir, exist_ok=True)
    os.makedirs(seg_mask_dir, exist_ok=True)

    mask0 = np.load(pn_mask_path)
    for idx in range(len(mask0)):
        mask = mask0[idx]
        inst_ids = []
        for i in range(5):
            inst_id = np.unique(mask[..., i]).astype(int)[1:]
            inst_ids.extend(inst_id)

        # binary_masks = np.where(mask[..., :-1] > 0, 1, 0)
        # binary_mask = binary_masks.sum(axis=-1)

        try:
            assert len(inst_ids) == len(
                set(inst_ids)
            ), "inst ids have duplicate numbers!"
            # assert (
            #     len(np.unique(binary_mask)) <= 2
            # ), "more than one inst in the same area!"
        except Exception as e:
            print("inst ids have duplicate numbers!", e)
            # if have duplicate numbers, then remap the inst ids
            mats = np.stack(
                [np.ones(shape=(256, 256)) * 10000 * i for i in range(1, 7)]
            )
            transformed_mask = mask + np.transpose(mats, (1, 2, 0))
            mats = []
            for i in range(1, 7):
                matrix = transformed_mask[..., i - 1]
                new = np.where(matrix == i * 10000, 0, matrix)
                mats.append(new)
            mask = np.transpose(np.stack(mats), (1, 2, 0))
            assert len(np.unique(mask[..., :-1])) == len(inst_ids) + 1
            # print(inst_ids)
            # continue

        inst_mask = mask[..., :-1].sum(axis=-1)
        relabeled_inst_mask = remap_label(inst_mask, by_size=True)
        inst_path = f"{inst_dir}/{prefix}_{idx}.npy"
        np.save(inst_path, relabeled_inst_mask)

        type_mask = mask[..., [5, 0, 1, 2, 3, 4]].argmax(axis=-1)
        seg_mask_path = f"{seg_mask_dir}/{prefix}_{idx}.npy"
        np.save(seg_mask_path, type_mask)


def pn2img(pn_path, save_dir, prefix="train"):
    """
    params:
        pn_dir: directory of pannuke images
        save_dir: directory to save images of the form .png
    returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    images = np.load(pn_path)
    for i in range(len(images)):
        image = images[i]
        save_path = osp.join(save_dir, "{}_{}.png".format(prefix, i))
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    # 将图像转换为 png 格式
    prefix = "train"
    pn_path = "/root/autodl-tmp/pannuke_pre/datasets/pannuke/fold1/Fold 1/images/fold1/images.npy"
    save_dir = f"/root/autodl-tmp/pannuke_pre/datasets/pannuke/{prefix}/imgs"
    pn2img(pn_path, save_dir, prefix)

    # 生成 inst, seg_mask
    # pn_mask_path = (
    #     "/root/autodl-tmp/pannuke_pre/datasets/pannuke/train/masks/fold1/masks.npy"
    # )
    # save_dir = "/root/autodl-tmp/pannuke_pre/datasets/pannuke/train/"
    # prefix = "train"
    # pn2inst_type(pn_mask_path, save_dir, prefix)

    # pn_mask_path = (
    #     "/root/autodl-tmp/pannuke_pre/datasets/pannuke/test/masks/fold3/masks.npy"
    # )
    # save_dir = "/root/autodl-tmp/pannuke_pre/datasets/pannuke/test/"
    # prefix = "test"
    # pn2inst_type(pn_mask_path, save_dir, prefix)

    # pn_mask_path = (
    #     "/root/autodl-tmp/pannuke_pre/datasets/pannuke/test/masks/fold1/masks.npy"
    # )
    # save_dir = "/root/autodl-tmp/pannuke_pre/datasets/pannuke/test/"
    # prefix = "test"
    # pn2inst_type(pn_mask_path, save_dir, prefix)
