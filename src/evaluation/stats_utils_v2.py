import numpy as np
import scipy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .utils import fn_time

torch.cuda.empty_cache()


# -------------------------- Optimised for Speed
def get_fast_aji(true_id_list, true_masks, pred_id_list, pred_masks):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    # prefill with value
    # 假如这个地方， true_id_list , pred_id_list 都不包含 0，背景
    pairwise_inter = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    for true_id in true_id_list:  # 0-th is background
        t_mask = true_masks[true_id].astype(int)
        for pred_id in pred_id_list:
            p_mask = pred_masks[pred_id].astype(int)

            if np.sum(p_mask[t_mask > 0]) == 0:  # 判断是否有重合，没有就go on
                continue
            # if pred_id == 0:  # ignore
            #     continue  # overlaping background

            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id, pred_id] = inter
            pairwise_union[true_id, pred_id] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, don't care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true)  # index to instance ID
    paired_pred = list(paired_pred)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    # except:
    #     print("something is wrong with get_fast_aji")
    #     raise ValueError
    #     # return 0
    return aji_score


#####
def get_fast_aji_plus(true_id_list, true_masks, pred_id_list, pred_masks):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping,
    unlike AJI where a prediction instance can be paired against many GT instances
    (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list:  # 0-th is background
        t_mask = true_masks[true_id].astype(int)
        #         pred_true_overlap = pred[t_mask > 0]
        #         pred_true_overlap_id = np.unique(pred_true_overlap)
        #         pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_id_list:
            p_mask = pred_masks[pred_id].astype(int)

            if np.sum(p_mask[t_mask > 0]) == 0:  # 判断是否有重合，没有就go on
                continue
            # if pred_id == 0:  # ignore
            #     continue  # overlaping background

            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id, pred_id] = inter
            pairwise_union[true_id, pred_id] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    # extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true)  # index to instance ID
    paired_pred = list(paired_pred)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


#####
def get_fast_pq(true_id_list, true_masks, pred_id_list, pred_masks, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list:  # 0-th is background
        t_mask = true_masks[true_id].astype(int)
        for pred_id in pred_id_list:
            p_mask = pred_masks[pred_id].astype(int)

            if np.sum(p_mask[t_mask > 0]) == 0:  # 判断是否有重合，没有就go on
                continue

            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()

            iou = inter / (total - inter)
            pairwise_iou[true_id, pred_id] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        # Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        # extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou])
        paired_pred = list(paired_pred[paired_iou > match_iou])
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    try:
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        # return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true,
        # unpaired_pred]
        return [dq, sq, dq * sq]
    except Exception as e:
        print("dq,sq, pq something wrong!", e)
        # return [0, 0, 0]
        raise ValueError


def get_fast_dice_2(true_id, true_masks, pred_id, pred_masks):
    """Ensemble dice."""
    overall_total = 0
    overall_inter = 0
    # for true_idx in range(len(true_id)):
    for true_idx in true_id:  # 0-th is background
        t_mask = true_masks[true_idx].astype(int)
        for pred_idx in pred_id:
            p_mask = pred_masks[pred_idx].astype(int)

            if np.sum(p_mask[t_mask > 0]) == 0:  # 判断是否有重合，没有就go on
                continue

            # print(true_idx, pred_idx)
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / (overall_total + 1e-6)


####
def get_dice_2(true_id, true_masks, pred_id, pred_masks):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = true_masks[t].astype(int)
        for p in pred_id:
            p_mask = pred_masks[p].astype(int)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / (total_markup + 1e-6)


#####
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


#####
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


@fn_time
def run_nuclei_type_stat(preds, trues, type_uid_list):
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image.
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types

    """
    # 首先， paired_all ,...all 都是针对所有图片来说的
    # true_inst_type_all, pred_inst_type_all , 每个图片有不同的inst_id 表示为不同的inst，但是新图片会从头开始。
    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    length = len(preds)

    print("========开始对图片计算overall分类效果========")
    for file_idx in tqdm(range(length)):
        true_centroid = (trues[file_idx]["centroids"]).astype("float32")
        true_inst_type = trues[file_idx]["labels"][:, None]
        pred_centroid = (preds[file_idx]["centroids"]).astype("float32")
        pred_inst_type = preds[file_idx]["labels"][:, None]

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)

    # TODO 是指的是所有的true_inst_type 么？
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]

    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
            + 1e-6
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [acc_type, f1_d]
    for type_uid in type_uid_list:
        # print("type_uid: ---", type_uid)
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    # print(np.array(results_list))
    return results_list


def eveluate_one_pic_class(
    true_centroid, pred_centroid, true_inst_type, pred_inst_type
):
    """
    输入具体的数值计算;
    借用hover 的代码,
    true, pred 都是 mat格式, 有 inst_centroid, inst_type 等key

    单张图片只计算 acc, f1; overall 再计算 confusion matrix, precision, recall
    """
    paired, unpaired_true, unpaired_pred = pair_coordinates(
        true_centroid, pred_centroid, 12
    )
    paired_pred_type = pred_inst_type[paired[:, 1]]
    paired_true_type = true_inst_type[paired[:, 0]]

    unpaired_pred_type = pred_inst_type[unpaired_pred]
    unpaired_true_type = true_inst_type[unpaired_true]

    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]
    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    acc = tp_tn_dt / (
        tp_tn_dt + fp_fn_dt + 1e-6
    )  # 只考虑已经匹配的预测情况。预测为True的数据中有多少是正确预测的。实际上是 Precision 值么？
    f1 = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d + 1e-6)  # 这个计算的都只是实例匹配层面的结果。
    return acc, f1


@fn_time
def eveluate_one_pic_inst(true_masks, pred_masks):
    if isinstance(true_masks, dict):
        true_masks = true_masks["masks"]
        pred_masks = pred_masks["masks"]
    true_id_list = list(range(len(true_masks)))
    pred_id_list = list(range(len(pred_masks)))
    metrics = []
    # aji = get_fast_aji(true_id_list, true_masks, pred_id_list, pred_masks)
    aji_plus = get_fast_aji_plus(true_id_list, true_masks, pred_id_list, pred_masks)
    # dice = get_dice_2(true_id_list, true_masks, pred_id_list, pred_masks)
    dice = get_fast_dice_2(true_id_list, true_masks, pred_id_list, pred_masks)
    # pq = get_fast_pq(true_id_list, true_masks, pred_id_list, pred_masks)

    metrics.append(dice)
    # metrics.append(aji)
    metrics.append(aji_plus)
    # metrics.append(pq[0])
    # metrics.append(pq[1])
    # metrics.append(pq[2])
    if metrics[0] > 1:
        print("something is wrong with eveluate_one_pic_inst")
        aji_plus = get_fast_aji_plus(true_id_list, true_masks, pred_id_list, pred_masks)
        # dice = get_dice_2(true_id_list, true_masks, pred_id_list, pred_masks)
        dice = get_dice_2(true_id_list, true_masks, pred_id_list, pred_masks)
    return metrics
