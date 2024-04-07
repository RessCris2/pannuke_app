import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.measure import label
from skimage.morphology import (
    dilation,
    disk,
    erosion,
    reconstruction,
    remove_small_objects,
    square,
)
from skimage.segmentation import watershed


def prepare_prob(img, convertuint8=True, inverse=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img


def h_reconstruction_erosion(prob_img, h):
    """
    Performs a H minimma reconstruction via an erosion method.
    """

    def making_top_mask(x, lamb=h):
        return min(255, x + lamb)

    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(seed, mask, method="erosion").astype(np.dtype("ubyte"))
    return recons


def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    img = prepare_prob(img, convertuint8=convertuint8, inverse=inverse)
    recons = h_reconstruction_erosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask == 0] = 0
        return res


def get_contours(img):
    """
    Returns only the contours of the image.
    The image has to be a binary image
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))


def generate_wsl(ws):
    """
    Generates watershed line that correspond to areas of touching objects.
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad


def arrange_label(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background=background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = arrange_label(mat)
    return mat


def dynamic_watershed_alias(p_img, lamb=8, p_thresh=0.9, min_size=100, mode="dist"):
    """
    Applies our dynamic watershed to 2D prob/dist image.
    """
    b_img = (p_img > p_thresh) + 0
    if mode == "prob":
        Probs_inv = prepare_prob(p_img)
    else:
        # 如果预测的是dist， 则不进行 img_as_ubyte 的处理么？
        Probs_inv = p_img

    Hrecons = h_reconstruction_erosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons, mask=b_img)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    ar_label = arrange_label(ws_labels)
    # TODO: test 是否需要加
    ar_label = remove_small_objects(ar_label, min_size=min_size, connectivity=1)
    wsl = generate_wsl(ar_label)
    ar_label[wsl > 0] = 0
    ar_label = arrange_label(ar_label)
    return ar_label


def post_process(prob_image, param=7, thresh=0.5, mode="dist"):
    """
    Perform dynamic_watershed_alias with some default parameters.
    """
    segmentation_mask = dynamic_watershed_alias(prob_image, param, thresh, mode)
    return segmentation_mask


# if __name__ == "__main__":
#     ma = cv2.imread(
#         "/root/autodl-tmp/com_models/DIST/datafolder/TNBC_NucleiSegmentation/GT_01/\
#         01_1.png",
#         0,
#     )
#     mask = img_as_float(ma)
#     post_process(mask)
