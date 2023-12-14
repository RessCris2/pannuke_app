import sys

import numpy as np

sys.path.append("/root/autodl-tmp/pannuke_app/")
import pandas as pd
from src.evaluation.stats_utils import get_fast_aji_plus as plus_2
from src.evaluation.stats_utils_v2 import eveluate_one_pic_inst, get_fast_aji_plus
from src.models.hover.compute_stats import run_nuclei_inst_stat

true_masks = np.load("true_masks.npy")
pred_masks = np.load("pred_masks.npy")
true_id_list = list(range(len(true_masks)))
pred_id_list = list(range(len(pred_masks)))
score1 = get_fast_aji_plus(true_id_list, true_masks, pred_id_list, pred_masks)
