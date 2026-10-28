import numpy as np
import os
join = os.path.join
import shutil
import nibabel as nib
import random
from tqdm import tqdm
import tifffile as tiff

import numpy as np
from medpy.metric.binary import hd95, asd #确实都是直接用这个来计算的

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)

def main(gt_label_path, pred_label_path):
    label_list = [i for i in os.listdir(pred_label_path) if i.endswith(".tif")]
    all_dice = 0
    all_hd95 = 0
    all_asd = 0

    # print('start: ')

    for j in tqdm(sorted(label_list)):
        gt = np.round(tiff.imread(join(gt_label_path, j))).astype(np.int16)
        pred = np.round(tiff.imread(join(pred_label_path, j))).astype(np.int16)

        sin_dice = dice_score(pred, gt)
        all_dice += sin_dice
        
        # # print(j, np.max(gt), np.max(pred))
        # hd_score = hd95(pred, gt, voxelspacing=voxel_space)
        # asd_score = asd(pred, gt, voxelspacing=voxel_space)
        # all_hd95 += hd_score
        # all_asd += asd_score

    print(pred_label_path,' hd95:  ', all_hd95 / len(label_list))
    print(pred_label_path,' asd:  ', all_asd / len(label_list))
    print(pred_label_path,' dice:  ', all_dice / len(label_list))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--pred_path",type=str, required=True)
    parser.add_argument("--gt_path",type=str, required=True)
    args = parser.parse_args()
    pred_label_path = args.pred_path
    gt_label_path = args.gt_path

    main(gt_label_path, pred_label_path)


