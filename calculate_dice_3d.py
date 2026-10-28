import numpy as np
import os
join = os.path.join
import shutil
import nibabel as nib
import random
from tqdm import tqdm
import tifffile as tiff
import pandas as pd
import numpy as np
from medpy.metric.binary import hd95, asd #确实都是直接用这个来计算的

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)


def main(gt_label_path, pred_label_path, save_csv_path='./'):
    gt_list = [i for i in os.listdir(gt_label_path) if i.endswith(".nii.gz")]
    all_dice = 0
    all_hd95 = 0
    all_asd = 0

    # print('start: ')

    results = []

    for j in tqdm(sorted(gt_list)):
        j_pred = j.replace('_','-')
        if(not os.path.exists(join(pred_label_path, j_pred))):
            continue
        gt_nib = nib.load(join(gt_label_path, j))
        gt = np.round(gt_nib.get_fdata()).astype(np.int16)

        pred = np.round(nib.load(join(pred_label_path, j_pred)).get_fdata()).astype(np.int16)

        sin_dice = dice_score(pred, gt)
        all_dice += sin_dice
        
        # hd_score = hd95(pred, gt, voxelspacing=voxel_space)
        # asd_score = asd(pred, gt, voxelspacing=voxel_space)
        hd_score = hd95(pred, gt)
        # hd_score = 0
        all_hd95 += hd_score

        results.append({
            "Case": j,
            "Dice": sin_dice,
            "HD95": hd_score
        })
    if(save_csv_path!=None):
        df = pd.DataFrame(results)
        mean_results = {
            "Case": "AVERAGE",
            "Dice": df["Dice"].mean(),
            "HD95": df["HD95"].mean()
        }
        df = pd.concat([df, pd.DataFrame([mean_results])], ignore_index=True)
        df.to_csv(save_csv_path, index=False)

    print(pred_label_path,' hd95:  ', all_hd95 / len(gt_list))
    print(pred_label_path,' asd:  ', all_asd / len(gt_list))
    print(pred_label_path,' dice:  ', all_dice / len(gt_list))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--pred_path",type=str, required=True)
    parser.add_argument("--gt_path",type=str, required=True)
    parser.add_argument("--save_csv_path",type=str, default=None)
    args = parser.parse_args()
    pred_label_path = args.pred_path
    gt_label_path = args.gt_path
    save_csv_path = args.save_csv_path

    main(gt_label_path, pred_label_path, save_csv_path)


