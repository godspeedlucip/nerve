import numpy as np
import os
join = os.path.join
import shutil
import nibabel as nib
import random
from tqdm import tqdm
import json
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    src_data_path = args.input
    dest_data_path = args.output
    
    dest_label_path = join(dest_data_path, 'labelsTs')
    dest_image_path = join(dest_data_path, 'imagesTs')

    os.makedirs(dest_label_path, exist_ok=True)
    os.makedirs(dest_image_path, exist_ok=True)

    # 移动label
    for i in tqdm(os.listdir(join(src_data_path,'vol_gt'))): #mr_001_sample_00_0-25.tif
        i_name = i.replace('_','-')
        shutil.copy(join(src_data_path,'vol_gt',i), join(dest_label_path, i_name))

    # 移动image
    for i in tqdm(os.listdir(join(src_data_path,'vol_train_no_masked'))): #mr_001_sample_00_0-25.tif
        i_name = i.replace('_','-')
        shutil.copy(join(src_data_path,'vol_train_no_masked',i), join(dest_image_path, i_name.split('.')[0]+'_0000.nii.gz'))

    # 再创建一个总的json文件
    numTraining = len(os.listdir(join(src_data_path,'vol_gt')))
    to_be_written = {
        "channel_names":{
            "0": "zscore" #尝试一下z_score归一化
        },
        "labels":{
            "background": 0,
            "vessel": 1
        },
        "numTraining": numTraining,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }

    top_json_save_path = join(dest_data_path, "dataset.json")
    with open(top_json_save_path, 'w') as f:
        json.dump(to_be_written, f, indent=4)
