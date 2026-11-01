import numpy as np
import os
join = os.path.join
import shutil
import nibabel as nib
import random
from tqdm import tqdm
import json


if __name__=='__main__':

    ## 创建符合nnUNet格式的数据集
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range",type=int, default=60)
    parser.add_argument("--tar_num", type=int, required=True)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--dataset_num", type=int, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--data_name", type=str, default='costa')# 0,1
    args = parser.parse_args()
    
    data_dir = '/data1/liuyang/stent/nnUNet/data/nnUNet_raw'
    radius = args.radius
    tar_range = args.tar_range
    tar_num = args.tar_num
    dataset_num = args.dataset_num
    resolution = args.resolution
    data_name = args.data_name
    to_train = args.train
    to_test = args.test


    assert to_train in set([0,1])
    assert to_test in set([0,1])

    # for i in ['train', 'test']:
    to_process = []
    if(to_train):
      to_process.append('train')
    if(to_test):
      to_process.append('test')

    for i in to_process:
        src_data_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resolution}/trun_data_02/radius-{radius}/{i}_data/{tar_range}_{tar_num}-fdkNoFilter'
        dest_data_path = f"/data1/liuyang/stent/nnUNet/data/nnUNet_raw/Dataset{dataset_num}_{data_name}-radius{radius}-{tar_range}-{tar_num}"

        if(i=='train'):
            dest_label_path = join(dest_data_path, 'labelsTr')
            dest_image_path = join(dest_data_path, 'imagesTr')
        else:
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
        
        if(i=='train'):
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