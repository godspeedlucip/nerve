import numpy as np
import os
join = os.path.join
import shutil
import nibabel as nib
import random
from tqdm import tqdm
import json
from utils.file_rw import read_sp
import tifffile as tiff


if __name__=='__main__':
    ## 创建符合nnUNet格式的数据集
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_type", type=str, default='stent')
    parser.add_argument("--data_name", type=str, default='costa')
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--dataset_num", type=int, required=True)
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--radius", type=int, default=64) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--tar_range", type=int, default=60)# 0,1
    parser.add_argument("--tar_num", type=int, default=3)# 0,1
    parser.add_argument("--fdk_or_not", type=int, default=False)# 0,1
    args = parser.parse_args()
    
    src_data_dir = '/data1/liuyang/stent/DTE/data/resample_data'
    resolution=args.resolution
    dataset_num = args.dataset_num
    data_type = args.data_type
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    radius = args.radius
    tar_range = args.tar_range
    tar_num = args.tar_num
    fdk_or_not = args.fdk_or_not
    fdk_or_not = 'Yes' if fdk_or_not else 'No'

    assert data_type in set(['stent', 'vessel'])
    assert to_train in set([0,1])
    assert to_test in set([0,1])

    # for i in ['train', 'test']:
    to_process = []
    if(to_train):
      to_process.append('train')
    if(to_test):
      to_process.append('test')

    for mode in to_process:
      src_data_path = join(src_data_dir, f'{data_name}/voxel_{resolution}/dtr/recon/{data_type}/radius-{radius}/{mode}_data/{tar_range}_{tar_num}-fdk{fdk_or_not}Filter')
      dest_data_path = f"/data1/liuyang/stent/nnUNet/data/nnUNet_raw/Dataset{dataset_num}_recon-{data_name}-{data_type}-radius{radius}-{resolution}-{tar_range}-{tar_num}-fdk{fdk_or_not}"

      if(mode=='train'):
        dest_label_path = join(dest_data_path, 'labelsTr')
        dest_image_path = join(dest_data_path, 'imagesTr')
      else:
        dest_label_path = join(dest_data_path, 'labelsTs')
        dest_image_path = join(dest_data_path, 'imagesTs')
      os.makedirs(dest_label_path, exist_ok=True)
      os.makedirs(dest_image_path, exist_ok=True)

      # 移动label
      label_path = join(src_data_path,'vol_gt')
      for i in tqdm(os.listdir(label_path)):
        save_name = i.replace('_','-')
        shutil.copy(join(label_path, i), join(dest_label_path, save_name))

      # 移动image
      image_path = join(src_data_path,'vol_train_no_masked')
      for i in tqdm(os.listdir(image_path)):
        save_name = (i.replace('_','-')).split('.nii.gz')[0]+'_0000.nii.gz'
        shutil.copy(join(image_path, i), join(dest_image_path, save_name))

      # 再创建一个总的json文件
      if(mode=='train'):
        numTraining = len(os.listdir(dest_image_path))
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
            "overwrite_image_reader_writer": "NibabelIO"
        }

        top_json_save_path = join(dest_data_path, "dataset.json")
        with open(top_json_save_path, 'w') as f:
            json.dump(to_be_written, f, indent=4)



