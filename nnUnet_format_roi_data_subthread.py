# 给2D的数据创建nnunet形式的数据


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
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--use_split", type=int, default=False)
    args = parser.parse_args()

    src_data_dir = '/data1/liuyang/stent/DTE/data/resample_data'
    resolution=args.resolution
    dataset_num = args.dataset_num
    data_type = args.data_type
    data_name = args.data_name
    to_train = args.train
    to_test = args.test
    radius = args.radius
    split = args.split
    use_split = args.use_split

    # assert data_type in set(['stent', 'vessel'])
    assert to_train in set([0,1])
    assert to_test in set([0,1])

    # for i in ['train', 'test']:
    to_process = []
    if(to_train):
      to_process.append('train')
    if(to_test):
      to_process.append('test')

    target_num = 18
    tar_index = np.linspace(0, 360, target_num+1)[0:-1].astype(np.int16)

    ## 防止data leaking的策略
    tar_range = 60
    tar_num = 3
    ranges = int(180 / tar_range)
    ranges_to_chose = ranges
    to_train_index = []
    if(ranges > 10):
        ranges_to_chose = 10 #最多采集10个吧，太多了也不行
    for j in random.sample(range(ranges), k=ranges_to_chose):
        index = list(np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.int16))
        to_train_index.extend(index)
    to_train_index = set(sorted(list(set(to_train_index))))

    for mode in to_process:
      # src_data_path = join(src_data_dir, f'{data_name}/voxel_{resolution}/roi_proj_3/radius-{radius}/{mode}_data')
      # src_data_path = join(src_data_dir, f'{data_name}/voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data')
      src_data_path = join(src_data_dir, f'{data_name}/voxel_{resolution}/new_proj/{data_type}/radius-{radius}/{mode}_data')
      dest_data_path = f"/data1/liuyang/stent/nnUNet/data/nnUNet_raw/Dataset{dataset_num}_-seg-{data_name}-{data_type}-radius{radius}-{resolution}"

      if(mode=='train'):
        dest_label_path = join(dest_data_path, 'labelsTr')
        dest_image_path = join(dest_data_path, 'imagesTr')
      else:
        dest_label_path = join(dest_data_path, 'labelsTs')
        dest_image_path = join(dest_data_path, 'imagesTs')
      os.makedirs(dest_label_path, exist_ok=True)
      os.makedirs(dest_image_path, exist_ok=True)

      # 加载split配置
      if use_split:
          split_json_path = join(src_data_dir, f'{data_name}/voxel_{resolution}/src_data/radius-{radius}/split/{mode}', f'part_{split}.json')
          with open(split_json_path, "r", encoding="utf-8") as f:
              split_json = json.load(f)
          split_keys = set(split_json.keys())
      else:
          split_keys = None

      # 移动label
      label_path = join(src_data_path,'label_proj','rec')
      for i in tqdm(os.listdir(label_path)): #mr_001_sample_00_0-25.tif
          if split_keys is not None:
              sample_base = i.split('_sample_')[0]
              if sample_base not in split_keys:
                  continue
          file_list = sorted(os.listdir(join(label_path, i)))
          for j in tar_index:
            if(j in to_train_index):
               continue
            data_path = file_list[j]
            image = read_sp(join(label_path, i, data_path))
            image[image > 0.001] = 1
            image[image <= 0.001] = 0
            image.astype(np.int8)
            j_name = data_path.replace('_','-')
            j_name = j_name.split('.npz')[0]+'.tif'
            tiff.imwrite(join(dest_label_path, j_name), image.astype(np.float32))

      # 移动image
      image_path = join(src_data_path,'image_proj','rec')
      for i in tqdm(os.listdir(image_path)): #mr_001_sample_00_0-25.
          if split_keys is not None:
              sample_base = i.split('_sample_')[0]
              if sample_base not in split_keys:
                  continue
          file_list = sorted(os.listdir(join(image_path, i)))
          for j in tar_index:
            if(j in to_train_index):
              continue
            data_path = file_list[j]
            image = read_sp(join(image_path, i, data_path))
            j_name = data_path.replace('_','-')
            j_name = j_name.split('-0000')[0]+'_0000.tif'
            tiff.imwrite(join(dest_image_path, j_name), image.astype(np.float32))

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
            "file_ending": ".tif",
            "overwrite_image_reader_writer": "NaturalImage2DIO"
        }

        top_json_save_path = join(dest_data_path, "dataset.json")
        with open(top_json_save_path, 'w') as f:
            json.dump(to_be_written, f, indent=4)
