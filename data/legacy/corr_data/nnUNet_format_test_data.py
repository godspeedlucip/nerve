# import numpy as np
# import os
# join = os.path.join
# import shutil
# import nibabel as nib
# import random
# from tqdm import tqdm
# import json

# # def create_json_each(dest_image_path, dest_label_path):
# #     x_voxel_size, y_voxel_size, z_voxel_size = 0.15, 0.15, 0.15
# #     voxel_size = {"spacing": [z_voxel_size, x_voxel_size, y_voxel_size]}

# #     image_dest_list = sorted(os.listdir(dest_image_path))
# #     for i in image_dest_list:
# #         with open(join(dest_image_path, i.split('.')[0].split('_')[0]+'.json'), 'w') as f:
# #             json.dump(voxel_size, f, indent=4)

# #     label_dest_list = sorted(os.listdir(dest_label_path))
# #     for i in label_dest_list:
# #         with open(join(dest_label_path, i.split('.')[0].split('_')[0]+'.json'), 'w') as f:
# #             json.dump(voxel_size, f, indent=4)

# # def create_json_all(dest_data_path):
# #     numTraining = len(os.listdir(join(dest_data_path,'labelsTr')))
# #     to_be_written = {
# #         "channel_names":{
# #             "0": "zscore" #尝试一下z_score归一化
# #         },
# #         "labels":{
# #             "background": 0,
# #             "vessel": 1
# #         },
# #         "numTraining": numTraining,
# #         "file_ending": ".tif",
# #         "overwrite_image_reader_writer": "Tiff3DIO" #对于tiff文件最好是显示的指定用哪个读取器来读取
# #     }
# #     top_json_save_path = join(dest_data_path, "dataset.json")
# #     with open(top_json_save_path, 'w') as f:
# #         json.dump(to_be_written, f, indent=4)


# if __name__=='__main__':
#     # import argparse

#     # parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
#     # parser.add_argument("--tar_range",type=int)
#     # parser.add_argument("--tar_num", type=int)
#     # parser.add_argument("--dataset_num", type=int)
#     # parser.add_argument("--filter", type=bool, default=False)
#     # parser.add_argument("--mode", type=str, default='train')
#     # args = parser.parse_args()

#     # tar_range = args.tar_range
#     # tar_num = args.tar_num
#     # filter = args.filter
#     # dataset_num = args.dataset_num
#     # mode = args.mode

#     # assert mode=='train' or mode=='test', "mode must be 'train' or 'test'"

#     # print('tar_range: ',tar_range,
#     #       'tar_num: ',tar_num,
#     #       'filter: ',filter,
#     #       'mode: ',mode)

#     # use_filter = 'Yes' if filter else 'No'

#     # data_dir = '/data1/liuyang/stent/nnUNet/data/nnUNet_raw'
#     # for i in os.listdir(data_dir):
#     #     has_num = int(i[7:10])
#     #     # assert has_num!=dataset_num

#     # # dataset_num = 325 # 300~400之间

#     # src_data_path = f'/data1/liuyang/stent/DTR/data1/trun_data/{mode}_data/{tar_range}_{tar_num}_fdk{use_filter}'
#     # if(mode=='train'):
#     #     dest_data_path = f"/data1/liuyang/stent/nnUNet/data/nnUNet_raw/Dataset{dataset_num}_{tar_range}-{tar_num}-{use_filter}"
#     # else:
#     #     dest_data_path = f"/data1/liuyang/stent/nnUNet/data/nnUNet_raw/Dataset{dataset_num}_{tar_range}-{tar_num}-{use_filter}"

#     # if(mode=='train'):
#     #     dest_label_path = join(dest_data_path, 'labelsTr')
#     #     dest_image_path = join(dest_data_path, 'imagesTr')
#     # else:
#     #     dest_label_path = join(dest_data_path, 'labelsTs')
#     #     dest_image_path = join(dest_data_path, 'imagesTs')

#     # os.makedirs(dest_label_path, exist_ok=True)
#     # os.makedirs(dest_image_path, exist_ok=True)

#     # # 移动label
#     # for i in os.listdir(join(src_data_path,'vol_gt')): #mr_001_sample_00_0-25.tif
#     #     i_name = i.replace('_','-')
#     #     shutil.copy(join(src_data_path,'vol_gt',i), join(dest_label_path, i_name))

#     # # 移动image
#     # for i in os.listdir(join(src_data_path,'vol_train')): #mr_001_sample_00_0-25.tif
#     #     i_name = i.replace('_','-')
#     #     shutil.copy(join(src_data_path,'vol_train',i), join(dest_image_path, i_name.split('.')[0]+'_0000.tif'))

#     # # 创建配置文件
#     # # 之后还要再给每个训练和标签文件都写入一个json文件
#     # create_json_each(dest_image_path, dest_label_path)

#     # if(mode=='train'):
#     #     create_json_all(dest_data_path)


#     ## 创建符合nnUNet格式的数据集
#     import argparse
#     parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
#     parser.add_argument("--tar_range",type=int, default=60)
#     parser.add_argument("--tar_num", type=int, required=True)
#     parser.add_argument("--radius", type=int, default=64)
#     parser.add_argument("--data_type", type=str, required=True)
#     parser.add_argument("--data_name", type=str,default='costa')
#     parser.add_argument("--resolution", type=float, required=True)
#     parser.add_argument("--dataset_num", type=int, required=True)
#     args = parser.parse_args()
    
#     src_data_dir = '/data1/liuyang/stent/interior/data/corr_data/data'
#     radius = args.radius
#     tar_range = args.tar_range
#     tar_num = args.tar_num
#     resolution=args.resolution
#     dataset_num = args.dataset_num
#     data_type = args.data_type
#     data_name = args.data_name


#     src_data_path = join(src_data_dir, f'{data_name}/voxel_{resolution}/trun_data/radius-{radius}/corr_{data_type}/test_data/{tar_range}_{tar_num}-fdkNoFilter')
#     dest_data_path = join(save_dir, f"{data_name}/{data_type}/radius{radius}-{tar_range}-{tar_num}")
    
#     dest_label_path = join(dest_data_path, 'labelsTs')
#     dest_image_path = join(dest_data_path, 'imagesTs')

#     os.makedirs(dest_label_path, exist_ok=True)
#     os.makedirs(dest_image_path, exist_ok=True)

#     # 移动label
#     for i in tqdm(os.listdir(join(src_data_path,'vol_gt'))): #mr_001_sample_00_0-25.tif
#         i_name = i.replace('_','-')
#         shutil.copy(join(src_data_path,'vol_gt',i), join(dest_label_path, i_name))

#     # 移动image
#     for i in tqdm(os.listdir(join(src_data_path,'vol_train_no_masked'))): #mr_001_sample_00_0-25.tif
#         i_name = i.replace('_','-')
#         shutil.copy(join(src_data_path,'vol_train_no_masked',i), join(dest_image_path, i_name.split('.')[0]+'_0000.nii.gz'))

#     # 再创建一个总的json文件
#     numTraining = len(os.listdir(join(src_data_path,'vol_gt')))
#     to_be_written = {
#         "channel_names":{
#             "0": "zscore" #尝试一下z_score归一化
#         },
#         "labels":{
#             "background": 0,
#             "vessel": 1
#         },
#         "numTraining": numTraining,
#         "file_ending": ".nii.gz",
#         "overwrite_image_reader_writer": "SimpleITKIO"
#     }

#     top_json_save_path = join(dest_data_path, "dataset.json")
#     with open(top_json_save_path, 'w') as f:
#         json.dump(to_be_written, f, indent=4)
