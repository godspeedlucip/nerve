import os,shutil
join = os.path.join
import numpy as np
import tifffile as tiff

from utils.base_utils import set_geo
import random
import tigre
import tigre.algorithms as algs
import argparse
import torch
from tqdm import tqdm

# def get_trun_data():

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range",type=int)
    parser.add_argument("--tar_num", type=int)
    parser.add_argument("--proj_path", type=str)
    parser.add_argument("--vol_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--filter", type=bool, default=False)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ',device)

    tar_range = args.tar_range
    tar_num = args.tar_num                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    proj_path = args.proj_path
    vol_path = args.vol_path
    save_path = args.save_path
    filter = args.filter

    print('tar_range: ',tar_range,
          'tar_num: ',tar_num,
          'proj_path: ',proj_path,
          'vol_path: ',vol_path,
          'save_path: ',save_path,
          'filter: ',filter)

    use_filter = 'Yes' if filter else 'No'
    work_name = f'{tar_range}_{tar_num}_fdk{use_filter}'
    save_dir = f"{join(save_path,work_name)}"

    os.makedirs(join(save_dir,'vol_train'),exist_ok=True)
    os.makedirs(join(save_dir,'vol_gt'),exist_ok=True)


    ## 设置几何
    z_dim_max, x_dim_max, y_dim_max = 256, 128, 128
    z_voxel_size, x_voxel_size, y_voxel_size = 0.2, 0.2, 0.2
    nDetector, dDetector=[512,512],[0.2,0.2]
    geo_mode = 'cone'
    geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
            dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
            nDetector=np.array(nDetector),
            dDetector=np.array(dDetector),
            geo_type=geo_mode)


    # 180度范围内，尽量多取几例数据来训练
    for i in tqdm(os.listdir(proj_path), desc="正在处理投影："):
        proj  = tiff.imread(join(proj_path,i))

        ranges = int(180 / tar_range)
        ranges_to_chose = ranges
        if(ranges >= 10):
            ranges_to_chose = 10 #最多采集10个吧，太多了也不行
        for j in random.sample(range(ranges), k=ranges_to_chose):
            index = np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.uint16)
            print(index)
            angles = np.array([np.deg2rad(i) for i in index])
            proj_part = proj[index]
            if(filter):
                img_fdk_recon = algs.fdk(proj_part, geo, angles, filter='hamming') #有限角的重建结果
            else:
                img_fdk_recon = algs.fdk_no_filter(proj_part, geo, angles) #fdk不用滤波重建

            file_name = f'{i.split(".")[0]}_{str(int(tar_range*j))}-{str(int(tar_range*(j+1)))}.tif'
            tiff.imwrite(join(save_dir,'vol_train',file_name), img_fdk_recon)
            shutil.copy(join(vol_path,i),join(save_dir,'vol_gt',file_name))
        # print(i)