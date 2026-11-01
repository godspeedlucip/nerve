
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import plyfile,copy
import pandas as pd
import tifffile as tiff
import torch

from scipy import io
import os
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from scipy.signal import find_peaks
import open3d as o3d
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, Rbf
import SimpleITK as sitk

from geomdl import NURBS, utilities
from scipy.ndimage import convolve, gaussian_filter
import tigre
import tigre.algorithms as algs

import open3d as o3d

from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

import nibabel as nib

import random
from tqdm import tqdm

import SimpleITK as sitk # 推荐用于医学图像处理
from skimage.morphology import skeletonize_3d
from skimage.measure import regionprops,label

import json

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom



def read_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if not pcd.has_points():
        print(f"警告: 文件 {ply_path} 中没有读取到任何点。")
        return pcd, np.array([])
        
    # 将 Open3D 点云对象转换为 NumPy 数组
    point_cloud_np = np.asarray(pcd.points)
    return  point_cloud_np

def set_geo(DSD  =1250, 
            DSO = 750,
            nDetector = np.array([512,512]),
            dDetector = np.array([0.3, 0.3]), 
            nVoxel = np.array([248, 510, 636]),
            dVoxel = np.array([0.3, 0.3, 0.6]),
            geo_type='cone'):
    # Lets create a geometry object
    geo = tigre.geometry()
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = DSD  # Distance Source Detector      (mm)
    geo.DSO = DSO  # Distance Source Origin        (mm)
    # Detector parameters
    geo.nDetector = nDetector  # number of pixels              (px)
    geo.dDetector =dDetector  # size of each pixel            (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = nVoxel  # number of voxels              (vx)
    geo.dVoxel= dVoxel # size of each voxel            (mm)
    geo.sVoxel = geo.dVoxel * geo.nVoxel  # total size of the image       (mm)
    
    # Offsets
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
    # geo.offOrigin = geo.sVoxel /2 

    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # geo.offDetector = geo.sDetector / 2

    geo.accuracy = 0.5  # Variable to define accuracy of
    geo.COR = 0  # y direction displacement for
    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    # geo.mode = "cone"  # Or 'parallel'. Geometry type.
    geo.mode = geo_type  # Or 'parallel'. Geometry type.
    return geo

def truncate_image_single(image, image_mask): #截取单个图像
    ## 只截取ROi的区域（因为实际的数据采集中，探测器大小和重建区域大小也只有ROI的大小）
    """
    输入数据为重建得到的图像以及真实的图像，以及采样点。是对单个数据的处理
    """
    image_mask_data = image_mask.get_fdata()
    image_data = image.get_fdata()

    # 是否选择截取球形区域里面的数据
    # truncate_vol = np.where(image_mask_data, image_data, np.min(image_data)) #用球形去截取
    truncate_vol = image_data #不截取，后面在一个正方形里面都有会数据

    border = np.array(np.where(image_mask_data>0)).T
    border_min = np.min(border,axis=0) #三个轴的最小值列表
    border_max = np.max(border,axis=0) #三个轴的最大值列表
    start_indices = border_min
    # end_indices = border_max + 1
    end_indices = border_max

    # 直接对nib文件进行操作，以便同步改变其header信息
    truncate_vol_nib = nib.Nifti1Image(
        dataobj=truncate_vol,
        affine=image.affine,
        header=image.header
    )

    roi_trun_nib = crop_nifti_image(truncate_vol_nib, start_point=start_indices, end_point=end_indices)
    return roi_trun_nib #这里返回的是一个nib数据截断文件


def crop_nifti_image(original_img , start_point = (0, 0, 0), end_point = (-1, -1, -1)):
    """
    加载 NIfTI 图像，从中心裁剪一个指定大小的正方体区域，
    并自动更新 Header/Affine 信息后保存为新的 .nii.gz 文件。

    Args:
        input_filepath (str): 输入 .nii.gz 文件的路径。
        output_filepath (str): 输出裁剪后的 .nii.gz 文件的路径。
        crop_size (int): 裁剪区域的边长（例如 64），假定为正方体区域。
    """
    # 1. 加载原始图像
    original_data = original_img.get_fdata()
    original_shape = original_data.shape
    
    i_start = start_point[0]
    j_start = start_point[1]
    k_start = start_point[2]
    
    # 计算结束点
    i_end = end_point[0]
    j_end = end_point[1]
    k_end = end_point[2]

    if(i_end > original_shape[0]):
        i_end = original_shape[0]
    if(j_end > original_shape[1]):
        j_end = original_shape[1]
    if(k_end > original_shape[2]):
        k_end = original_shape[2]
    
    # 3. 使用 nibabel 的 .slicer 属性进行裁剪.slicer 会自动裁剪数据，并计算新的仿射矩阵 (Affine)
    cropped_img = original_img.slicer[
        i_start:i_end, 
        j_start:j_end, 
        k_start:k_end, 
        ... # ... 用于处理可能的第 4 维 (例如时间/通道)
    ]
    return cropped_img


## 稀疏采样重建之后直接截取，不然会消耗很多存储空间
def sparse_recon(proj_path, gt_path, gt_mask_path, tar_range, tar_num, save_dir, sample_pc_dir, filter = False, already_has = None):
    if(filter):
        save_path = join(save_dir,f"{tar_range}_{tar_num}-fdkYesFilter")
    else:
        save_path = join(save_dir,f"{tar_range}_{tar_num}-fdkNoFilter")
    
    vol_train_save_path = join(save_path,'vol_train')
    vol_gt_save_path = join(save_path,'vol_gt')
    os.makedirs(vol_train_save_path, exist_ok=True)
    os.makedirs(vol_gt_save_path, exist_ok=True)

    # 180度范围内，尽量多取几例数据来训练
    proj_list = sorted(os.listdir(proj_path))
    gt_list = sorted(os.listdir(gt_path))
    gt_mask_list = sorted(os.listdir(gt_mask_path))

    # for index,i in tqdm(enumerate(proj_list)):
    for index in tqdm(range(len(proj_list))):
        i = proj_list[index]
        if(already_has is not None and i.split('.')[0] in already_has):
            print( f'already processed: {i}' )
            continue

        proj  = np.load(join(proj_path,proj_list[index]))['data'] # 投影是npz格式. 这里的Proj默认就是 z, x, y的格式
        vol_gt_nib = nib.load(join(gt_path, gt_list[index])) # 图像是nii.gz格式
        vol_gt_mask_nib = nib.load(join(gt_mask_path, gt_mask_list[index]))

        vol_gt = np.transpose(vol_gt_nib.get_fdata(), [2, 0, 1])
        
        ## 每个文件对应的投影参数都不是一样的
        z_dim_max, x_dim_max, y_dim_max = vol_gt.shape[0], vol_gt.shape[1], vol_gt.shape[2]
        [x_voxel_size, y_voxel_size, z_voxel_size] = [round(j, 4) for j in  vol_gt_nib.header['pixdim'][1:4]]
         
        ranges = int(180 / tar_range)
        ranges_to_chose = ranges
        if(ranges > 10):
            ranges_to_chose = 10 #最多采集10个吧，太多了也不行

        vol_gt_truncated = truncate_image_single(vol_gt_nib, vol_gt_mask_nib) # 对nib文件进行了裁剪，返回的也是nii.gz格式的文件

        for j in random.sample(range(ranges), k=ranges_to_chose):
            # index = list(np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.uint8)) #注意uint8最多表示到256，因此这里绝对会溢出，因此要全部替换为uint16
            index = list(np.linspace(tar_range*j, tar_range*(j+1), tar_num).astype(np.uint16))
            angles = np.array([np.deg2rad(i) for i in index]) #angle根据index来

            proj_part = proj[index] # 假如index很小，只有4，那么在写入并用ImageJ读取时就会报错，但是当index的长度为10时，就不会报错了。所以这里不用过度担心
            geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
                        dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                        nDetector=np.array([1024*2, 1024*2]),
                        dDetector=np.array([x_voxel_size, y_voxel_size]))
            
            if(filter):
                imgFDK = algs.fdk(proj_part, geo, angles) #有限角的重建结果
            else:
                imgFDK = algs.fdk_no_filter(proj_part, geo, angles) #fdk不用滤波重建

            imgFDK_nib  = nib.Nifti1Image(
                dataobj=np.transpose(imgFDK, [1, 2, 0]), #nii文件中图像的维度顺序是x, y, z
                affine=vol_gt_nib.affine,
                header=vol_gt_nib.header
            )

            vol_train_truncated = truncate_image_single(imgFDK_nib, vol_gt_mask_nib) #只能输入nib文件进行裁剪
            
            file_name = f'{i.split(".")[0]}_{str(int(tar_range*j))}-{str(int(tar_range*(j+1)))}.nii.gz'
            
            nib.save(vol_train_truncated, join(vol_train_save_path, file_name))
            nib.save(vol_gt_truncated, join(vol_gt_save_path, file_name))



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--tar_range",type=int, required=True)
    parser.add_argument("--tar_num", type=int, required=True)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    args = parser.parse_args()

    tar_range = args.tar_range
    tar_num = args.tar_num
    radius = args.radius
    resulotion = args.resolution

    print(f'tar_range: {tar_range}\n',
          f'tar_num: {tar_num}\n',
          f'radius: {radius}\n')


    # 以下是COSTA数据集：
    # # 首先生成训练数据
    # print('生成训练数据')
    # train_proj_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}//src_data/radius-{radius}/train_data/proj_gt'
    # train_gt_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}//src_data/radius-{radius}/train_data/vol_gt'
    # train_gt_mask_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}//src_data/radius-{radius}/train_data/vol_gt_mask'
    # train_save_dir = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}/trun_data_02/radius-{radius}/train_data'
    # train_sample_pc_dir = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}/src_data/radius-{radius}//train_data/point_cloud'
    # already_has = None
    # sparse_recon(train_proj_path, train_gt_path, train_gt_mask_path, tar_range, tar_num, train_save_dir, sample_pc_dir=train_sample_pc_dir, filter=False, already_has = already_has)


    # 之后生成测试数据
    test_proj_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}//src_data/radius-{radius}/test_data/proj_gt'
    test_gt_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}//src_data/radius-{radius}/test_data/vol_gt'
    test_gt_mask_path = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}//src_data/radius-{radius}/test_data/vol_gt_mask'
    test_save_dir = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}/trun_data/radius-{radius}/test_data'
    test_sample_pc_dir = f'/data1/liuyang/stent/interior/data/costa/resample/voxel_{resulotion}/src_data/radius-{radius}//test_data/point_cloud'
    already_has = None
    sparse_recon(test_proj_path, test_gt_path, test_gt_mask_path, tar_range, tar_num, test_save_dir, sample_pc_dir=test_sample_pc_dir, filter=False, already_has = already_has)





