
import numpy as np
import plyfile, copy, pickle, os, random, json, re, pickle
import pandas as pd
import tifffile as tiff
from scipy import io
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline, Rbf
import SimpleITK as sitk
from scipy.ndimage import convolve, gaussian_filter
import tigre
import tigre.algorithms as algs
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import zoom
from collections import defaultdict
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import uniform_filter
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter



def resample_nifti_to_spacing(img, target_spacing=(1.0, 1.0, 1.0), interpolation_order=1, thred=0.5, need_thred = True):
    """
    使用 NiBabel 和 SciPy 对 NIfTI 图像进行重采样到指定体素大小。

    Args:
        input_path (str): 输入 .nii 或 .nii.gz 文件的路径。
        output_path (str): 输出重采样后文件的路径。
        target_spacing (tuple): 目标体素大小 (X, Y, Z)，单位通常是毫米(mm)。
        interpolation_order (int): SciPy 插值阶数。0=最近邻, 1=线性, 3=三次(默认)。
    """
    data = img.get_fdata()
    affine = img.affine
    
    # 从仿射矩阵中提取原始体素大小
    # 仿射矩阵的 (3x3) 旋转/缩放部分每列的欧几里得范数即为体素大小
    original_spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    
    # 2. 计算缩放因子 (Zoom Factors)
    # 缩放因子 = 原始大小 / 目标大小
    zoom_factors = original_spacing / np.array(target_spacing)

    # 3. 应用插值重采样
    # 使用 SciPy 的 zoom 函数对数据进行插值
    resampled_data = zoom(
        data,
        zoom_factors,
        order=interpolation_order,
        mode='nearest' # 处理边界条件
    )
    if(need_thred):
        resampled_data[resampled_data >= thred]  =1
        resampled_data[resampled_data < thred]  = 0
        resampled_data.astype(np.int8)

    # 4. 计算新的仿射矩阵 (New Affine)
    # 保持图像在物理空间中的位置不变 (即坐标原点不变)，
    # 只修改体素大小部分。通过将仿射矩阵的每一列除以对应的缩放因子来实现。
    new_affine = affine.copy()
    for i in range(3):
        # 将原始仿射矩阵的 (3x3) 缩放/旋转部分按比例调整
        # 缩放因子 (zoom_factors[i]) 越大，新的体素在物理空间中越小，
        # 因此仿射矩阵中的对应值需要减小 (除以 zoom_factors[i])。
        new_affine[:3, i] = affine[:3, i] / zoom_factors[i]
    
    
    # 5. 创建并保存新的 NiBabel 图像
    # 使用重采样后的数据和新的仿射矩阵创建新的 Nifti1Image
    # resampled_img = nib.Nifti1Image(resampled_data.astype(img.get_data_dtype()), new_affine, header=img.header)
    resampled_img = nib.Nifti1Image(resampled_data, new_affine, header=img.header)
    
    # 可选：更新头文件中的体素大小信息 (pixdim)，虽然通常依赖 affine 矩阵
    resampled_img.header.set_zooms(target_spacing) 
    
    return resampled_img

def resample_dir(src_dir_name, dest_dir_name, target_spacing=(0.2, 0.2, 0,2), type_of='image'):
    for i in tqdm(os.listdir(src_dir_name)):
        nib_src = nib.load(join(src_dir_name, i))
        if(type_of=='image'):
            thred = False
            interp = 3
        else:
            thred = True
            interp = 1
        resample_nib_data = resample_nifti_to_spacing(nib_src, target_spacing, interpolation_order=interp, need_thred=thred)
        nib.save(resample_nib_data, join(dest_dir_name, i))


## 这是给原始的costa和topbrain数据集进行重采样的流程
# if __name__=='__main__':
#     dataset_name = 'costa' #topbrain
#     data_dir = f'/data1/liuyang/stent/DTE/data/src_data/{dataset_name}'
#     save_dir = f'/data1/liuyang/stent/DTE/data/resample_data/{dataset_name}'
#     mode = ['train','test']
#     target_spacing = (0.2, 0.2, 0.2)
#     for i in mode:
#         image_path = join(data_dir, f"{i}_data/image")
#         label_path = join(data_dir, f"{i}_data/label")
#         image_save_path = join(save_dir, f"{i}_data/image")
#         label_save_path = join(save_dir, f"{i}_data/label")
#         os.makedirs(image_save_path, exist_ok=True)
#         os.makedirs(label_save_path, exist_ok=True)
#         print(i)
#         resample_dir(image_path, image_save_path, target_spacing, type_of='image')
#         resample_dir(label_path, label_save_path, target_spacing, type_of='label')


# # 这是给fill hole之后的cost和topbrain数据集进行重采样的流程
# if __name__=='__main__':
#     dataset_name = 'costa' #topbrain
#     data_dir = f'/data1/liuyang/stent/DTE/data/src_data/costa/'
#     save_dir = '/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/src_data/radius-64/vessel'
#     mode = ['train','test']
#     target_spacing = (0.2, 0.2, 0.2)
#     for i in mode:
#         image_path = join(data_dir, f"{i}_data/fill_hole_image")
#         image_save_path = join(save_dir, f"{i}_data/fill_hole_image")
#         os.makedirs(image_save_path, exist_ok=True)
#         print(i)
        # resample_dir(image_path, image_save_path, target_spacing, type_of='image')




if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--resolution", type=float, default='0.2')
    parser.add_argument("--data_name", type=str, default='costa') #costa、topbrain
    parser.add_argument("--train", type=int, default=True) #0,1
    parser.add_argument("--test", type=int, default=True)# 0,1
    parser.add_argument("--save_dir", type=str, default='/data1/liuyang/stent/DTE/data/resample_data')
    parser.add_argument("--type_of", type=str, default='image')
    parser.add_argument("--data_type", type=str, default='vessel')
    args = parser.parse_args()

    radius = args.radius
    resolution = args.resolution
    save_dir = args.save_dir
    data_name = args.data_name
    data_type = args.data_type
    to_train = args.train
    to_test = args.test
    type_of = args.type_of
    assert data_type in set(['stent', 'vessel'])
    assert to_train in set([0,1])
    assert to_test in set([0,1])
    assert type_of in set(['image', 'label'])


    data_dir = f'/data1/liuyang/stent/DTE/data/src_data/{data_name}'
    save_dir = f"/data1/liuyang/stent/DTE/data/resample_data/{data_name}/voxel_{resolution}/src_data/radius-64/{data_type}"
    mode = ['train','test']
    target_spacing = (resolution, resolution, resolution)
    for i in mode:
        image_path = join(data_dir, f"{i}_data/fill_hole_image")
        image_save_path = join(save_dir, f"{i}_data/fill_hole_image")
        os.makedirs(image_save_path, exist_ok=True)
        print(i)
        resample_dir(image_path, image_save_path, target_spacing=target_spacing, type_of=type_of)

    # for i in mode:
    #     image_path = join(data_dir, f"{i}_data/label")
    #     image_save_path = join(save_dir, f"{i}_data/label")
    #     os.makedirs(image_save_path, exist_ok=True)
    #     print(i)
    #     resample_dir(image_path, image_save_path, target_spacing, type_of='label')

