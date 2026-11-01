### 这里假设支架已经全都已经得到正确的ply格式。


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import plyfile
import tifffile as tiff

from scipy import io

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

import open3d as o3d

from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

import random

from utils.base_utils import Normalize, stent_simu, expand_and_smooth_voxel_mask, voxelize_1, voxelize_2, center_point_cloud_to_voxel_space, set_geo, flip_point_cloud
from utils.gen_curve import generate_constrained_spiral_curve_iterative, generate_single_sin_curve, generate_3d_sine_curve_pointcloud, generate_random_smooth_curve, create_nurbs_centerline_fixed_length, create_outward_arc_u_shape_point_cloud
from utils.curve_trans import transform_pointcloud_frenet_serret, transform_pointcloud_parallel, compute_frenet_serret, compute_parallel_transport_frames
from utils.file_rw import read_ply


def get_deform_stent_Frenet_Serret(point_data,curve_points=None):
    # curve_points: (N,3)
    ## 将支架扭曲
    point_voxel_data = np.copy(point_data)
    max_z = np.max(point_voxel_data[:,2])
    min_z = np.min(point_voxel_data[:,2])

    TOTAL_LENGTH = np.abs(int(max_z-min_z)) #支架的长度（像素个数）
    
    # 1. 原始点云
    cyl_points = stent_simu(point_voxel_data)

    # 2. 目标曲线
    if(curve_points is None):
        if(random.randint(0,100) % 2==0):
            control_points = 4
        else:
            control_points = 5
        curve_points = create_nurbs_centerline_fixed_length(control_points, (128,128,256), target_length=int(TOTAL_LENGTH)) #一半的数据较为复杂，其余的较为简单
    
    # curve_points = curve_points.T
    
    # 3. Frenet-Serret
    _, N_hat, B_hat = compute_frenet_serret(curve_points)

    # 4. 点云变换
    cyl_points_deformed = transform_pointcloud_frenet_serret(cyl_points, curve_points, N_hat, B_hat)  # cyl_points: (N,3), curve_points:(3, N_c)

    return cyl_points_deformed,curve_points


def get_deform_stent_parallel_transport(point_voxel_data,curve_points=None):
    # curve_points: (N,3)
    ## 将支架扭曲
    max_z = np.max(point_voxel_data[:,2])
    min_z = np.min(point_voxel_data[:,2])

    TOTAL_LENGTH = np.abs(int(max_z-min_z)) #支架的长度（像素个数）
    
    # 1. 原始点云
    cyl_points = stent_simu(point_voxel_data)

    # 2. 目标曲线
    if(curve_points is None):
        curve_points = create_nurbs_centerline_fixed_length(4, (128,128,256), target_length=int(TOTAL_LENGTH))

    _, N_hat, B_hat = compute_parallel_transport_frames(curve_points)

    # 4. 点云变换
    cyl_points_deformed = transform_pointcloud_parallel(cyl_points, curve_points, N_hat, B_hat)  # cyl_points: (N,3), curve_points:(3, N_c)

    return cyl_points_deformed,curve_points



## 这里读取点云的都是要在读取直线的支架再对其进行曲线变换的
def main(ply_path, data_save_path):
    os.makedirs(join(data_save_path,'proj'),exist_ok=True)
    os.makedirs(join(data_save_path,'vol'),exist_ok=True)
    nurbs_num = 6
    u_nums = 4
    double_sin_num = 2
    spiral_num = 1
    single_sin_num = 2
    self_num = 1
    random_num = 4

    angles = np.linspace(0,np.pi*2,360)

    z_dim_max, x_dim_max, y_dim_max = 256, 128, 128
    z_voxel_size, x_voxel_size, y_voxel_size = 0.2, 0.2, 0.2
    nDetector, dDetector=[512,512],[0.2,0.2]
    geo_mode = 'cone'
    geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
            dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
            nDetector=np.array(nDetector),
            dDetector=np.array(dDetector),
            geo_type=geo_mode)
    

    proj_index = np.linspace(0, 360, 360+1)[:-1].astype(np.uint16)
    
    to_process = set(['stent1','stent2','bent_stent'])

    for i in os.listdir(ply_path):
        #所有的点云文件都已经被初始化到256的z轴长度。
        print(i)
        if(i.split('.')[0] not in to_process):
            continue

        ## NURBS曲线
        for j in range(nurbs_num):
            # 变成曲线
            times = 0
            while(1):
                # print(point_cloud.shape)
                point_cloud = read_ply(join(ply_path,i))
                control_points = 4
                curve_points = create_nurbs_centerline_fixed_length(control_points, (128,128,256), target_length=256) #一半的数据较为复杂，其余的较为简单
                curve_points = curve_points.T
                point_cloud_deformed, control_points = get_deform_stent_Frenet_Serret(point_cloud,curve_points)
                deform_voxel,if_inside = voxelize_2(point_cloud_deformed)
                deform_voxel = np.transpose(deform_voxel,[2,0,1]).astype(np.float32)
                times += 1
                if(times>=10):
                    print(i,'  ',j,'  not inside',)
                    break
                if(if_inside):
                    break

            gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.5, kernel_size=2, valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_nurbs_{str(j).zfill(2)}.tif'),projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_nurbs_{str(j).zfill(2)}.tif'),gau_voxel_data)

        ## 随机的曲线
        for j in range(random_num):
            times = 0
            while(1):
                # print(point_cloud.shape)
                point_cloud = read_ply(join(ply_path,i))
                control_points = 4
                curve_points = generate_random_smooth_curve() #一半的数据较为复杂，其余的较为简单
                curve_points = curve_points.T
                point_cloud_deformed, control_points = get_deform_stent_Frenet_Serret(point_cloud,curve_points)
                deform_voxel,if_inside = voxelize_2(point_cloud_deformed)
                deform_voxel = np.transpose(deform_voxel,[2,0,1]).astype(np.float32)
                times += 1
                if(times>=10):
                    print(i,'  ',j,'  not inside',)
                    break
                if(if_inside):
                    break

            gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.5,kernel_size=2,valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_random_{str(j).zfill(2)}.tif'), projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_random_{str(j).zfill(2)}.tif'),gau_voxel_data)


        ## u curve U型线
        for j in range(u_nums):
            # 变成曲线
            times = 0
            while(1):
                # print(point_cloud.shape)
                point_cloud = read_ply(join(ply_path,i))
                curve_points = create_outward_arc_u_shape_point_cloud(total_length=300, height=80, total_points=20000,flip_or_not=True) #生成U型曲线
                if(j<2): # 一半的数据用来翻转, 其余的不翻转
                    curve_points = flip_point_cloud(curve_points,axis=2)
                
                point_cloud_deformed, control_points = get_deform_stent_parallel_transport(point_cloud,curve_points=curve_points)
                deform_voxel,if_inside = voxelize_2(point_cloud_deformed)
                deform_voxel = np.transpose(deform_voxel,[2,0,1]).astype(np.float32)
                times += 1
                if(times>=10):
                    print(i,'  ',j,'  not inside',)
                    break
                if(if_inside):
                    break
            gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.5,kernel_size=2,valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_u_{str(j).zfill(2)}.tif'),projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_u_{str(j).zfill(2)}.tif'),gau_voxel_data)


        ## spiral curve
        for j in range(spiral_num):
            # 变成曲线
            times = 0
            while(1):
                # print(point_cloud.shape)
                point_cloud = read_ply(join(ply_path,i))
                curve_points = generate_constrained_spiral_curve_iterative(target_length=280,num_points=30000,voxel_size=(128, 128, 256),xy_padding=20,z_min_span=128) #螺旋曲线  
                curve_points  =curve_points.T
                point_cloud_deformed, control_points = get_deform_stent_Frenet_Serret(point_cloud,curve_points=curve_points)
                deform_voxel,if_inside = voxelize_2(point_cloud_deformed)
                deform_voxel = np.transpose(deform_voxel,[2,0,1]).astype(np.float32)
                times += 1
                if(times>=10):
                    print(i,'  ',j,'  not inside',)
                    break
                if(if_inside):
                    break
            gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.5,kernel_size=2,valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_spiral_{str(j).zfill(2)}.tif'),projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_spiral_{str(j).zfill(2)}.tif'),gau_voxel_data)
        

        ## double sin curve 双正弦曲线
        for j in range(double_sin_num):
            # 变成曲线
            times = 0
            while(1):
                point_cloud = read_ply(join(ply_path,i))
                curve_points = generate_3d_sine_curve_pointcloud(N=20000,wavelength_x=120, wavelength_y=150,total_length=350)
                curve_points  = curve_points.T
                point_cloud_deformed, control_points = get_deform_stent_Frenet_Serret(point_cloud,curve_points=curve_points)
                deform_voxel,if_inside = voxelize_2(point_cloud_deformed)
                deform_voxel = np.transpose(deform_voxel,[2,0,1]).astype(np.float32)
                times += 1
                if(times>=10):
                    print(i,'  ',j,'  not inside',)
                    break
                if(if_inside):
                    break
            gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.5,kernel_size=2,valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_doubleSin_{str(j).zfill(2)}.tif'),projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_doubleSin_{str(j).zfill(2)}.tif'),gau_voxel_data)
        
        ## single sin curve 单正弦曲线
        for j in range(single_sin_num):
            # 变成曲线
            times = 0
            while(1):
                point_cloud = read_ply(join(ply_path,i))
                curve_points = generate_single_sin_curve(N=20000, total_length=300, voxel_size=(128,128,256),rate=0.75)
                curve_points  = curve_points.T
                point_cloud_deformed, control_points = get_deform_stent_Frenet_Serret(point_cloud,curve_points=curve_points)
                deform_voxel,if_inside = voxelize_2(point_cloud_deformed)
                deform_voxel = np.transpose(deform_voxel,[2,0,1]).astype(np.float32)
                times += 1
                if(times>=10):
                    print(i,'  ',j,'  not inside',)
                    break
                if(if_inside):
                    break
            gau_voxel_data = expand_and_smooth_voxel_mask(deform_voxel,sigma=0.5,thred=0.5,kernel_size=2,valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_singleSin_{str(j).zfill(2)}.tif'),projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_singleSin_{str(j).zfill(2)}.tif'),gau_voxel_data)

        ## 自身的原始的曲线
        for j in range(self_num):
            point_cloud = read_ply(join(ply_path,i))
            self_stent = center_point_cloud_to_voxel_space(point_cloud) # 首先将其移动到体素空间中心
            self_stent,if_inside = voxelize_2(self_stent)
            self_stent = np.transpose(self_stent,[2,0,1])
            gau_voxel_data = expand_and_smooth_voxel_mask(self_stent,sigma=0.5,thred=0.5,kernel_size=2,valid_thred=2)
            projections = tigre.Ax(gau_voxel_data.astype(np.float32), geo, angles)
            tiff.imwrite(join(data_save_path,'proj',f'{i.split(".")[0]}_self_{str(j).zfill(2)}.tif'),projections[proj_index])
            tiff.imwrite(join(data_save_path,'vol',f'{i.split(".")[0]}_self_{str(j).zfill(2)}.tif'),gau_voxel_data)


if __name__=='__main__':
    ## 先得到训练数据
    ply_path = './data/ply/train'
    data_save_path = './data/train_data'
    main(ply_path=ply_path, data_save_path=data_save_path)
