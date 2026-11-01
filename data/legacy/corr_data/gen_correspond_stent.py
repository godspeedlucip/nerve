import numpy as np
import plyfile, copy, pickle, os, random, json, argparse, warnings
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

from gen_stent.utils.base_utils import stent_simu, expand_and_smooth_voxel_mask, voxelize_1, voxelize_2, center_point_cloud_to_voxel_space, set_geo, flip_point_cloud, get_min_max, visualize_braid_stent
from gen_stent.utils.gen_curve import generate_constrained_spiral_curve_iterative, generate_single_sin_curve, generate_3d_sine_curve_pointcloud, generate_random_smooth_curve, create_nurbs_centerline_fixed_length, create_outward_arc_u_shape_point_cloud
from gen_stent.utils.curve_trans import transform_pointcloud_frenet_serret, transform_pointcloud_parallel, compute_frenet_serret, compute_parallel_transport_frames
from gen_stent.utils.file_rw import read_ply
from gen_stent.utils.gen_stent import w_like_Stent, ring_Stent, Stent_Ply, sin_Stent
from gen_stent.utils.get_stent_para import map_stent_to_curve

from utils.centerline import (
    build_graph,
    extract_random_segment,
    sort_points_nearest_neighbor,
    sort_centerline_robust,
    interpolate_linear_precise,
    process_point_cloud,
    smooth_curve,
    compute_curve_length,
    remove_small_components,
    mean_of_8_neighbors_batch
)

from utils.postprocess import (
    thicken_by_distance,
    expand_and_smooth_voxel_mask,
    change_radius
)

def get_segments(components, spacing, target_length, number_segments):
    indices = list(range(len(components)))
    random.shuffle(indices)
    valid_selected, segment_25mms = [],[]
    for idx in indices:
        segment_25mm = extract_random_segment(
            points_voxel=components[idx],
            spacing=spacing,
            length_mm=target_length #这里的length是物理长度
        )
        
        if segment_25mm is not None:
            segment_25mms.append(segment_25mm)
            valid_selected.append(components[idx])

        if(len(segment_25mms)==number_segments):
            break
    
    if(len(segment_25mms)==0):
        ## 如果完全不满足，就返回None
        return segment_25mms

    ## 就算选到重复的也要保证数量
    while(len(segment_25mms) != number_segments):
        indices = list(range(len(valid_selected)))
        random.shuffle(indices)
        idx = indices[0]
        segment_25mm = extract_random_segment(
            points_voxel=valid_selected[idx],
            spacing=spacing,
            length_mm=target_length #这里的length是物理长度
        )
        if segment_25mm is not None:
            segment_25mms.append(segment_25mm)
            valid_selected.append(valid_selected[idx])
    
    return segment_25mms

def get_stent_generator_train(ply_dir):
    stent1_generator = Stent_Ply(read_ply(join(ply_dir, 'stent1.ply')))
    stent2_generator = Stent_Ply(read_ply(join(ply_dir, 'stent2.ply')))
    stent3_generator = Stent_Ply(read_ply(join(ply_dir, 'stent3.ply')))
    stent4_generator = Stent_Ply(read_ply(join(ply_dir, 'stent4.ply')))
    wall_stent_generator = Stent_Ply(read_ply(join(ply_dir, 'wall_stent.ply')))
    sin_stent_generator = sin_Stent()
    w_like_stent_generator = w_like_Stent()
    ring_stent_generator = ring_Stent()

    stent_name = ['stent1', 'stent2', 'stent3', 'stent4', 'wallStent', 'sinStent', 'wLikeStent', 'ringStent']

    return [stent1_generator, stent2_generator, stent3_generator, stent4_generator, wall_stent_generator, sin_stent_generator, w_like_stent_generator, ring_stent_generator], stent_name

def get_stent_generator_test(ply_dir):
    stent5_generator = Stent_Ply(read_ply(join(ply_dir, 'stent5.ply')))
    w_stent_generator = Stent_Ply(read_ply(join(ply_dir, 'w_stent.ply')))
    stent_name = ['stent5', 'w_stent']
    return [stent5_generator, w_stent_generator], stent_name


def stent_expand(sub_voxel, stent_name):
    if(stent_name in set(['wall_stent'])): #这个支架太密集了，直接返回原始的数据就行了
        return sub_voxel, True
    else:
        expand_sub_voxel, sub_voxel_skeleton, flag = thicken_by_distance(sub_voxel, r_voxel = 1) #尝试扩大其半径
        return expand_sub_voxel, flag

def main(data_dir, stent_ply_dir, save_dir, mode='train'):
    # 设置保存路径
    stent_save_path = join(save_dir,'stent')
    config_save_path = join(save_dir,'config')
    os.makedirs(stent_save_path, exist_ok=True)
    os.makedirs(config_save_path, exist_ok=True)

    ## 一些提前就创建完成的文件
    resample_dir = join(data_dir, 'resample')
    distance_dir = join(data_dir, 'distance_map')
    skeleton_dir = join(data_dir, 'skeleton')

    number_segments = 10 #每例数据要选10个ROI
    target_length = 25 #生成的支架的长度, 单位是mm
    segment_target_points = 2000 #每条线段的应该有的点的数量
    smooth_window = 800 #对线段进行平滑的窗口

    if(mode=='train'):
        generators, stent_names = get_stent_generator_train(stent_ply_dir)
    else:
        generators, stent_names = get_stent_generator_test(stent_ply_dir)

    # all_config = dict()
    to_process = sorted(os.listdir(resample_dir))

    for i in tqdm(to_process):

        ## 加载必要的数据
        nib_mask = nib.load(join(resample_dir,i)) #(w,h,d). 假设w是x的话，那么h就是y。那么确实是 (x,y,z)排列的
        distance_mask = np.load(join(distance_dir, i.split('.nii.gz')[0]+'.npz'))['data'] 
        skeleton_mask = nib.load(join(skeleton_dir,i)).get_fdata()
        nib_mask_data = nib_mask.get_fdata()
        voxel_size = nib_mask.shape
        spacing = nib_mask.header.get_zooms()

        # 提取到半径介于5到11之间
        min_radius = 5
        max_radius = 11
        min_radius = int(min_radius * 0.2/0.15)
        max_radius = int(max_radius * 0.2/0.15)
        skeleton_point_cloud = np.vstack([np.where(skeleton_mask>0)]).T
        x_coords, y_coords, z_coords = skeleton_point_cloud[:, 0], skeleton_point_cloud[:, 1], skeleton_point_cloud[:, 2]
        local_radii = distance_mask[x_coords, y_coords, z_coords]
        thick_indices = (local_radii >= min_radius) & (local_radii < max_radius)
        ok_point_cloud = skeleton_point_cloud[thick_indices,:]
        points_filter, components = remove_small_components(ok_point_cloud, k = 5, max_connect_dist = 2, min_component_size=40)

        segment_25mms = get_segments(components, spacing, target_length, number_segments)
        if(segment_25mms == []): #如果选不到合适1的segment，就去找下一个
            print(f'{i}  无法选择到合适的血管, 尝试调整一下参数')
            continue
        segment_25mms_copy = copy.deepcopy(segment_25mms)
        segment_25mms = [i / np.array(spacing).reshape(1,3) for i in segment_25mms_copy] #得到物理体素坐标
        segments_config = dict()
        for segment_idx in range(len(segment_25mms)):
            ## 获得目标中心线
            segment = segment_25mms[segment_idx]
            proceed_points = process_point_cloud(segment.astype(np.float32), target_count=segment_target_points) #对于我这个任务来说，不进行排序反而效果更好
            proceed_points_smooth = smooth_curve(proceed_points, window=smooth_window) #得到平滑之后的中心线
            curve_len = compute_curve_length(proceed_points_smooth) #这里计算的是体素距离
            nz_voxels = int(curve_len)+1
            curve_dist = mean_of_8_neighbors_batch(distance_mask, proceed_points_smooth, False) #得到平滑之后的曲线的半径
            modify_curve_dist = curve_dist * 0.9 #半径适当的减少一些

            ## 接下来创建支架
            stent_idx = random.randint(0, len(generators)-1)
            stent_generator, stent_name = generators[stent_idx], stent_names[stent_idx]
            straight_z_stent = stent_generator.create_straight_stent_along_z(nz_voxels) #生成的都是体素空间的支架
            straight_z_stent = change_radius(modify_curve_dist, spacing, straight_z_stent)
            
            ## 将支架配准到给定的中心线
            curve_point = np.copy(proceed_points_smooth)
            # stent_point = np.copy(straight_z_stent / np.array(spacing)) # 要将其从物理空间转换到体素空间
            stent_point_deform = map_stent_to_curve(straight_z_stent, curve_point)
            
            ## 将支架体素化
            voxel_stent_point_deform = voxelize_1(stent_point_deform, voxel_size)

            ## 保存应该进行截取的配置等
            x_min, x_max, y_min, y_max, z_min, z_max = np.min(stent_point_deform[:,0]), np.max(stent_point_deform[:,0]), np.min(stent_point_deform[:,1]), np.max(stent_point_deform[:,1]), np.min(stent_point_deform[:,2]), np.max(stent_point_deform[:,2])
            x_center, y_center, z_center = int(x_min + x_max) / 2, int(y_min + y_max) / 2, int(z_min + z_max) / 2 
            segments_config[str(segment_idx)] = dict()
            segments_config[str(segment_idx)]["center"] = [x_center, y_center, z_center]
            segments_config[str(segment_idx)]["border"] = [x_min, x_max, y_min, y_max, z_min, z_max]
            segments_config[str(segment_idx)]["stent_name"] = stent_name

            ## 对支架进行后处理（主要是改变支架线条的大小）。目前还是决定暂时用thicken_by_distance的方法进行扩张
            start_indices = np.clip((np.array((x_min,y_min,z_min)) *0.8).astype(np.int16),  0, voxel_size)
            end_indices = np.clip((np.array((x_max,y_max,z_max)) *1.2).astype(np.int16), 0, voxel_size)

            sub_voxel = voxel_stent_point_deform[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] #将这一部分数据提取出来
            sub_vessel = nib_mask_data[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] #同时取出对应的血管，便于后续的处理
            # expand_sub_voxel, sub_voxel_skeleton, flag = thicken_by_distance(sub_voxel, r_voxel = 1) #尝试扩大其半径
            expand_sub_voxel, flag = stent_expand(sub_voxel, stent_name)

            if(not flag): #说明输入的很可能是空数组
                warnings.warn(f"{i} 得到的很可能是空的数组. 一定要检查一下")

            ## 限制支架不要超过血管范围
            outside_mask = (expand_sub_voxel==1) & (sub_vessel==0)
            expand_sub_voxel[outside_mask] = 0

            ## 将改变的部分重新赋值进去
            voxel_stent_point_deform[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] = expand_sub_voxel #将出来之后的数据放回原来的地方

            ## 创建并保存支架nib文件
            stent_nib = nib.Nifti1Image(voxel_stent_point_deform.astype(np.int16), affine=nib_mask.affine, header=nib_mask.header)
            save_name = i.split('.nii.gz')[0] + f'_sample_{str(segment_idx).zfill(2)}.nii.gz'
            nib.save(stent_nib, join(stent_save_path, save_name))

        ## 每次创建完成一个，就应该同步写入配置文件
        with open(join(config_save_path, f"{i.split('.nii.gz')[0]}.pickle"), 'wb') as f:
            pickle.dump(segments_config, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_dir",type=str, default='/data2/liuyang/liuyang/stent/interior/data/costa/helper')
    parser.add_argument("--stent_ply_dir", type=str, default='/data2/liuyang/liuyang/stent/interior/corr_data/gen_stent/ply')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("-radius", type=int, default=64)
    args = parser.parse_args()

    data_dir = args.data_dir
    stent_ply_dir = args.stent_ply_dir
    save_path = args.save_path
    resulotion = args.resolution
    radius = args.radius

    ## 生成训练和测试数据
    print('生成训练数据')
    mode='train'
    final_save_dir = join(save_path, f'voxel_{resulotion}/radius-{radius}/corr_stent/{mode}_data')
    final_data_dir = join(data_dir, f'{mode}_data/voxel_{resulotion}')
    final_stent_ply_dir = join(stent_ply_dir, mode)
    main(data_dir=final_data_dir, stent_ply_dir=final_stent_ply_dir, save_dir=final_save_dir, mode=mode)

    ## 生成训练和测试数据
    print('生成测试数据')
    mode='test'
    final_save_dir = join(save_path, f'voxel_{resulotion}/radius-{radius}/corr_stent/{mode}_data')
    final_data_dir = join(data_dir, f'{mode}_data/voxel_{resulotion}')
    final_stent_ply_dir = join(stent_ply_dir, mode)
    main(data_dir=final_data_dir, stent_ply_dir=final_stent_ply_dir, save_dir=final_save_dir, mode=mode)


