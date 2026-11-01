import numpy as np
import plyfile, copy, pickle, os, random, json, argparse, warnings
import pandas as pd
import tifffile as tiff
from scipy import io
join = os.path.join
from PIL import Image
import matplotlib.pyplot as plt
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

from utils.vessel import (
    get_roi
)

from utils.file_rw import save_sparse_projections

from utils.stent import (
    get_segments,
    get_stent_generator_train,
    get_stent_generator_test,
    stent_expand
)


def gen_stent(segment, stent_generator, stent_names, distance_mask, nib_mask, segment_target_points=2000, smooth_window=800, spacing=0.2, voxel_size=(0 ,0 ,0)):
    proceed_points = process_point_cloud(segment.astype(np.float32), target_count=segment_target_points) #对于我这个任务来说，不进行排序反而效果更好
    proceed_points_smooth = smooth_curve(proceed_points, window=smooth_window) #得到平滑之后的中心线
    curve_len = compute_curve_length(proceed_points_smooth) #这里计算的是体素距离
    nz_voxels = int(curve_len)+1
    curve_dist = mean_of_8_neighbors_batch(distance_mask, proceed_points_smooth, False) #得到平滑之后的曲线的半径
    modify_curve_dist = curve_dist * 0.9 #半径适当的减少一些

    ## 接下来创建支架
    stent_idx = random.randint(0, len(stent_generator)-1)
    stent_generator, stent_name = stent_generator[stent_idx], stent_names[stent_idx]
    straight_z_stent = stent_generator.create_straight_stent_along_z(nz_voxels) #生成的都是体素空间的支架
    straight_z_stent = change_radius(modify_curve_dist, spacing, straight_z_stent)
    
    ## 将支架配准到给定的中心线
    curve_point = np.copy(proceed_points_smooth)
    stent_point_deform = map_stent_to_curve(straight_z_stent, curve_point)
    
    ## 将支架体素化
    voxel_stent_point_deform = voxelize_1(stent_point_deform, voxel_size)

    ## 保存应该进行截取的配置等
    x_min, x_max, y_min, y_max, z_min, z_max = np.min(stent_point_deform[:,0]), np.max(stent_point_deform[:,0]), np.min(stent_point_deform[:,1]), np.max(stent_point_deform[:,1]), np.min(stent_point_deform[:,2]), np.max(stent_point_deform[:,2])
    x_center, y_center, z_center = int(x_min + x_max) / 2, int(y_min + y_max) / 2, int(z_min + z_max) / 2 
    infor = dict()
    infor["center"] = [x_center, y_center, z_center]
    infor["border"] = [x_min, x_max, y_min, y_max, z_min, z_max]
    infor["stent_name"] = stent_name

    ## 对支架进行后处理（主要是改变支架线条的大小）。目前还是决定暂时用thicken_by_distance的方法进行扩张
    start_indices = np.clip((np.array((x_min,y_min,z_min)) *0.8).astype(np.int16),  0, voxel_size)
    end_indices = np.clip((np.array((x_max,y_max,z_max)) *1.2).astype(np.int16), 0, voxel_size)
    
    sub_voxel = voxel_stent_point_deform[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] #将这一部分数据提取出来
    sub_vessel = nib_mask.dataobj[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] # 取出Nib中的数据
    expand_sub_voxel, flag = stent_expand(sub_voxel, stent_name)

    ## 限制支架不要超过血管范围
    outside_mask = (expand_sub_voxel==1) & (sub_vessel==0)
    expand_sub_voxel[outside_mask] = 0

    ## 将改变的部分重新赋值进去
    voxel_stent_point_deform[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] = expand_sub_voxel #将出来之后的数据放回原来的地方

    ## 创建并保存支架nib文件
    stent_nib = nib.Nifti1Image(voxel_stent_point_deform.astype(np.int16), affine=nib_mask.affine, header=nib_mask.header)

    return stent_nib, infor, flag

def main(data_dir, stent_save_dir, vessel_save_dir, stent_ply_dir, radius=63, mode='train'):
    ## 一些提前就创建完成的文件
    resample_dir = join(data_dir, 'resample')
    distance_dir = join(data_dir, 'distance_map')
    skeleton_dir = join(data_dir, 'skeleton')

    ## vessel的保存位置
    vessel_vol_gt_save_path = join(vessel_save_dir,'vol_gt')
    vessel_proj_gt_save_path = join(vessel_save_dir,'proj_gt')
    vessel_vol_gt_mask_save_path = join(vessel_save_dir,'vol_gt_mask')
    os.makedirs(vessel_vol_gt_save_path, exist_ok=True)
    os.makedirs(vessel_proj_gt_save_path, exist_ok=True)
    os.makedirs(vessel_vol_gt_mask_save_path, exist_ok=True)

    ## stent的保存位置
    stent_vol_gt_save_path = join(stent_save_dir,'vol_gt')
    stent_proj_gt_save_path = join(stent_save_dir,'proj_gt')
    stent_vol_gt_mask_save_path = join(stent_save_dir,'vol_gt_mask')
    config_save_path = join(stent_save_dir,'config')
    os.makedirs(config_save_path, exist_ok=True)
    os.makedirs(stent_vol_gt_save_path, exist_ok=True)
    os.makedirs(stent_proj_gt_save_path, exist_ok=True)
    os.makedirs(stent_vol_gt_mask_save_path, exist_ok=True)

    number_segments = 10 #每例数据要选10个ROI
    target_length = 25 #生成的支架的长度, 单位是mm
    segment_target_points = 2000 #每条线段的应该有的点的数量
    smooth_window = 800 #对线段进行平滑的窗口

    if(mode=='train'):
        generators, stent_names = get_stent_generator_train(stent_ply_dir)
    else:
        generators, stent_names = get_stent_generator_test(stent_ply_dir)

    to_process = sorted(os.listdir(resample_dir))

    for i in tqdm(to_process):
        # if(mode=='train' and i !='IXI_Guys_074.nii.gz'):
        #     continue
        ## 加载必要的数据
        nib_mask = nib.load(join(resample_dir,i)) #(w,h,d). 假设w是x的话，那么h就是y。那么确实是 (x,y,z)排列的
        distance_mask = np.load(join(distance_dir, i.split('.nii.gz')[0]+'.npz'))['data'] 
        skeleton_mask = nib.load(join(skeleton_dir,i)).get_fdata()
        nib_mask_data = nib_mask.get_fdata()
        voxel_size = nib_mask.shape
        spacing = nib_mask.header.get_zooms()

        # 根据骨架图提取到半径介于5到11之间的血管骨架
        min_radius = 5
        max_radius = 11
        # min_radius = int(min_radius * 0.2/)
        # max_radius = int(max_radius * 0.2/0.15)
        skeleton_point_cloud = np.vstack([np.where(skeleton_mask>0)]).T
        x_coords, y_coords, z_coords = skeleton_point_cloud[:, 0], skeleton_point_cloud[:, 1], skeleton_point_cloud[:, 2]
        local_radii = distance_mask[x_coords, y_coords, z_coords]
        thick_indices = (local_radii >= min_radius) & (local_radii < max_radius)
        ok_point_cloud = skeleton_point_cloud[thick_indices,:]
        points_filter, components = remove_small_components(ok_point_cloud, k = 5, max_connect_dist = 2, min_component_size=40)

        ## 对能否创建支架进行判断
        segment_25mms = get_segments(components, spacing, target_length, number_segments)
        if(segment_25mms == []): #如果选不到合适1的segment，就去找下一个
            print(f'{i}  无法选择到合适的血管, 尝试调整一下参数')
            continue
        segment_25mms_copy = copy.deepcopy(segment_25mms)
        segment_25mms = [i / np.array(spacing).reshape(1,3) for i in segment_25mms_copy] #得到物理体素坐标

        ## 设置投影相关的几何参数
        angles = np.linspace(0, 2*np.pi, 360)
        radius = radius
        x_dim_max, y_dim_max, z_dim_max = nib_mask.shape[0], nib_mask.shape[1], nib_mask.shape[2], 
        x_voxel_size, y_voxel_size, z_voxel_size = [round(i, 4) for i in  nib_mask.header['pixdim'][1:4]] #voxel space=
        geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
            dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
            nDetector=np.array([1024*2, 1024*2]), #投影要大一点，因为重采样后的数据会很大
            dDetector=np.array([x_voxel_size, y_voxel_size]))
        src_vessel_proj = tigre.Ax(np.transpose(nib_mask_data.astype(np.float32), [2, 0, 1]), geo, angles) #原始图像的投影

        segments_config = dict()
        for segment_idx in range(len(segment_25mms)):
            ## 获得目标中心线
            segment = segment_25mms[segment_idx]
            
            ## 创建支架
            stent_nib, infor_dict, flag = gen_stent(segment, generators, stent_names, distance_mask, nib_mask, segment_target_points, smooth_window, spacing, voxel_size)
            segments_config[segment_idx] = infor_dict
            if(not flag): #说明输入的很可能是空数组
                warnings.warn(f"{i} 得到的很可能是空的数组. 一定要检查一下")

            ## 来截取血管
            idx_center = infor_dict['center']
            idx_center = np.array(idx_center).astype(np.int16)
            # 获得该中心点对应的roi. 图像都是x y z的顺序, 而投影都是转换到z, x, y的顺序去投的
            vessel_roi_proj, vessel_roi_gt, roi_image_mask = get_roi(mask=nib_mask_data, sample_point=idx_center, src_proj=src_vessel_proj, angles=angles, radius=radius, geo=geo) #与原始投影、原始图像的大小是一样的，但是只有ROI的区域有值，其他地方都是0
            vessel_roi_gt_nib  = nib.Nifti1Image(
                dataobj=vessel_roi_gt,
                affine=nib_mask.affine,
                header=nib_mask.header
            )
            roi_gt_mask_nib  = nib.Nifti1Image(
                dataobj=roi_image_mask,
                affine=nib_mask.affine,
                header=nib_mask.header
            )

            ## 来截取支架. 由于支架已经是局部创建的，因此没必要结合roi_mask来提取投影, 直接对其投影即可
            src_stent_proj = tigre.Ax(np.transpose(stent_nib.dataobj.astype(np.float32), [2, 0, 1]), geo, angles) #原始图像的投影

            save_name = f"{i.split('.')[0]}_sample_{str(segment_idx).zfill(2)}"
            ## 保存vessel文件
            nib.save(vessel_roi_gt_nib, join(vessel_vol_gt_save_path, save_name+'.nii.gz'))
            nib.save(roi_gt_mask_nib, join(vessel_vol_gt_mask_save_path, save_name+'.nii.gz'))
            save_sparse_projections(proj=vessel_roi_proj, out_dir=join(vessel_proj_gt_save_path, save_name))

            ## 保存stent文件
            save_name = i.split('.nii.gz')[0] + f'_sample_{str(segment_idx).zfill(2)}'
            nib.save(stent_nib, join(stent_vol_gt_save_path, save_name+'.nii.gz'))
            nib.save(roi_gt_mask_nib, join(stent_vol_gt_mask_save_path, save_name+'.nii.gz'))
            save_sparse_projections(proj=src_stent_proj, out_dir=join(stent_proj_gt_save_path, save_name))
            
        ## 每次创建完成一个，就应该同步写入配置文件
        with open(join(config_save_path, f"{i.split('.nii.gz')[0]}.pickle"), 'wb') as f:
            pickle.dump(segments_config, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--helper_path",type=str, default='/data1/liuyang/stent/interior/data/other/topbrain_ct/helper')
    parser.add_argument("--stent_ply_dir", type=str, default='/data1/liuyang/stent/interior/data/corr_data/gen_stent/ply')
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--radius", type=int, default=64)
    args = parser.parse_args()

    helper_path = args.helper_path
    stent_ply_dir = args.stent_ply_dir
    save_path = args.save_path
    resulotion = args.resolution
    radius = args.radius

    # 生成训练和测试数据
    print('生成训练数据')
    mode='train'
    stent_save_dir = join(save_path, f'voxel_{resulotion}/src_data/radius-{radius}/corr_stent/{mode}_data')
    vessel_save_dir = join(save_path, f'voxel_{resulotion}/src_data/radius-{radius}/corr_vessel/{mode}_data')
    final_helper_path = join(helper_path, f'{mode}_data/voxel_{resulotion}')
    final_stent_ply_dir = join(stent_ply_dir, mode)
    main(data_dir=final_helper_path, stent_ply_dir=final_stent_ply_dir, stent_save_dir=stent_save_dir, vessel_save_dir=vessel_save_dir, radius=radius, mode=mode)

    print('生成测试数据')
    mode='test'
    stent_save_dir = join(save_path, f'voxel_{resulotion}/src_data/radius-{radius}/corr_stent/{mode}_data')
    vessel_save_dir = join(save_path, f'voxel_{resulotion}/src_data/radius-{radius}/corr_vessel/{mode}_data')
    final_helper_path = join(helper_path, f'{mode}_data/voxel_{resulotion}')
    final_stent_ply_dir = join(stent_ply_dir, mode)
    main(data_dir=final_helper_path, stent_ply_dir=final_stent_ply_dir, stent_save_dir=stent_save_dir, vessel_save_dir=vessel_save_dir, radius=radius, mode=mode)


