import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import plyfile,copy,argparse
import pandas as pd
import tifffile as tiff
import os
join = os.path.join
import open3d as o3d
from matplotlib.colors import Normalize
import SimpleITK as sitk
import tigre
import tigre.algorithms as algs
import open3d as o3d
import tqdm
import SimpleITK as sitk # 推荐用于医学图像处理
from skimage.morphology import skeletonize_3d
from skimage.measure import regionprops,label
import json

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

def get_rectangle_roi(roi_center,roi_size): #获得长方形区域的roi坐标
    # box_min = (roi_center - roi_size / 2.0).astype(np.uint8)
    # box_max = (roi_center + roi_size / 2.0).astype(np.uint8)
    box_min = (roi_center - roi_size / 2.0).astype(np.uint16)
    box_max = (roi_center + roi_size / 2.0).astype(np.uint16)
    x_coords, y_coords ,z_coords= np.mgrid[
        box_min[0] : box_max[0] + 1,
        box_min[1] : box_max[1] + 1,
        box_min[2] : box_max[2] + 1
    ]
    return np.vstack([x_coords.ravel(), y_coords.ravel() ,z_coords.ravel()])

def get_circle_roi(image_shape,roi_center,roi_radius): #获得圆形区域的roi坐标
    Nz, Nx, Ny = image_shape
    center_z, center_x, center_y = roi_center
    z_coords, x_coords, y_coords = np.mgrid[0:Nz, 0:Nx, 0:Ny]
    
    #计算每个体素到圆心在 XY 平面上的距离平方
    distance_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2 + (z_coords - center_z)**2
    radius_sq = roi_radius**2
    mask = distance_sq <= radius_sq

    mask = mask.astype(np.uint8)
    z,x,y = np.where(mask>0)
    return np.vstack([z,x,y])

def get_roi_proj(image,geo,angles,valid_mesh):
    # 输入的mesh的shape要是(3,num_of_points)
    roi_image = np.zeros_like(image,np.float32)
    z_coords, x_coords ,y_coords = valid_mesh[0],valid_mesh[1],valid_mesh[2]
    roi_image[z_coords, x_coords, y_coords] = 1 #创建一个只有ROI区域有值的体膜
    roi_mask = tigre.Ax(roi_image,geo,angles)
    proj_data = tigre.Ax(image,geo,angles)
    roi_mask[roi_mask>0.01] = 1 #将其转换为mask
    roi_mask[roi_mask<=0.01] = 0
    roi_proj = proj_data * roi_mask #逐元素相乘提取数据
    return roi_proj,proj_data


## 找到满足要求的动脉区域（最有可能发生脑梗塞的区域）
def find_counterpart(mask,main_artery_indices):
    mask = np.round(mask).astype(np.uint8)
    mask_i = np.isin(mask, main_artery_indices)
    mask_filtered = np.where(mask_i, mask, 0)
    return mask_filtered

## 生成ROI区域的外接圆（半径应该稍微设置大一点）
def get_filled_bounding_cube_from_sphere(
    center: np.ndarray,
    radius: float,
    resolution_steps: int = 200  # 正方体每个轴上采样的点数 (例如 32x32x32)
) -> np.ndarray:
    """
    根据球体中心和半径，直接生成其外接正方体内部所有点的点云表示。

    Args:
        center: 球体的中心坐标 (Cx, Cy, Cz)。
        radius: 球体的半径 R。
        resolution_steps: 正方体每个轴上均匀采样的点数 (体素分辨率 M)。

    Returns:
        filled_cube_points: 填充了整个正方体体积的点云，形状为 (M^3, 3)。
    """
    
    C_z, C_x, C_y = center
    R = radius

    # --- 1. 确定外接正方体的边界 ---
    
    # 最小和最大边界
    X_min, X_max = C_x - R, C_x + R
    Y_min, Y_max = C_y - R, C_y + R
    Z_min, Z_max = C_z - R, C_z + R
    
    # 检查基本条件
    if R <= 0:
        print("警告：半径必须大于0。返回空点云。")
        return np.empty((0, 3))
    if resolution_steps < 2:
        # 至少需要2个点来定义范围
        resolution_steps = 2
        
    # --- 2. 生成填充整个正方体体积的点云 ---
    
    # 在每个轴上生成均匀分布的坐标
    # 在 [Min, Max] 范围内生成 M 个点
    x_grid = np.linspace(X_min, X_max, resolution_steps)
    y_grid = np.linspace(Y_min, Y_max, resolution_steps)
    z_grid = np.linspace(Z_min, Z_max, resolution_steps)
    
    # 使用 np.meshgrid 生成所有组合的坐标
    # indexing='ij' 适用于图像和体素处理，确保 XYZ 轴正确对应
    Z, X, Y = np.meshgrid(z_grid, x_grid, y_grid, indexing='ij')

    # 将三维网格展平为 (M^3, 3) 的点云形式
    filled_cube_points = np.stack([Z.flatten(), X.flatten(), Y.flatten()], axis=1)
    
    return filled_cube_points


def voxelize(
    point_cloud: np.ndarray, 
    voxel_size = (256, 128, 128), 
    fill_value = 1
) -> np.ndarray:
    """
    使用 NumPy 矢量化操作快速将点云体素化为占用网格。

    Args:
        point_cloud: 形状为 (N, 3) 的点云数组，坐标顺序为 (Z, X, Y)。
        voxel_size: 体素网格的维度 (Depth, Height, Width) 或 (Z, X, Y)。
        fill_value: 用于标记占用体素的值 (通常是 1)。

    Returns:
        voxel: 形状为 voxel_size 的体素占用网格 (NumPy 数组)。
    """
    
    # 1. 坐标取整并转换为整数索引
    indices = np.round(point_cloud).astype(int) 
    
    Z_indices = indices[:, 0]
    X_indices = indices[:, 1]
    Y_indices = indices[:, 2]

    D, H, W = voxel_size # D=Z_max, H=X_max, W=Y_max
    
    # 确保索引在 [0, size-1] 范围内
    valid_mask = (Z_indices >= 0) & (Z_indices < D) & \
                 (X_indices >= 0) & (X_indices < H) & \
                 (Y_indices >= 0) & (Y_indices < W)

    # 3. 筛选有效索引
    valid_Z = Z_indices[valid_mask]
    valid_X = X_indices[valid_mask]
    valid_Y = Y_indices[valid_mask]
    
    # 将三个坐标数组堆叠起来，形成 (M, 3) 形状的有效索引数组
    valid_indices = (valid_Z, valid_X, valid_Y)
    
    # 4. 创建并填充体素网格
    # 初始化体素网格
    voxel = np.zeros(voxel_size, dtype=np.float32) 
    
    # 利用 NumPy 的高级索引（矢量化）一次性设置所有有效体素的值
    # 这是关键步骤，取代了 Python 循环
    voxel[valid_indices] = fill_value
    
    return voxel

def sample_on_original_points_pca(points: np.ndarray, num_samples: int = 10):
    """
    通过 PCA 找到最长轴，将点云排序后，沿着该轴均匀选择原始点云上的点。
    
    参数:
        points: np.ndarray (N, 3) 原始点云坐标
        num_samples: int，采样数量
    返回:
        samples: np.ndarray (num_samples, 3) 选自原始点云的坐标
    """
    if len(points) < num_samples:
        return points # 点太少，返回全部

    # 1. PCA 分解 (与原代码相同)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    main_dir = eigvecs[:, 0]

    # 2. 投影到主轴，并根据投影值排序
    projections = centered @ main_dir
    # 获取排序后的索引：沿着主轴从小到大排列
    sorted_indices = np.argsort(projections)

    # 3. 均匀采样索引
    total_points = len(points)
    
    # 确定采样间隔
    interval = total_points / num_samples
    
    # 确定一个随机起始偏移量 (确保随机性)
    start_offset = np.random.uniform(0, interval)

    # 生成 num_samples 个均匀采样的浮点索引
    floating_indices = np.arange(num_samples) * interval + start_offset
    
    # 将浮点索引转换为整数索引 (并取模确保不越界)
    sample_indices = np.floor(floating_indices).astype(np.int32) % total_points

    # 4. 从排序后的索引中取出采样的点
    # 选中排序后列表中的这些点
    selected_indices_in_original = sorted_indices[sample_indices]
    
    # 返回原始坐标
    samples_on_original = points[selected_indices_in_original]
    
    return samples_on_original

def extract_centerlines(main_artery_mask, min_length=10):
    """
    对三维二值掩码进行骨架化，并分离出多条独立的中心线。

    参数:
        main_artery_mask (np.ndarray): 主血管的三维布尔/二值掩码 (D, H, W)。
        min_length (int): 过滤掉长度小于此值的短小中心线段。

    返回:
        centerlines_list (list): 包含每条独立中心线坐标点的列表。
        skeleton_labeled (np.ndarray): 带有标签的中心线三维图像 (D, H, W)。
    """
    
    if main_artery_mask.ndim != 3:
        raise ValueError("输入掩码必须是三维数组。")
    skeleton = skeletonize_3d(main_artery_mask)
    skeleton_labeled, num_labels = label(skeleton, return_num=True)
    centerlines_list = []
    for region in regionprops(skeleton_labeled):
        if region.area >= min_length:
            coords = region.coords
            centerlines_list.append(coords)
    return centerlines_list, skeleton_labeled

def get_roi(mask, sample_point, para_path, radius=50, angles = np.linspace(0,2*np.pi,360)):
    """
    mask: 图像的mask
    sample_point: 一个中心点
    para_path: 图像的参数的文件
    radius: 采样圆的半径
    """
    ## 处理单个对象，得到其截断图像（非截断区域设置为0）

    ## 设置几何
    x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = read_para(para_path)
    geo = set_geo(nVoxel=np.array([z_dim_max, x_dim_max, y_dim_max]),
                dVoxel=np.array([z_voxel_size, x_voxel_size, y_voxel_size]),
                nDetector=np.array([1024,1024]),
                dDetector=np.array([0.5,0.5]))

    # for index,point in enumerate(sample_points):
    roi_center = np.array([sample_point[0], sample_point[1], sample_point[2]])
    ## 这里的image应该是mask
    point_mesh = get_circle_roi(mask.shape,roi_center,radius) # (3, N)的点云形式 #真值

    ## 进行投影
    roi_proj,_ = get_roi_proj(mask,geo,angles,point_mesh)

    ## 将ROI的点云形式转换为体素形式的mask
    mask_mask = np.zeros([z_dim_max, x_dim_max, y_dim_max], dtype=bool)
    for i in point_mesh.T:
        z_idx = int(round(i[0]))
        x_idx = int(round(i[1]))
        y_idx = int(round(i[2]))
        mask_mask[z_idx, x_idx, y_idx] = True
    roi_gt = np.where(mask_mask, mask, 0)

    return roi_proj,roi_gt

def read_para(para_path):
    # 读取几何参数
    with open(para_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
        x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = loaded_data['x_voxel_size'], loaded_data['y_voxel_size'], loaded_data['z_voxel_size'], loaded_data['x_dim_max'], loaded_data['y_dim_max'], loaded_data['z_dim_max']
    return float(x_voxel_size), float(y_voxel_size), float(z_voxel_size), int(x_dim_max), int(y_dim_max), int(z_dim_max)


# 下面这个work的思路是直接得到真实图像和训练图像，但是应该是先得到完整的投影和图像，再选择角度去反投影。最后一步才是裁剪，
def get_sample_proj_vol(mask_dir,para_path, save_dir = './', number_roi_in_one_sample = 10, save_file = False, radius=60, already_has = None):
    # 设置保存路径
    centerline_save_path = join(save_dir,'centerline')
    point_cloud_save_path = join(save_dir,'point_cloud')
    print(centerline_save_path)
    os.makedirs(centerline_save_path, exist_ok=True)
    os.makedirs(point_cloud_save_path, exist_ok=True)
    print(point_cloud_save_path)
    # if(save_file):
    # 如果需要保留这些中间文件
    vol_gt_save_path = join(save_dir,'vol_gt')
    proj_gt_save_path = join(save_dir,'proj_gt')
    # vol_train_save_path = join(save_path,'vol_train')
    # proj_train_save_path = join(save_path,'proj_train')
    os.makedirs(vol_gt_save_path, exist_ok=True)
    os.makedirs(proj_gt_save_path, exist_ok=True)
    # os.makedirs(vol_train_save_path, exist_ok=True)
    # os.makedirs(proj_train_save_path, exist_ok=True)

    # x_voxel_size, y_voxel_size, z_voxel_size, x_dim_max, y_dim_max, z_dim_max = read_para(para_path)

    # 规划主要动脉标签
    main_artery = ['L-M1','R-M1','L-A1A2','R-A1A2','L-A1A2','BA'] # 大脑中动脉、前动脉、后动脉、基地动脉都是比较任意发生脑梗塞的区域
    main_artery_indices = []
    LABEL_TO_IDX = {data['LABEL']: idx for idx, data in BRAIN_VESSEL_LABELS_MR.items()}
    for label in main_artery:
        idx = LABEL_TO_IDX.get(label)
        if idx is not None:
            main_artery_indices.append(idx)
        else:
            print(f"警告：标签 '{label}' 未在字典中找到。")

    # 其他参数
    angles = np.linspace(0,2*np.pi,360)
    radius = radius


    ## 对每个mask，都在主动脉区域选取10个点，并截取以这个点为圆心的圆形区域作为训练数据
    # num = 0
    for i in os.listdir(mask_dir):
        # 取得主要动脉的中心线
        if(already_has is not None and i in already_has):
            print('already: ', i)
            continue

        mask = tiff.imread(join(mask_dir,i))
        main_artery_mask = find_counterpart(mask,main_artery_indices)
        _, skeleton_labeled = extract_centerlines(main_artery_mask)
        tiff.imwrite(join(centerline_save_path,i),skeleton_labeled.astype(np.uint8)) #保留中心线文件

        # 在这些中心线上均匀且随机的取10个点
        point_cloud = np.vstack([np.where(skeleton_labeled>0)]).T #skeleton_labeled 来自于mask, 因此point_cloud的点坐标是(z,x,y), 因此sample_points的点坐标也是(z,x,y)
        sample_points = sample_on_original_points_pca(point_cloud, num_samples=number_roi_in_one_sample)
        
        # 保存选取的点
        convert_point_cloud_to_ply(sample_points,join(point_cloud_save_path,i.split('.')[0]+'.ply'))

        # 以点为中心取ROI区域
        mask[mask>0] = 1 #将mask置为0-12
        
        # 首先将roi区域内的mask内容提取出来，作为图像的真值
        for sample_index, sample_point in enumerate(sample_points):
            # 获得该中心点对应的roi
            roi_proj,roi_gt = get_roi(mask=mask, sample_point=sample_point, para_path=para_path, angles=angles, radius=radius) #与原始投影、原始图像的大小是一样的，但是只有ROI的区域有值，其他地方都是0
            
            # save_name = f"{i.split('.')[0]}_sample_{str(sample_idnex).zfill(2)}.tif"
            # tiff.imwrite(join(vol_gt_save_path,save_name),roi_gt)
            # tiff.imwrite(join(proj_gt_save_path,save_name),roi_proj)

            save_name = f"{i.split('.')[0]}_sample_{str(sample_index).zfill(2)}.npz"
            np.savez_compressed(join(vol_gt_save_path,save_name),data=roi_gt)
            np.savez_compressed(join(proj_gt_save_path,save_name),data=roi_proj)

            # # 重建投影
            # angles = np.linspace(0,2*np.pi,360)
            # recon_img = recon_single_object(roi_proj, angles, para_path=para_path)
            
            # # 将有效区域截取出来
            # truncate_image(vol_train, vol_gt, sample_point, para_path, save_dir, radius=50, more_radius = 5):
        print(i)
        # break
        # num += 1
        # if(num==2):
        #     break

def read_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if not pcd.has_points():
        print(f"警告: 文件 {ply_path} 中没有读取到任何点。")
        return pcd, np.array([])
        
    # 将 Open3D 点云对象转换为 NumPy 数组
    point_cloud_np = np.asarray(pcd.points)
    return  point_cloud_np


def convert_point_cloud_to_ply(point_cloud,file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(file_path, pcd)

if __name__ == "__main__":
    ## 对于这段代码，确实不需要more_radius, 这个值是后面在截取truncate图像的时候才会用到的。这里只是生成给定半径下的roi投影以及重建图像

    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius",type=int)
    parser.add_argument("--number_roi_in_one_sample", type=int, default=10)
    args = parser.parse_args()
    radius = args.radius
    number_roi_in_one_sample = args.number_roi_in_one_sample

    BRAIN_VESSEL_LABELS_MR = {
        # 颅内动脉及静脉窦 (IDX 0-34, 43个)
        0:  {"LABEL": "Clear Label", "FULL_NAME_EN": "N/A", "FULL_NAME_ZH": "清除标签 / 背景", "TYPE": "Background"},
        1:  {"LABEL": "BA", "FULL_NAME_EN": "Basilar Artery", "FULL_NAME_ZH": "基底动脉", "TYPE": "Artery (Posterior)"},
        2:  {"LABEL": "R-P1P2", "FULL_NAME_EN": "Right Posterior Cerebral Artery, segments P1 & P2", "FULL_NAME_ZH": "右侧大脑后动脉 P1 和 P2 段", "TYPE": "Artery (Posterior)"},
        3:  {"LABEL": "L-P1P2", "FULL_NAME_EN": "Left Posterior Cerebral Artery, segments P1 & P2", "FULL_NAME_ZH": "左侧大脑后动脉 P1 和 P2 段", "TYPE": "Artery (Posterior)"},
        4:  {"LABEL": "R-ICA", "FULL_NAME_EN": "Right Internal Carotid Artery", "FULL_NAME_ZH": "右侧颈内动脉", "TYPE": "Artery (Anterior)"},
        5:  {"LABEL": "R-M1", "FULL_NAME_EN": "Right Middle Cerebral Artery, segment M1", "FULL_NAME_ZH": "右侧大脑中动脉 M1 段", "TYPE": "Artery (Anterior)"},
        6:  {"LABEL": "L-ICA", "FULL_NAME_EN": "Left Internal Carotid Artery", "FULL_NAME_ZH": "左侧颈内动脉", "TYPE": "Artery (Anterior)"},
        7:  {"LABEL": "L-M1", "FULL_NAME_EN": "Left Middle Cerebral Artery, segment M1", "FULL_NAME_ZH": "左侧大脑中动脉 M1 段", "TYPE": "Artery (Anterior)"},
        8:  {"LABEL": "R-Pcom", "FULL_NAME_EN": "Right Posterior Communicating Artery", "FULL_NAME_ZH": "右侧后交通动脉", "TYPE": "Artery (Communicating)"},
        9:  {"LABEL": "L-Pcom", "FULL_NAME_EN": "Left Posterior Communicating Artery", "FULL_NAME_ZH": "左侧后交通动脉", "TYPE": "Artery (Communicating)"},
        10: {"LABEL": "Acom", "FULL_NAME_EN": "Anterior Communicating Artery", "FULL_NAME_ZH": "前交通动脉", "TYPE": "Artery (Communicating)"},
        11: {"LABEL": "R-A1A2", "FULL_NAME_EN": "Right Anterior Cerebral Artery, segments A1 & A2", "FULL_NAME_ZH": "右侧大脑前动脉 A1 和 A2 段", "TYPE": "Artery (Anterior)"},
        12: {"LABEL": "L-A1A2", "FULL_NAME_EN": "Left Anterior Cerebral Artery, segments A1 & A2", "FULL_NAME_ZH": "左侧大脑前动脉 A1 和 A2 段", "TYPE": "Artery (Anterior)"},
        13: {"LABEL": "R-A3", "FULL_NAME_EN": "Right Anterior Cerebral Artery, segment A3", "FULL_NAME_ZH": "右侧大脑前动脉 A3 段", "TYPE": "Artery (Anterior)"},
        14: {"LABEL": "L-A3", "FULL_NAME_EN": "Left Anterior Cerebral Artery, segment A3", "FULL_NAME_ZH": "左侧大脑前动脉 A3 段", "TYPE": "Artery (Anterior)"},
        15: {"LABEL": "3rd-A2", "FULL_NAME_EN": "3rd Anterior Cerebral Artery, segment A2 (Variant)", "FULL_NAME_ZH": "第三大脑前动脉 A2 段 (变异)", "TYPE": "Artery (Variant)"},
        16: {"LABEL": "3rd-A3", "FULL_NAME_EN": "3rd Anterior Cerebral Artery, segment A3 (Variant)", "FULL_NAME_ZH": "第三大脑前动脉 A3 段 (变异)", "TYPE": "Artery (Variant)"},
        17: {"LABEL": "R-M2", "FULL_NAME_EN": "Right Middle Cerebral Artery, segment M2", "FULL_NAME_ZH": "右侧大脑中动脉 M2 段", "TYPE": "Artery (Anterior)"},
        18: {"LABEL": "R-M3", "FULL_NAME_EN": "Right Middle Cerebral Artery, segment M3", "FULL_NAME_ZH": "右侧大脑中动脉 M3 段", "TYPE": "Artery (Anterior)"},
        19: {"LABEL": "L-M2", "FULL_NAME_EN": "Left Middle Cerebral Artery, segment M2", "FULL_NAME_ZH": "左侧大脑中动脉 M2 段", "TYPE": "Artery (Anterior)"},
        20: {"LABEL": "L-M3", "FULL_NAME_EN": "Left Middle Cerebral Artery, segment M3", "FULL_NAME_ZH": "左侧大脑中动脉 M3 段", "TYPE": "Artery (Anterior)"},
        21: {"LABEL": "R-P3P4", "FULL_NAME_EN": "Right Posterior Cerebral Artery, segments P3 & P4", "FULL_NAME_ZH": "右侧大脑后动脉 P3 和 P4 段", "TYPE": "Artery (Posterior)"},
        22: {"LABEL": "L-P3P4", "FULL_NAME_EN": "Left Posterior Cerebral Artery, segments P3 & P4", "FULL_NAME_ZH": "左侧大脑后动脉 P3 和 P4 段", "TYPE": "Artery (Posterior)"},
        23: {"LABEL": "R-VA", "FULL_NAME_EN": "Right Vertebral Artery", "FULL_NAME_ZH": "右侧椎动脉", "TYPE": "Artery (Posterior)"},
        24: {"LABEL": "L-VA", "FULL_NAME_EN": "Left Vertebral Artery", "FULL_NAME_ZH": "左侧椎动脉", "TYPE": "Artery (Posterior)"},
        25: {"LABEL": "R-SCA", "FULL_NAME_EN": "Right Superior Cerebellar Artery", "FULL_NAME_ZH": "右侧小脑上动脉", "TYPE": "Artery (Cerebellar)"},
        26: {"LABEL": "L-SCA", "FULL_NAME_EN": "Left Superior Cerebellar Artery", "FULL_NAME_ZH": "左侧小脑上动脉", "TYPE": "Artery (Cerebellar)"},
        27: {"LABEL": "R-AICA", "FULL_NAME_EN": "Right Anterior Inferior Cerebellar Artery", "FULL_NAME_ZH": "右侧小脑前下动脉", "TYPE": "Artery (Cerebellar)"},
        28: {"LABEL": "L-AICA", "FULL_NAME_EN": "Left Anterior Inferior Cerebellar Artery", "FULL_NAME_ZH": "左侧小脑前下动脉", "TYPE": "Artery (Cerebellar)"},
        29: {"LABEL": "R-PICA", "FULL_NAME_EN": "Right Posterior Inferior Cerebellar Artery", "FULL_NAME_ZH": "右侧小脑后下动脉", "TYPE": "Artery (Cerebellar)"},
        30: {"LABEL": "L-PICA", "FULL_NAME_EN": "Left Posterior Inferior Cerebellar Artery", "FULL_NAME_ZH": "左侧小脑后下动脉", "TYPE": "Artery (Cerebellar)"},
        31: {"LABEL": "R-AChA", "FULL_NAME_EN": "Right Anterior Choroidal Artery", "FULL_NAME_ZH": "右侧脉络膜前动脉", "TYPE": "Artery (Deep)"},
        32: {"LABEL": "L-AChA", "FULL_NAME_EN": "Left Anterior Choroidal Artery", "FULL_NAME_ZH": "左侧脉络膜前动脉", "TYPE": "Artery (Deep)"},
        33: {"LABEL": "R-OA", "FULL_NAME_EN": "Right Ophthalmic Artery", "FULL_NAME_ZH": "右侧眼动脉", "TYPE": "Artery (Intracranial)"},
        34: {"LABEL": "L-OA", "FULL_NAME_EN": "Left Ophthalmic Artery", "FULL_NAME_ZH": "左侧眼动脉", "TYPE": "Artery (Intracranial)"},
        
        # 颅外血管及深静脉（新增或调整）
        35: {"LABEL": "R-ECA", "FULL_NAME_EN": "Right External Carotid Artery", "FULL_NAME_ZH": "右侧颈外动脉", "TYPE": "Artery (Extracranial)"},
        36: {"LABEL": "L-ECA", "FULL_NAME_EN": "Left External Carotid Artery", "FULL_NAME_ZH": "左侧颈外动脉", "TYPE": "Artery (Extracranial)"},
        37: {"LABEL": "R-STA", "FULL_NAME_EN": "Right Superficial Temporal Artery", "FULL_NAME_ZH": "右侧颞浅动脉", "TYPE": "Artery (Extracranial)"},
        38: {"LABEL": "L-STA", "FULL_NAME_EN": "Left Superficial Temporal Artery", "FULL_NAME_ZH": "左侧颞浅动脉", "TYPE": "Artery (Extracranial)"},
        39: {"LABEL": "R-MaxA", "FULL_NAME_EN": "Right Maxillary Artery", "FULL_NAME_ZH": "右侧上颌动脉", "TYPE": "Artery (Extracranial)"},
        40: {"LABEL": "L-MaxA", "FULL_NAME_EN": "Left Maxillary Artery", "FULL_NAME_ZH": "左侧上颌动脉", "TYPE": "Artery (Extracranial)"},
        41: {"LABEL": "R-MMA", "FULL_NAME_EN": "Right Middle Meningeal Artery", "FULL_NAME_ZH": "右侧脑膜中动脉", "TYPE": "Artery (Extracranial)"},
        42: {"LABEL": "L-MMA", "FULL_NAME_EN": "Left Middle Meningeal Artery", "FULL_NAME_ZH": "左侧脑膜中动脉", "TYPE": "Artery (Extracranial)"},
        # 注意：如果您的原始数据中包含 VoG (35), StS (36), SSS (40) 等静脉标签，
        # 但新的列表用动脉覆盖了这些 IDX，那么请以新的列表为准。
        # 这里我们假设您已经更新了标签文件，用新的动脉标签替换了旧的静脉标签。
    }

    # already_has_proceed =  [
    #     "mr_011.tif",
    #     "mr_018.tif",
    #     "mr_006.tif",
    #     "mr_020.tif",
    #     "mr_005.tif",
    #     "mr_016.tif",
    #     "mr_001.tif",
    #     "mr_012.tif",
    #     "mr_026.tif",
    #     "mr_021.tif",
    #     "mr_022.tif",
    #     "mr_015.tif",
    #     "mr_008.tif",
    #     "mr_013.tif",
    #     "mr_004.tif",
    #     "mr_017.tif",
    #     "mr_014.tif",
    #     "mr_025.tif",
    #     "mr_002.tif"
    # ]
    already_has_proceed = None

    ## 得到多个小ROI的数据
    para_path = '/data1/liuyang/stent/interior/data/voxel_and_dim_info.json' #存储了体素大小基本信息的参数文件
    # radius = 64
    # number_roi_in_one_sample = 10
    # 训练数据
    mask_dir = '/data1/liuyang/stent/interior/data/root_data/tif//train_data/label'
    save_dir = f'/data1/liuyang/stent/interior/data/src_data//radius-{radius}/train_data'
    get_sample_proj_vol(mask_dir=mask_dir, save_dir=save_dir, para_path=para_path, number_roi_in_one_sample=number_roi_in_one_sample, radius=radius, already_has = already_has_proceed)

    mask_dir = '/data1/liuyang/stent/interior/data/root_data/tif//test_data/label'
    save_dir = f'/data1/liuyang/stent/interior/data/src_data//radius-{radius}/test_data'
    get_sample_proj_vol(mask_dir=mask_dir, save_dir=save_dir, para_path=para_path, number_roi_in_one_sample=number_roi_in_one_sample, radius=radius, already_has = already_has_proceed)


