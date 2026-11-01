
import numpy as np
import tigre


def get_circle_roi_fast(image_shape, roi_center, roi_radius):
    cx, cy, cz = roi_center
    r = int(np.ceil(roi_radius))
    Nx, Ny, Nz = image_shape

    # 1. 确定 Bounding Box 的范围（防止越界）
    extra_bounding = int(r * 0.25)
    x_min, x_max = max(0, cx - (r+extra_bounding)), min(Nx, cx + (r+extra_bounding) + 1)
    y_min, y_max = max(0, cy - (r+extra_bounding)), min(Ny, cy + (r+extra_bounding) + 1)
    z_min, z_max = max(0, cz - (r+extra_bounding)), min(Nz, cz + (r+extra_bounding) + 1)

    # 2. 仅在局部区域生成网格
    x, y, z = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]

    # 3. 计算距离平方（利用广播机制，ogrid 比 mgrid 更省内存）
    dist_sq = (z - cz)**2 + (x - cx)**2 + (y - cy)**2
    
    # 4. 筛选并恢复全局坐标
    mask = dist_sq <= roi_radius**2
    local_coords = np.where(mask)
    
    global_x = local_coords[0] + x_min
    global_y = local_coords[1] + y_min
    global_z = local_coords[2] + z_min

    # return np.vstack([global_x, global_y, global_z]), np.array([x_min, y_min, z_min], [x_max, y_max, z_max])
    return np.vstack([global_x, global_y, global_z])

def get_roi_proj(src_proj, geo, angles, valid_mesh, image_sizes):
    """
    valid_mesh: (3,num_of_points). roi区域的点云形式
    src_proj: 原始图像的投影
    image_sizes: 原始图像的shape
    """
    ## 创建一个只有ROI区域有值的体膜
    roi_image_mask = np.zeros(image_sizes, np.float32)
    x_coords, y_coords, z_coords = valid_mesh[0], valid_mesh[1], valid_mesh[2]
    roi_image_mask[x_coords, y_coords, z_coords] = 1

    ## 对roi区域投影
    roi_mask = tigre.Ax(np.transpose(roi_image_mask, [2, 0, 1]), geo, angles)
    roi_mask[roi_mask>0.01] = 1 #将其转换为mask
    roi_mask[roi_mask<=0.01] = 0

    #逐元素相乘提取数据
    roi_proj = src_proj * roi_mask
    return roi_proj, roi_image_mask

def get_roi(mask, sample_point, src_proj, radius=64, angles = np.linspace(0,2*np.pi,360), geo = None):
    """
    mask: 图像的mask
    sample_point: 一个中心点
    para_path: 图像的参数的文件
    radius: 采样圆的半径
    """
    roi_center = np.array([sample_point[0], sample_point[1], sample_point[2]])
    point_mesh = get_circle_roi_fast(mask.shape, roi_center, radius) # (3, N)的点云形式 #真值
    roi_proj, roi_gt_mask = get_roi_proj(src_proj, geo, angles, point_mesh, image_sizes=mask.shape) #这个函数是通过点的坐标去构建三维体素，因此就算传入的mask是(x, y, z)的shape也没关系
    fill_value = np.min(mask)
    roi_gt = np.where(roi_gt_mask, mask, fill_value)
    return roi_proj, roi_gt, roi_gt_mask

