import numpy as np
import tigre
from scipy.ndimage import gaussian_filter
import nibabel as nib

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
    
    ## 确保索引不超出范围
    valid_mask = (Z_indices >= 0) & (Z_indices < D) & \
                 (X_indices >= 0) & (X_indices < H) & \
                 (Y_indices >= 0) & (Y_indices < W)

    valid_Z = Z_indices[valid_mask]
    valid_X = X_indices[valid_mask]
    valid_Y = Y_indices[valid_mask]
    
    valid_indices = (valid_Z, valid_X, valid_Y)
    voxel = np.zeros(voxel_size, dtype=np.float32) 
    voxel[valid_indices] = fill_value
    
    return voxel



### 生成血管相关的代码
# def set_geo(DSD  =1250, 
#             DSO = 750,
#             nDetector = np.array([512,512]),
#             dDetector = np.array([0.3, 0.3]), 
#             nVoxel = np.array([248, 510, 636]),
#             dVoxel = np.array([0.3, 0.3, 0.6]),
#             geo_type='cone'):
def set_geo(DSD  =1200, 
            DSO = 780,
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


def hu_to_mu(hu, mu_water=0.0227): #水的衰减系数是0.019 mm-1，0.19 cm-1
    return mu_water * (1.0 + hu / 1000.0)
    
def add_scatter_noise(
    combined_proj,
    rng= np.random.default_rng(2026),
    I0_range=(5e4, 2e5),          # 入射光子数/强度的范围，用于泊松采样
    poisson_strength_range=(0.6, 1.0),  # 泊松噪声强度缩放（<1 表示更弱）
    corr_sigma_range=(0.6, 2.5),  # “相关噪声”相关尺度：对噪声做高斯平滑的 sigma
    scatter_frac_range=(0.01, 0.08),    # 散射占主信号强度的比例
    scatter_sigma_range=(8.0, 30.0),    # 散射的低频尺度：更大=更平滑
    eps=1e-8
):
    # -----------------------------
    # (4) 加 correlated Poisson-noise & scatter
    # 常见物理模型：对数域投影 p -> 强度域 I = I0 * exp(-p)
    # scatter 作为强度域的加性项（低频平滑）
    # Poisson 在强度域采样后再 log 回投影域
    # correlated：对“噪声场”做平滑，引入相关性
    # -----------------------------
    I0 = rng.uniform(*I0_range)  # 入射强度
    I = I0 * np.exp(-combined_proj)

    # scatter：用一个更大 sigma 的低频模糊来模拟“缓慢变化”的散射分布
    scatter_frac = rng.uniform(*scatter_frac_range)
    scatter_sigma = rng.uniform(*scatter_sigma_range)
    # 这里用 I 的低频分量做散射基底（也可以用常量/多项式背景等）
    scatter_base = gaussian_filter(I, sigma=scatter_sigma)
    scatter = scatter_frac * scatter_base

    I_scatter = I + scatter
    I_scatter = np.maximum(I_scatter, eps)

    # Poisson 强度缩放：相当于改变剂量/噪声水平（越小越噪）
    poisson_strength = rng.uniform(*poisson_strength_range)
    I_eff = I_scatter * poisson_strength
    I_eff = np.maximum(I_eff, eps)

    # 先做“独立泊松采样”
    I_poiss = rng.poisson(I_eff).astype(np.float32)
    I_poiss = np.maximum(I_poiss, eps)

    # 再把噪声变成“相关噪声”：对噪声残差进行平滑再加回去
    corr_sigma = rng.uniform(*corr_sigma_range)
    noise_residual = I_poiss - I_eff
    noise_residual_corr = gaussian_filter(noise_residual, sigma=corr_sigma)
    I_corr = I_eff + noise_residual_corr
    I_corr = np.maximum(I_corr, eps)

    # 回到对数域投影
    combined_proj_noisy = -np.log(I_corr / (I0 * poisson_strength + eps) + eps)

    return combined_proj_noisy



def add_scatter_noise_fast(
    combined_proj,
    rng=np.random.default_rng(2026),
    I0_range=(5e4, 2e5),
    poisson_strength_range=(0.6, 1.0),
    corr_sigma_range=(0.6, 2.5),
    scatter_frac_range=(0.01, 0.08),
    scatter_sigma_range=(8.0, 30.0),
    eps=1e-8,
):
    # 建议：统一用 float32，减少内存带宽压力（尤其 combined_proj 本来是 float32 的话）
    proj = np.asarray(combined_proj, dtype=np.float32)

    I0 = rng.uniform(*I0_range)
    # I = I0 * exp(-proj)
    I = np.exp(-proj, dtype=np.float32)
    I *= np.float32(I0)

    scatter_frac = np.float32(rng.uniform(*scatter_frac_range))
    scatter_sigma = float(rng.uniform(*scatter_sigma_range))

    # scatter_base = gaussian_filter(I, sigma=scatter_sigma)
    scatter_base = np.empty_like(I)
    gaussian_filter(I, sigma=scatter_sigma, output=scatter_base)

    # I_scatter = I + scatter_frac * scatter_base
    I_scatter = I  # 复用 I 的内存（如果你后面不再需要原 I）
    I_scatter += scatter_frac * scatter_base
    np.clip(I_scatter, eps, None, out=I_scatter)

    poisson_strength = np.float32(rng.uniform(*poisson_strength_range))
    I_eff = I_scatter
    I_eff *= poisson_strength
    np.clip(I_eff, eps, None, out=I_eff)

    # poisson
    I_poiss = rng.poisson(I_eff).astype(np.float32, copy=False)
    np.clip(I_poiss, eps, None, out=I_poiss)

    # noise_residual = I_poiss - I_eff  （I_eff 当前就是 I_eff）
    noise_residual = I_poiss
    noise_residual -= I_eff

    corr_sigma = float(rng.uniform(*corr_sigma_range))
    noise_residual_corr = np.empty_like(noise_residual)
    gaussian_filter(noise_residual, sigma=corr_sigma, output=noise_residual_corr)

    I_corr = I_eff
    I_corr += noise_residual_corr
    np.clip(I_corr, eps, None, out=I_corr)

    denom = np.float32(I0) * poisson_strength + np.float32(eps)

    # combined_proj_noisy = -log(I_corr / denom + eps)
    # 为了少一次临时数组：先做 I_corr /= denom，然后加 eps，再 log
    I_corr /= denom
    I_corr += np.float32(eps)
    combined_proj_noisy = -np.log(I_corr, dtype=np.float32)

    return combined_proj_noisy


## 最开始的版本
# def coarse_mri_to_ct(mra_nib, thred = 20):
#     def hu_to_mu(hu, mu_water=0.0277): #水的衰减系数是0.019 mm-1，0.19 cm-1
#         return mu_water * (1.0 + hu / 1000.0)
    
#     mra_np = mra_nib.get_fdata().astype(np.float32)
#     valid_mask = np.zeros_like(mra_np, dtype=np.int8)
#     valid_mask[mra_np>thred] = 1

#     # 归一化
#     mra_np_nor = (mra_np - mra_np.min()) / (mra_np.max() - mra_np.min())

#     # 强度反转（MRA 血管亮 → CT 血管暗）
#     pseudo_ct = 1.0 - mra_np_nor

#     # 非线性映射（模拟 CT 对比）
#     pseudo_ct = np.power(pseudo_ct, 0.5)

#     pseudo_ct = pseudo_ct * 2000 - 1000
#     pseudo_ct = hu_to_mu(pseudo_ct)

#     pseudo_ct[valid_mask==0] = 0
#     # ct_img = nib.Nifti1Image(pseudo_ct, mra_nib.affine, mra_nib.header)
#     return pseudo_ct.astype(np.float32)
#     # return ct_img

def add_skull_box_simple(
    vol: np.ndarray,
    t: int = 20,
    skull_mu_min = 0.057,
    skull_mu_max = 0.07
) -> np.ndarray:
    """
    在 3D 图像内部，距离边界 < t 的长方体壳区域，直接赋值 skull_mu（单位：mu）
    """
    out = np.copy(vol)
    Z, Y, X = out.shape
    skull_mu = np.random.uniform(skull_mu_min, skull_mu_max)
    out[:t, :, :] += skull_mu #相加
    out[Z - t:, :, :] += skull_mu
    out[:, :t, :] += skull_mu
    out[:, Y - t:, :] += skull_mu
    # out[:, :, :t] = skull_mu
    # out[:, :, X - t:] = skull_mu

    return out

def hu_to_mu(hu, mu_water=0.0277): #水的衰减系数是0.019 mm-1，0.19 cm-1
    return mu_water * (1.0 + hu / 1000.0)

def coarse_mri_to_ct(mra_data, target_min=0, target_max=0.05, scale =  1.5):
    def nor(data, target_min, target_max):
        # d_min, d_max = np.min(data), np.max(data)
        d_min, d_max = np.min(data), np.percentile(mra_data, 90)
        res = (data - d_min) / (d_max - d_min)
        # res = np.clip(res, 0, 1)
        res = res * (target_max - target_min) + target_min
        return res
    mra_data = mra_data.astype(np.float32)
    mra_data = nor(mra_data, target_min, target_max)
    mra_data /= scale
    return mra_data

