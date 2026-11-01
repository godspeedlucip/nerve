import numpy as np

def scale_z_axis_only(points, z_target_min=0, z_target_max=107):
    """
    只对点云的 Z 轴进行缩放，X 和 Y 轴保持原样。
    
    参数:
    - points: np.ndarray, (N, 3) 形状的点云
    - z_target_min: 目标 Z 轴最小值
    - z_target_max: 目标 Z 轴最大值
    """
    # 拷贝一份数据，避免修改原始点云
    scaled_points = points.copy()
    
    z_raw = points[:, 2]
    z_min = z_raw.min()
    z_max = z_raw.max()
    
    z_range = z_max - z_min
    
    # 防止除以 0（如果所有点的 z 都一样）
    if z_range == 0:
        scaled_points[:, 2] = z_target_min
        return scaled_points
    
    # 执行 Z 轴缩放公式
    scaled_points[:, 2] = (z_raw - z_min) / z_range * (z_target_max - z_target_min) + z_target_min
    
    return scaled_points

## 生成覆膜支架1
class w_like_Stent:
    def __init__(self):
        pass

    def create_straight_stent_segment(
        self,
        r_i_t,
        r_i_b,
        A_i,
        p_i,
        s_i,
        height,
        num_points=2000
    ):
        """
        创建一个完整的支架基本单元，z 方向缩放到 height
        """
        theta = np.linspace(0, 2 * np.pi, num_points)

        r = (1 - np.sin(p_i * theta + 1) / 2) * r_i_t + \
            (np.sin(p_i * theta + 1) / 2) * r_i_b

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        z_raw = A_i * np.sin(p_i * theta + s_i)

        z = (z_raw - z_raw.min()) / (z_raw.max() - z_raw.min()) * height

        return np.vstack([x, y, z]).T


    def create_straight_stent_along_z(
        self,
        nz_voxels,
        r_avg=10,
        A_i=15,
        p_i=7,
        s_i=np.pi / 2,
        preferred_segment_height=10,
        gap_voxels = None,
        num_points=2000
    ):
        """
        约束：
        1️⃣ 单元完整
        2️⃣ 单元之间有固定间距 gap_voxels
        3️⃣ 总长度严格等于 target_length_voxels
        """

        r_variation = r_avg * 0 #控制半径的变化
        r_i_t = r_avg + r_variation
        r_i_b = r_avg - r_variation

        # ① 根据期望高度，确定单元数量
        if(gap_voxels is None):
            gap_voxels = int(preferred_segment_height/2)

        num_segments = max(
            1,
            int(round(
                (nz_voxels + gap_voxels) /
                (preferred_segment_height + gap_voxels)
            ))
        )

        # ② 反推真实单元高度（核心公式）
        total_gap = (num_segments - 1) * gap_voxels
        segment_height = (nz_voxels - total_gap) / num_segments

        if segment_height <= 0:
            raise ValueError("间距过大，导致单元高度为非正值")

        # ③ 创建完整单元
        segment_model = self.create_straight_stent_segment(
            r_i_t, r_i_b,
            A_i, p_i, s_i,
            segment_height,
            num_points
        )

        # ④ 沿 z 轴堆叠（单元 + 间距）
        all_points = []
        current_z = 0.0

        for _ in range(num_segments):
            seg = segment_model.copy()
            seg[:, 2] += current_z
            all_points.append(seg)

            current_z += segment_height + gap_voxels

        return np.vstack(all_points)



## 创建覆膜支架2

class ring_Stent:
    def __init__(self):
        pass

    def ring_unit_points(
        self,
        R: float,
        A: float,
        k: int,
        num_points: int,
        phase: float = 0.0,
        center_xy=(0.0, 0.0),
    ):
        """
        返回：
        points: (N,3) 点云
        unit_height: 2A
        """
        if num_points < 32:
            raise ValueError("num_points too small")
        if A <= 0 or R <= 0:
            raise ValueError("R and A must be > 0")

        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        cx, cy = center_xy

        x = cx + R * np.cos(t + phase)
        y = cy + R * np.sin(t + phase)
        z = A * np.sin(k * t)  # [-A, A]

        pts = np.stack([x, y, z], axis=1)
        unit_height = 2.0 * A
        return pts, unit_height

    def choose_units_and_gap(
        self,
        total_length: float,
        unit_height: float,
        gap_expected: float,
    ):
        """
        保证：
        - unit 完整
        - 总长度严格 == total_length
        - gap 在必要时自动微调
        """
        if total_length <= 0:
            raise ValueError("total_length must be > 0")
        if unit_height <= 0:
            raise ValueError("unit_height must be > 0")
        if gap_expected < 0:
            raise ValueError("gap_expected must be >= 0")

        n = int(np.floor((total_length + gap_expected) /
                        (unit_height + gap_expected)))
        n = max(n, 1)

        if n == 1:
            if unit_height > total_length:
                raise ValueError(
                    "Single unit taller than total_length. "
                    "Reduce A or increase total_length."
                )
            return 1, 0.0

        gap = (total_length - n * unit_height) / (n - 1)

        while gap < 0 and n > 1:
            n -= 1
            if n == 1:
                return 1, 0.0
            gap = (total_length - n * unit_height) / (n - 1)

        return n, gap

    def thicken_points_voxel_style(
        self,
        points: np.ndarray,
        radius_voxels: int,
    ):
        """
        在 x-y-z 三个方向用“体素球核”加粗点云
        返回：(M,3) 点云
        """
        if radius_voxels <= 0:
            return points.copy()

        r = int(radius_voxels)
        grid = np.arange(-r, r + 1)
        dx, dy, dz = np.meshgrid(grid, grid, grid, indexing="ij")
        mask = (dx*dx + dy*dy + dz*dz) <= r*r
        offsets = np.stack([dx[mask], dy[mask], dz[mask]], axis=1)

        base = np.rint(points).astype(np.int32)
        thick = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

        # 去重
        thick = np.unique(thick, axis=0)
        return thick.astype(np.float32)


    def create_straight_stent_along_z(
        self,
        nz_voxels: int,
        gap_voxels = 5,
        R=18,
        A = 6,
        k = 4,
        num_points_per_unit: int = 800,
        center_xy=(0.0, 0.0),
        seed = 42,
    ):
        """
        返回：
        points: (N,3) 点云
        info: dict
        """
        rng = np.random.default_rng(seed)

        # 基本单元
        unit_pts, unit_h = self.ring_unit_points(
            R, A, k,
            num_points=num_points_per_unit,
            phase=0.0,
            center_xy=center_xy
        )

        # 选择单元数 & 实际间距
        n_units, gap_real = self.choose_units_and_gap(
            total_length=float(nz_voxels),
            unit_height=unit_h,
            gap_expected=gap_voxels
        )

        pitch = unit_h + gap_real

        all_pts = []
        for i in range(n_units):
            phase = rng.uniform(0, 2*np.pi)
            pts_i, _ = self.ring_unit_points(
                R, A, k,
                num_points=num_points_per_unit,
                phase=phase,
                center_xy=center_xy
            )
            pts_i[:, 2] += i * pitch
            all_pts.append(pts_i)

        points = np.vstack(all_pts)

        # z 起点归零（不裁单元）
        points[:, 2] -= points[:, 2].min()

        return points
    

## 生成sin_stent
class sin_Stent:
    def __init__(self):
        pass

    def create_straight_stent_along_z(
        self,
        # spacing_z: float,
        nz_voxels: int,
        R: float = 1.0,
        A: float = 0.3,
        k: int = 5,
        num_points_per_unit: int = 2000,
        z0: float = 0.0,
        center_xy=(0.0, 0.0)
    ):
        """
        生成物理坐标点云（不是体素），但保证Z方向物理长度对应固定 nz_voxels 个体素。

        返回：
        points: (N,3) 物理坐标点云
        """
        # nz_voxels = int(z_length / spacing_z)
        # print(nz_voxels)

        # if spacing_z <= 0:
        #     raise ValueError("spacing_z must be > 0")
        if nz_voxels <= 0:
            raise ValueError("nz_voxels must be > 0")
        if num_points_per_unit < 4:
            raise ValueError("num_points_per_unit too small")

        # 目标物理长度（由 voxel space 决定）
        # Lz = nz_voxels * spacing_z
        Lz = nz_voxels #还是和别的一样在体素空间中去计算

        # 1) 先生成一个基本单元（与原逻辑一致）
        theta_unit = np.linspace(0, 2 * np.pi, num_points_per_unit // 2, endpoint=False)

        cx, cy = center_xy

        x_upper = cx + R * np.cos(theta_unit)
        y_upper = cy + R * np.sin(theta_unit)
        z_upper = A * np.sin(k * theta_unit)

        x_lower = cx + R * np.cos(theta_unit)
        y_lower = cy + R * np.sin(theta_unit)
        z_lower = -A * np.sin(k * theta_unit) + 2 * A

        unit_points = np.vstack([
            np.stack([x_upper, y_upper, z_upper], axis=1),
            np.stack([x_lower, y_lower, z_lower], axis=1),
        ])

        unit_height = 4 * A
        if unit_height <= 0:
            raise ValueError("unit_height must be > 0 (check A)")

        # 2) 需要多少个 unit 才能覆盖 Lz（多生成一点再裁切）
        num_units = int(np.ceil((Lz + 1e-9) / unit_height)) + 1

        all_points = []
        for i in range(num_units):
            shifted = unit_points.copy()
            shifted[:, 2] += i * unit_height
            all_points.append(shifted)

        points = np.vstack(all_points)

        # 3) 把z起点归零 + 加上z0偏移（如果你想让它从某个物理z开始）
        points[:, 2] -= points[:, 2].min()
        points[:, 2] += z0

        # 4) 按目标物理长度裁切，确保Z方向就是固定 nz_voxels 对应的物理长度
        z_min = z0
        z_max = z0 + Lz
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        points = points[mask]

        return points


class Stent_Ply:
    def __init__(self,ply):
        self.ply = ply
        self.ply[:,0] -= np.max(self.ply[:,0])/2
        self.ply[:,1] -= np.max(self.ply[:,1])/2
        self.ply[:,2] -= np.min(self.ply[:,2])

    def create_straight_stent_along_z(
        self,
        nz_voxels: int,
    ):
        self.ply = scale_z_axis_only(self.ply, z_target_min=0, z_target_max=nz_voxels)
        return self.ply
