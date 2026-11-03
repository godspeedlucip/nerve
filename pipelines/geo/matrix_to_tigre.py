import numpy as np

# 定义PI常量（与CUDA中的PI对应）
PI = np.pi


# 模拟CUDA中的Source结构体
class Source:
    def __init__(self, x=0.0, y=0.0, z=0.0, theta=0.0, area_id=0):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.area_id = area_id
        self.sod = np.sqrt(x**2 + y**2 + z**2)
        self.uoffset = 0
        self.voffset = 0
        self.sdd = 0
        self.angle = 0

    def __repr__(self):
        return f"Source(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, theta={self.theta:.4f}, area_id={self.area_id}, sod={self.sod:.4f}, sdd={self.sdd:.4f},uoffset={self.uoffset:.4f}, voffset={self.voffset:.4f})"


# 行列式计算函数（对应__device__函数）
def cal_determinant(a, b, c, d, e, f, g, h, i):
    """计算3x3矩阵的行列式"""
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# 批量计算光源数组（对应__global__内核函数）
def init_source_array_device(source_array, Pmatrix, viewNum, startAngle):
    """
    批量初始化光源数组，对应CUDA内核的Python实现

    参数：
        source_array: 输出的Source对象列表（预分配，长度=viewNum）
        Pmatrix: 投影矩阵数组，shape=(viewNum, 12)，float32类型
        viewNum: 视角数量
        startAngle: 起始角度（度）
    """
    # 转换起始角度为弧度
    startAngle_rad = startAngle / 180.0 * PI

    # 遍历每个视角（替代CUDA的线程索引）
    for view_idx in range(viewNum):
        # 提取当前视角的12维投影矩阵
        P = Pmatrix[view_idx]
        P11, P12, P13, P14 = P[0], P[3], P[6], P[9]
        P21, P22, P23, P24 = P[1], P[4], P[7], P[10]
        P31, P32, P33, P34 = P[2], P[5], P[8], P[11]

        # 计算行列式（求解线性方程组 Ax = b）
        D = cal_determinant(P11, P12, P13, P21, P22, P23, P31, P32, P33)
        Dx = cal_determinant(-P14, P12, P13, -P24, P22, P23, -P34, P32, P33)
        Dy = cal_determinant(P11, -P14, P13, P21, -P24, P23, P31, -P34, P33)
        Dz = cal_determinant(P11, P12, -P14, P21, P22, -P24, P31, P32, -P34)

        # 计算光源坐标（避免除零）
        if abs(D) < 1e-8:
            raise ValueError(f"视角{view_idx}的投影矩阵行列式为0，无法求解光源坐标")

        source_x = Dx / D
        source_y = Dy / D
        source_z = Dz / D

        # 计算角度和区域ID
        theta = np.arctan2(source_y, source_x)  # 用arctan2避免象限问题（比atan更鲁棒）
        theta_rotated = theta + startAngle_rad
        area_id = np.floor((theta_rotated + PI / 4) / (PI / 2)).astype(int)
        area_id = (area_id % 4 + 4) % 4  # 确保结果在[0,3]范围内

        # 赋值到source_array
        source = source_array[view_idx]
        source.x = source_x
        source.y = source_y
        source.z = source_z
        fAngle2 = np.arctan2(P31, P33)
        if fAngle2 < 0:
            fAngle2 += 2*np.pi
        fAngle1 = np.degrees(np.arctan2(-P12, -P22))
        source.theta = fAngle2
        source.area_id = area_id
        source.sdd = -P22 * 0.150 / np.cos(np.radians(fAngle1))
        source.sod = np.sqrt(source_x**2 + source_y**2 + source_z**2)
        source.uoffset = (P14) / P34 - 1056/2
        source.voffset = (P24) / P34 - 1056/2


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 1. 配置参数
    viewNum = 601  # 4个视角
    startAngle = 0.0  # 起始角度0度

    # 2. 生成测试用投影矩阵（实际使用时替换为真实数据）
    # 这里用随机数模拟，真实场景中Pmatrix应是12维的投影矩阵
    Pmatrix = np.fromfile("PMatrix.raw", dtype=np.float32).reshape((viewNum, 12))
    print(Pmatrix[0])
    # 3. 预分配source_array（对应CUDA的设备内存分配）
    source_array = [Source() for _ in range(viewNum)]

    # 4. 调用函数计算
    init_source_array_device(source_array, Pmatrix, viewNum, startAngle)
    u = np.zeros([601], dtype=np.float32)
    v = np.zeros([601], dtype=np.float32)
    sod = np.zeros([601], dtype=np.float32)
    sdd = np.zeros([601], dtype=np.float32)
    angles = np.zeros([601], dtype=np.float32)

    # 5. 打印结果
    print("计算得到的光源数组：")
    for i, source in enumerate(source_array):
        print(f"视角{i}: {source}")
        u[i] = source.uoffset
        v[i] = source.voffset
        sod[i] = source.sod
        sdd[i] = source.sdd
        angles[i] = source.theta
        print(u[i])
        print(v[i])
        print(sod[i])
        print(sdd[i])
        print(angles[i])
    u.tofile("uoffset.raw")
    v.tofile("voffset.raw")
    sod.tofile("sod.raw")
    sdd.tofile("sdd.raw")
    angles.tofile("angle.raw")


# import tifffile
# import numpy as np

# proj = tifffile.imread("./proj_dsa_clean.tif")
# tifffile.imwrite("./proj_dsa_clean_cut.tif", proj[0:399])
