import numpy as np
import os
join = os.path.join
import nibabel as nib
import os
from collections import defaultdict
import nibabel as nib
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter


def build_graph(points_mm, k=3, max_connect_dist=3.0):
    """构建 skeleton 图"""
    N = len(points_mm)
    nbrs = NearestNeighbors(n_neighbors=min(k, N)).fit(points_mm)
    dist, idx = nbrs.kneighbors(points_mm)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(1, idx.shape[1]):
            if dist[i, j] <= max_connect_dist:
                G.add_edge(i, idx[i, j], weight=float(dist[i, j]))

    return G, nbrs


def extract_random_segment(points_voxel, spacing, length_mm=20.0,
                            k=3, max_connect_dist=3.0, max_trials=100):
    """
    随机选一个点，从该点出发找到某条可行路径，
    若能走 >= length_mm，则截取前 length_mm 的子路径。
    
    点云可以有分叉。
    """
    points = np.asarray(points_voxel)
    points_mm = points * np.array(spacing).reshape(1,3)

    G, nbrs = build_graph(points_mm, k, max_connect_dist)

    N = len(points_mm)

    for _ in range(max_trials):
        # 1. 随机选一点
        start = np.random.randint(0, N)

        # 2. 从该点做 Dijkstra，找到所有可达点及距离
        lengths, paths = nx.single_source_dijkstra(G, start)

        # 3. 找到从 start 能走到的最远点
        farthest_node = max(lengths, key=lengths.get)
        max_dist = lengths[farthest_node]

        # 若最大可行路径 < 20mm，则换个点继续试
        if max_dist < length_mm:
            continue

        # 4. 得到 start → farthest_node 的路径
        node_path = paths[farthest_node]
        path_pts = points_mm[np.array(node_path)]

        # 5. 累积距离，截取 20mm 子段
        diffs = np.diff(path_pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumdist = np.concatenate([[0], np.cumsum(dists)])

        idx = np.where(cumdist >= length_mm)[0][0]
        return path_pts[:idx+1]
    
    return None #找不到一个满足要求的线段



## 还需要对点进行排序，防止点的顺序是错误的，导致插值时沿着错误的曲线去插值。

def sort_points_nearest_neighbor(points):
    """
    核心算法：路径重构（连点成线）。
    使用贪心算法（Nearest Neighbor）寻找合理的连接顺序。
    """
    points = np.array(points)
    N = len(points)
    if N < 2: return points

    # --- 第一步：找到一个端点作为起点 ---
    # 简单的启发式方法：距离几何中心最远的点通常是端点
    center = np.mean(points, axis=0)
    dists_to_center = np.linalg.norm(points - center, axis=1)
    start_idx = np.argmax(dists_to_center)

    # --- 第二步：贪心搜索路径 ---
    path_indices = [start_idx]
    visited = set([start_idx])
    current_idx = start_idx

    # 预计算所有点之间的距离矩阵 (如果点数 > 10000，这里需要优化)
    # cdist 比循环快得多
    all_dists = cdist(points, points)

    for _ in range(N - 1):
        # 获取当前点到其他所有点的距离
        dists = all_dists[current_idx].copy()
        
        # 将已访问点的距离设为无穷大
        dists[list(visited)] = np.inf
        
        # 找到最近的邻居
        next_idx = np.argmin(dists)
        
        path_indices.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx

    return points[path_indices]

def sort_centerline_robust(points): # 试过了，是效果最好的一个版本
    """
    对无序的中心线点云进行排序。
    """
    points = np.array(points)
    if len(points) < 3:
        return points
        
    N = len(points)
    
    # --- 1. 寻找合理的起点 ---
    # 方法：找距离几何中心最远的点，通常是端点之一
    centroid = np.mean(points, axis=0)
    dists_to_center = np.linalg.norm(points - centroid, axis=1)
    start_idx = np.argmax(dists_to_center)
    
    # --- 2. 构建近邻搜索 (关键：k要够大) ---
    # k 取 10 或 20，防止因噪声导致的局部短路或回断
    k = min(N, 100) 
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    # --- 3. 贪心遍历 ---
    visited = np.zeros(N, dtype=bool)
    ordered_indices = [start_idx]
    visited[start_idx] = True
    
    current_idx = start_idx
    
    for _ in range(N - 1):
        # 获取当前点的所有邻居索引
        candidates = indices[current_idx]
        
        found_next = False
        for neighbor_idx in candidates:
            # 跳过自己和已访问的点
            if not visited[neighbor_idx]:
                # 找到下一个点
                current_idx = neighbor_idx
                ordered_indices.append(current_idx)
                visited[current_idx] = True
                found_next = True
                break

    return points[ordered_indices]

def interpolate_linear_precise(points, target_count):
    """
    基于累积弧长进行插值，保持形状精准。
    """
    # 1. 计算每段线段的长度
    diffs = points[1:] - points[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    
    # 2. 计算累积距离 (从0开始)
    cumulative_dist = np.zeros(len(points))
    cumulative_dist[1:] = np.cumsum(dists)
    
    # 3. 归一化总长度 [0, 1]
    total_len = cumulative_dist[-1]
    if total_len == 0: return np.resize(points, (target_count, 3)) # 防止所有点重合
    normalized_dist = cumulative_dist / total_len
    
    # 4. 创建插值函数
    interpolator = interp1d(normalized_dist, points, kind='cubic', axis=0)
    # interpolator = interp1d(normalized_dist, points, kind='spline', axis=0)

    # 5. 生成新点
    new_dists = np.linspace(0, 1, target_count)
    return interpolator(new_dists)

def process_point_cloud(raw_points, target_count):
    """
    主函数：整理乱序点云 -> 插值
    """
    # # 1. 先排序（解决形状乱的问题）
    # sorted_points = sort_points_nearest_neighbor(raw_points)
    
    # # 2. 再插值（解决点数问题）
    # final_points = interpolate_linear_precise(sorted_points, target_count)


    final_points = interpolate_linear_precise(raw_points, target_count)
    
    return final_points


def smooth_curve(points, window=400, poly=3):
    """
    对曲线进行 Savitzky-Golay 平滑。
    window 必须为奇数
    """
    points = points.astype(float)
    if len(points) < window:
        window = len(points) // 2 * 2 + 1  # 自动选择最近可用窗口长度

    smooth = points.copy()
    for i in range(3):  # z, y, x
        smooth[:, i] = savgol_filter(points[:, i], window_length=window, polyorder=poly)

    return smooth

def compute_curve_length(point_cloud):
    """
    计算点云曲线的长度
    """
    diffs = np.diff(point_cloud, axis=0)  # 相邻点差
    seg_lengths = np.linalg.norm(diffs, axis=1)  # 每段线段长度
    return np.sum(seg_lengths)



## 首先要选择出来血管
## 移除中心线点数较少的部分。具体参数可以自行控制
def remove_small_components(points, k=2, max_connect_dist=3.0, min_component_size=3):
    """
    points: (N,3) 骨架点云
    k: 每个点连接多少个最近邻
    max_connect_dist: 最近邻距离超过该值就认为断裂
    min_component_size: 小于该点数的组件将被剔除
    """
    
    if len(points) == 0:
        return points
    
    # --- 1. 建立 kNN 图 ---
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    G = nx.Graph()

    # --- 2. 添加边（距离太大不连接） ---
    for i in range(points.shape[0]):
        for j in range(1, k):
            d = distances[i, j]
            if d <= max_connect_dist:
                G.add_edge(i, indices[i, j])

    # --- 3. 获取所有连通域 ---
    components = list(nx.connected_components(G))

    final_components = [] #将不同练在一起的线分开存放

    # --- 4. 过滤掉小组件 ---
    filtered_indices = []
    for comp in components:
        if len(comp) >= min_component_size:
            filtered_indices.extend(list(comp))
            final_components.append(points[np.array(list(comp))])

    # 返回过滤后的点云
    filtered_points = points[np.array(filtered_indices)] #将所有点放在一起返回
    return filtered_points, final_components



def mean_of_8_neighbors_batch(distance_map: np.ndarray, points: np.ndarray, clamp: bool = True) -> np.ndarray:
    """
    批量版：pts_zyx shape = (N,3)，每个点为 (z,y,x)
    返回 shape = (N,)
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) with order (z, y, x)")

    X, Y, Z = distance_map.shape

    ## 分别向下取整和加一得到环绕这个点的正方体的坐标
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    z0 = np.floor(z).astype(np.int64); z1 = z0 + 1
    y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
    x0 = np.floor(x).astype(np.int64); x1 = x0 + 1

    # 8角点分别取值，然后平均
    def safe_idx(arr, maxv):
        if clamp:
            return np.clip(arr, 0, maxv - 1)
        else:
            if np.any(arr < 0) or np.any(arr >= maxv):
                raise IndexError("Neighbor index out of bounds")
            return arr

    z0s = safe_idx(z0, Z); z1s = safe_idx(z1, Z)
    y0s = safe_idx(y0, Y); y1s = safe_idx(y1, Y)
    x0s = safe_idx(x0, X); x1s = safe_idx(x1, X)

    v000 = distance_map[x0s, y0s, z0s]
    v100 = distance_map[x1s, y0s, z0s]
    v010 = distance_map[x0s, y1s, z0s]
    v110 = distance_map[x1s, y1s, z0s]
    v001 = distance_map[x0s, y0s, z1s]
    v101 = distance_map[x1s, y0s, z1s]
    v011 = distance_map[x0s, y1s, z1s]
    v111 = distance_map[x1s, y1s, z1s]


    # print(v000, v001, v010, v011, v100, v101, v110, v111)

    vals = (v000 + v001 + v010 + v011 + v100 + v101 + v110 + v111) / 8.0
    return vals.astype(np.float64)





