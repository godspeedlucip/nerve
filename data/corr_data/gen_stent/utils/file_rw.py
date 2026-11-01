import os
import numpy as np
import plyfile
join = os.path.join
import open3d as o3d
import SimpleITK as sitk

def read_xyz_file(filename):
    """
    读取XYZ文件并返回坐标列表。

    :param filename: XYZ文件路径
    :return: 坐标列表
    """
    def parse_xyz_data(coordinates):
        """
        解析XYZ文件中的坐标数据。

        :param coordinates: 坐标列表
        """
        x_coords = []
        y_coords = []
        z_coords = []
        for i, (x, y, z) in enumerate(coordinates):
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
        points = np.vstack([np.array(x_coords),np.array(y_coords),np.array(z_coords)]).T
        return points
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # 忽略空行
                parts = line.split()
                if len(parts) == 3:
                    x, y, z = map(float, parts)
                    coordinates.append((x, y, z))
    return parse_xyz_data(coordinates)


def convert_point_cloud_to_ply(point_cloud,file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(file_path, pcd)

def read_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if not pcd.has_points():
        print(f"警告: 文件 {ply_path} 中没有读取到任何点。")
        return pcd, np.array([])
        
    # 将 Open3D 点云对象转换为 NumPy 数组
    point_cloud_np = np.asarray(pcd.points)
    return  point_cloud_np


def read_ply_with_open3d(file_path):
    """
    Args:
        file_path (str): PLY 文件的路径。

    Returns:
        np.ndarray: 一个形状为 (N, 3) 的 NumPy 数组，其中 N 是点的数量，
                    3 代表 x, y, z 坐标。
                    如果文件读取失败，则返回 None。
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points