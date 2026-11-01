import os
import tifffile as tiff
import argparse
import pyvista as pv
pv.OFF_SCREEN = True
import numpy as np
import nibabel as nib
join = os.path.join
from tqdm import tqdm


def visualize_stent_vessel_and_screenshot(
    vessel_voxel,
    stent_points,
    spacing=(1, 1, 1),
    out_dir="screenshots",
    window_size=(1000, 800),
):
    """
    生成三个预定义角度（XY / XZ / YZ）的 2D 截图
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # 1. 血管体素 → 等值面
    # -------------------------------
    grid = pv.ImageData()

    dims = np.array(vessel_voxel.shape) + 1
    grid.dimensions = dims
    grid.spacing = spacing

    padded = np.pad(vessel_voxel, ((0, 1), (0, 1), (0, 1)), mode="edge")
    grid.point_data["values"] = padded.flatten(order="F")

    vessel_mesh = grid.contour([0.5])

    # -------------------------------
    # 2. 支架点云
    # -------------------------------
    if stent_points.size > 0:
        stent_poly = pv.PolyData(stent_points)
        stent_glyphs = stent_poly.glyph(
            scale=False,
            orient=False,
            geom=pv.Sphere(radius=0.3)
        )
    else:
        stent_glyphs = None

    # -------------------------------
    # 3. Plotter（off-screen）
    # -------------------------------
    pl = pv.Plotter(
        off_screen=True,
        window_size=window_size
    )
    pl.set_background("#303030")

    pl.add_mesh(
        vessel_mesh,
        color="mistyrose",
        opacity=0.6,
        smooth_shading=True,
        diffuse=0.8,
        specular=0.3,
    )

    if stent_glyphs is not None:
        pl.add_mesh(
            stent_glyphs,
            color="cyan",
            ambient=0.3,
            diffuse=0.8,
            specular=0.9,
            specular_power=15,
        )

    pl.add_axes()

    # -------------------------------
    # 4. 三个预定义角度截图
    # -------------------------------
    views = {
        "view_xy": pl.view_xy,  # axial
        "view_xz": pl.view_xz,  # coronal
        "view_yz": pl.view_yz,  # sagittal
    }

    for name, view_func in views.items():
        view_func()
        pl.camera.zoom(1.2)  # 可选：轻微放大
        out_path = os.path.join(out_dir, f"{name}.png")
        pl.screenshot(out_path)
        # print(f"Saved: {out_path}")

    pl.close()


def read_data(image_path):
    if image_path.endswith(".nii.gz"):
        return nib.load(image_path).get_fdata().astype(np.float32)
    elif image_path.endswith(".tif"):
        return np.transpose(
            tiff.imread(image_path).astype(np.float32),
            [2, 0, 1]
        )
    else:
        raise ValueError("不支持的文件类型")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vessel_dir", type=str, required=True)
    parser.add_argument("--stent_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="screenshots")
    args = parser.parse_args()
    
    vessel_dir = args.vessel_dir
    stent_dir = args.stent_dir
    out_dir = args.out_dir

    all_list = sorted(os.listdir(vessel_dir))

    for i in tqdm(all_list):
        vessel_path = os.path.join(vessel_dir, i)
        stent_path = os.path.join(stent_dir, i)
        output_dir = join(out_dir, i.split('.nii.gz')[0] if i.endswith('.nii.gz') else i.split('.tif')[0])
        vessel = read_data(vessel_path)
        stent = read_data(stent_path)

        stent_points = np.vstack(np.where(stent > 0)).T
        stent_points = stent_points.astype(np.float32)

        visualize_stent_vessel_and_screenshot(
            vessel_voxel=vessel,
            stent_points=stent_points,
            out_dir=output_dir
        )
