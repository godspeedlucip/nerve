# `corr_data`：DTE 上游 3D 支架/血管生成器

## 1. 当前主入口

`gen_stent_and_vessel.py` 是当前最完整的一站式入口。它使用 helper 中的 `resample/`, `distance_map/`, `skeleton/`，从血管骨架选出 25 mm 的中心线段，再把随机支架模型变形到中心线上。每个 phantom 默认选 10 个 segment。

主要参数和实际含义：

| 参数 | 当前值/默认 | 说明 |
|---|---:|---|
| `resolution` | CLI 必填，常用 0.2 | 传入输出目录和 helper 版本；体素 spacing 来自 NIfTI header |
| `radius` | 64 | DTE ROI 半径，主要影响后续 vessel ROI 投影 |
| `target_length` | 25 mm | 每条支架中心线 segment 的物理长度 |
| `number_segments` | 10 | 每个 phantom 尝试生成的支架数量 |
| `segment_target_points` | 2000 | 中心线重采样点数 |
| `smooth_window` | 800 | 中心线平滑窗口 |
| `min_radius/max_radius` | 5/11 | 在 distance map 中挑选的血管骨架半径范围，当前代码按体素值使用 |
| train generator | 8 种 | `stent1..4`、wall、sin、wLike、ring |
| test generator | 2 种 | `stent5`、`w_stent` |
| projection | 360 角度 | 上游 geometry 的 detector 为 2048×2048，pixel 来自 volume spacing，不是 DTE 的 1500/0.308 设置 |

输出默认结构（不传目录名时是 `corr_stent`/`corr_vessel`；下面的 DTE 命令显式写成 `stent`/`vessel`）：

```text
<save_path>/voxel_0.2/src_data/radius-64/
├── corr_vessel/<mode>_data/
│   ├── vol_gt/          # ROI 内的血管体积
│   ├── proj_gt/<sample>/ # 血管 360 稀疏投影
│   └── vol_gt_mask/     # ROI mask
└── corr_stent/<mode>_data/
    ├── vol_gt/          # 二值支架体积
    ├── proj_gt/<sample>/# 二值支架 360 稀疏投影
    ├── vol_gt_mask/     # 与该 sample 对齐的 ROI mask
    └── config/          # 每个 phantom 的 sample 配置 pickle
```

一个 `proj_gt/<sample>/` 通常包含 `proj_000.npz` 到 `proj_359.npz`。这些是上游直接投影结果；DTE 后续脚本又会对 3D image/stent 做一次 detector 为 1500、pixel 为 0.308 的 ROI 投影，所以两层 `proj_gt` 不应混为同一个格式。

## 2. marker 方案

当前主入口新增了可选 marker，默认关闭：

```bash
PY=/data1/liuyang/conda/envs/common/bin/python
$PY gen_stent_and_vessel.py \
  --save_path /data1/liuyang/stent/DTE/data/resample_data/costa_new_marker \
  --resolution 0.2 --radius 64 \
  --stent_dir_name stent --vessel_dir_name vessel \
  --marker --marker_length_mm 2.0 --marker_diameter_mm 1.0 \
  --marker_inset_mm 0.0 --marker_mu_ratio 2.0
```

新增输出：

```text
corr_stent/<mode>_data/
├── marker_vol_gt/<sample>.nii.gz
├── marker_proj_gt/<sample>/proj_000.npz ... proj_359.npz
└── config/<phantom>.pickle
```

config 中的 `marker` 字段保存：形状、请求的 length/diameter/inset、mu ratio、volume spacing、两个端点的 physical/voxel centers、端点 tangents 和实际 marker voxel 数。marker 以中心线切向方向放置两个短实心圆柱，并且按 vessel mask 裁剪到血管范围内。

`marker_mu_ratio` 只用于后续把 marker 写入输入 image；3D marker NIfTI 本身是 occupancy mask。DTE label 使用支架和 marker 的二值并集，避免历史脚本把 label 数值 2 当作两倍衰减。

## 3. 输入数据和 helper

当前脚本的 CLI 仍有历史遗留：默认 `helper_path` 指向 `/data1/liuyang/stent/interior/data/costa/helper`，split 文件注释掉了 CT 版本并默认写成 topbrain MR 路径，同时 `to_process` 当前设置为 `None`。真正运行前需要检查：

```text
<helper_path>/<mode>_data/voxel_0.2/resample/
<helper_path>/<mode>_data/voxel_0.2/distance_map/
<helper_path>/<mode>_data/voxel_0.2/skeleton/
```

PLY 训练/测试目录由 `get_stent_generator_train/test` 读取；PLY 的 z 轴会被缩放到 segment 长度，再经过 `change_radius` 和 `map_stent_to_curve` 变形。`gen_stent/utils/gen_stent.py` 中的 `w_like_Stent`、`ring_Stent`、`sin_Stent` 和 `Stent_Ply` 是形状生成器；`gen_curve.py`、`curve_trans.py` 保留了更多旧曲线/坐标变换实验。

## 4. 其他上游版本的判定

- [`gen_stent_and_vessel_copy.py`](../legacy/corr_data/gen_stent_and_vessel_copy.py)：当前入口的近似复制，保留作历史对照；没有 marker 接口。
- [`gen_stent_and_vessel_copy_diffR.py`](../legacy/corr_data/gen_stent_and_vessel_copy_diffR.py)：支架半径变化的实验版本，参数和输出结构不同。
- [`gen_stent_and_vessel_with_no_proj.py`](../legacy/corr_data/gen_stent_and_vessel_with_no_proj.py)：生成 3D 支架/血管但跳过直接投影，适合只做体积实验；不是 DTE 360 投影主线。
- [`to_down/`](../legacy/corr_data/to_down/)：下采样/早期复制目录，内部又复制了一整套 utils；不要把它与当前 `utils/` 的输出混合。
- [`gen_correspond_stent.py`](../legacy/corr_data/gen_correspond_stent.py)、[`gen_correspond_vessel.py`](../legacy/corr_data/gen_correspond_vessel.py)：更早的分开生成 stent/vessel 版本。
- [`gen_proj_stent.py`](../legacy/corr_data/gen_proj_stent.py)、[`get_2d_from_3d.py`](../legacy/corr_data/get_2d_from_3d.py)：旧的直接 3D→2D/格式转换入口。
- [`gen_truncate_data.py`](../legacy/corr_data/gen_truncate_data.py)、[`gen_truncate_data_copy.py`](../legacy/corr_data/gen_truncate_data_copy.py)、[`gen_truncate_data_dte.py`](../legacy/corr_data/gen_truncate_data_dte.py)、[`gen_truncate_data_yes_fdk.py`](../legacy/corr_data/gen_truncate_data_yes_fdk.py)：旧的裁剪/重建实验；`dte` 和 `yes_fdk` 后缀代表不同实验，而不是当前 DTE 入口。
- [`nnUnet_format_train_data.py`](../legacy/corr_data/nnUnet_format_train_data.py)、[`nnUNet_format_test_data.py`](../legacy/corr_data/nnUNet_format_test_data.py)：旧 nnU-Net 3D 格式整理脚本，和 DTE/gen_data 下的 ROI 2D 格式化不是一回事。

这些文件不删除，原因是它们可以帮助追溯现有 `costa`、`topbrain`、`proj` 和 `dtr` 数据是怎样产生的；但重新生成 marker 数据时只使用当前入口，并在输出根目录保存参数配置。

