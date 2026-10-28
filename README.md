# DTE/gen_data 文件与版本说明

本目录有三类代码：3D/2D 模拟数据生成、DTR 后处理、模型数据格式转换。下表中的“主线”表示当前重生成优先检查的路径；“历史”表示不删除但不建议与主线混合。

## 1. 生成和预处理脚本

| 文件 | 输入 | 输出/作用 | 状态 |
|---|---|---|---|
| `fill_image_hole.py` | 原始 image、vessel label NIfTI | `fill_hole_image`，把血管区域填成邻域估计值，供植入支架 | 主线前处理；`expand_mask` 实际固定迭代 3 次 |
| `resample_image.py` | NIfTI image/label | 目标 spacing 下的 NIfTI；image 用三次插值，label 用线性插值后阈值化 | 主线前处理，但 CLI 有硬编码路径和参数问题 |
| `gen_vol_gt_mask_proj.py` | 3D `vol_gt_mask` | 360 个角度、`1500×1500`、`0.308 mm` detector 的稀疏 ROI mask 投影 | 当前 ROI mask 主线 |
| `gen_stent_image_proj_mr.py` | MR/背景 image、stent `vol_gt`、ROI mask 投影 | MR→CT、加头骨、植入支架，输出 image/label 和四种稀疏 ROI 投影 | 当前 MR 支架主线；支持 marker |
| `gen_stent_image_proj_cta.py` | CTA image、stent `vol_gt`、ROI mask 投影 | HU→mu、植入支架并生成 360 投影 | CTA 分支；支持 marker |
| `gen_vessel_proj_mr.py` | MR image、vessel label、ROI mask，可选 stent | MR→CT、加头骨、血管造影剂，可选支架，输出 vessel label 投影 | `run.sh` 使用的主线；支持 marker |
| `gen_vessel_proj_cta.py` | CTA image、vessel label、ROI mask | HU→mu、加入血管造影剂并投影 | CTA 分支 |
| `gen_vessel_proj_mr_gradually.py` | 同 MR 血管路线，可选 stent | 把血管造影剂贡献在角度 0→150 逐渐增大，支架另外投影 | 实验分支；保留但不是默认主线 |
| `gen_vessel_vol_gt.py` | 旧的 source label、`vol_gt_mask` | 生成 `vol_gt` 与 `vol_gt_no_mask` | 旧/用途未完全确认；当前目录结构下通常不是第一步 |
| `ceshi.py` | 一个硬编码 MR 样本 | 生成 `other/ct_data.nii.gz` 供人工检查 | 调试脚本，不用于批量生成 |

所有投影脚本都使用 TIGRE `Ax`。常见 geometry 是 `DSD/DSO` 由 `utils/general.py` 设置、360 角度、detector 1500×1500、pixel 0.308；上游 `corr_data` 生成器另有 2048×2048/按体素 spacing 的投影，不能混用。

## 2. DTR 数据和后处理脚本

| 文件 | 作用 | 输入/输出重点 | 状态 |
|---|---|---|---|
| `nnUnet_format_roi_data.py` | ROI 投影转 nnU-Net 2D | 选 `0,20,...,340` 共 18 个角度，写 `imagesTr/labelsTr` 或 `imagesTs/labelsTs` | 早期格式化版本 |
| `nnUnet_format_roi_data_subthread.py` | 上述格式化的 split/子任务版本 | 支持 split JSON 和并行分片，避免同一 phantom 泄漏 | 推荐格式化入口 |
| `nnUnet_format_dtr_data.py` | 3D DTR NIfTI 转 nnU-Net 3D | 从 recon 的 `vol_train_no_masked/vol_gt_no_masked` 拷贝为 3D dataset | DTR 格式化 |
| `get_dtr_proj.py` | 从 360 投影选 DTR 角度块 | 旧的 `dtr/proj` 目录，按 `tar_range/tar_num` 选择 projection | 历史主线，使用前需修复/检查变量路径 |
| `get_dtr_map.py` | 原始 ROI image × nnU-Net 二值预测 | 生成 `predict/src_label`，后续喂给 DTE | 旧目录结构 |
| `dtr_post_preprocess.py` | 真实 `Dataset001_STENT` 的 DTE-MAP 预处理 | 同时处理 pred 和 label，写 `dtr_post/pred_dte_map`、`label_dte_map` | 较新的真实数据分支 |
| `get_dtr_proj_concat.py` | ROI depth 预测放回完整 detector | 结合 `vol_gt_mask_proj` 写 `predict/complete_label` | DTR 后处理 |
| `get_dtr_recon.py` | 选择的完整投影做 FDK/无 filter 重建 | 生成 recon 下的 3D image/GT/pred | DTR 后处理；排序匹配风险 |

`tar_range=60, tar_num=3` 这类设置通常表示从 180°中选择三个角度块；`post_process.sh` 中的 `180,180` 是另一个完整/密集实验设置，不能默认认为等价。

## 3. 评估与辅助文件

| 文件 | 作用 |
|---|---|
| `calculate_dice_2d.py` | 2D mask 指标 |
| `calculate_dice_3d.py` | 3D NIfTI 的 Dice、HD95、ASD |
| `calculate_dice_3d_1.py` | 3D 指标的另一个版本，结果口径需对照源码 |
| `utils/file_rw.py` | NIfTI/TIFF/稀疏 NPZ 的读取、写入、360 投影存盘 |
| `utils/general.py` | TIGRE geometry、HU→mu、MR→CT、加头骨、体素化；注意文件中 `hu_to_mu` 有后定义覆盖前定义的行为 |
| `utils/centerline.py` | 骨架图、连通组件、随机中心线段、平滑 |
| `utils/vessel.py` | 圆形 ROI 和 vessel 投影区域提取 |
| `utils/stent.py` | 支架 generator 列表、8 种 train 支架 + 2 种 test 支架、支架加粗 |
| `utils/postprocess.py` | 基于距离变换的加粗、半径变化 |
| `utils/marker.py` | 新增：将 stent/marker 体积组成 image 和二值 label；不依赖 TIGRE |
| `dtr_sh/nnune_2d.sh` | 旧 DTR 2D nnU-Net 预测，Dataset 903/904 |
| `dtr_sh/nnune_3d*.sh` | 旧 DTR 3D nnU-Net 预测，Dataset 905/906 |
| `dtr_sh/dte_map.sh` | 用 DTE model 生成 depth map，当前示例指向 `smile2023/dtr/proj` |
| `dtr_sh/post_process.sh` | 调 concat + recon 的旧批处理命令 |
| `run.sh` | 12 split 并行运行 MR vessel/stent 路线；当前只跑 test，6 GPU 每 GPU 2 进程 |
| `all_predict.sh` | 早期 `all_proj` 全量投影预测，Dataset 801 |
| `dte.ipynb`, `dte_2.ipynb`, `dte_3.ipynb` | 历史实验 notebook，参数不应作为 CLI 默认值 |
| `log/`, `nohup.out` | 运行日志，不是输入数据 |
| `gen_data.zip` | `gen_data` 目录的历史快照，不能代替当前源码 |

## 4. `other/` 子目录

- `gen_all_proj.py`：早期全 360 投影生成；当前有效代码主要只投 `vol_gt_mask`，与 `gen_vol_gt_mask_proj.py` 重复。
- `gen_roi_proj_from_all_proj.py`：从 `all_proj` 提取 ROI 的旧实现，核心代码几乎全部注释。
- `gen_ct_vessel_image_from_mra.py`：旧的 MRA→CT/血管填充值实现，核心代码几乎全部注释，后来逻辑放到 `ceshi.py`/MR 投影脚本。
- `trans.py`：把稀疏 `.npz` 转 jpg/tif 的可视化/格式转换工具，不生成新的物理投影。
- `test.sh`：磁盘/IO 测试，不属于 DTE 数据生成主线；运行前要检查其临时目录和覆盖行为。
- `ct_data.nii.gz`：调试生成的示例 NIfTI。

## 5. 版本组合建议

### 推荐组合

```text
fill_image_hole.py
→ resample_image.py
→ interior/data/corr_data/gen_stent_and_vessel.py
→ gen_vol_gt_mask_proj.py
→ gen_vessel_proj_mr.py 或 gen_stent_image_proj_mr.py
→ nnUnet_format_roi_data_subthread.py
→ DTE/model/script/get_data.py / gen_data_new.py
→ DTR 脚本
```

### 不建议直接混合

`all_proj` 与 `new_proj`、`proj` 与 `dtr/proj`、`new_proj` 与 `new_proj_2`、`full_dtr` 与 `dtr/recon`、不同 `radius`、不同 detector 尺寸、不同 `vol_gt_mask_proj`。它们即使文件名相似，也可能在尺寸、角度索引或衰减值上不兼容。

