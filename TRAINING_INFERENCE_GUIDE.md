# nnUNetv2 统一版本：训练与推理完整指南

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [实验规划与预处理](#3-实验规划与预处理)
4. [训练](#4-训练)
   - 4.1 [标准训练](#41-标准训练)
   - 4.2 [切换 Trainer](#42-训练时切换-trainer损失函数--策略)
   - 4.3 [DTM 训练](#43-dtm-训练)
   - 4.4 [Dual Containment 训练](#44-dual-containment-训练两网络联合训练)
   - 4.5 [AnisoStent 训练](#45-anisostent-训练网络增强模块)
   - 4.6 [统一增强配置与迁移](#46-统一增强配置与迁移)
5. [推理](#5-推理)
6. [后处理](#6-后处理)
7. [评估](#7-评估)
8. [集成与寻找最佳配置](#8-集成与寻找最佳配置)
9. [模型导出与共享](#9-模型导出与共享)
10. [模式切换速查](#10-模式切换速查)
11. [完整工作流示例](#11-完整工作流示例)

---

## 1. 环境准备

### 1.1 环境变量

```bash
# 必须设置
export nnUNet_raw=/path/to/nnUNet_raw            # 原始数据目录
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed  # 预处理数据目录
export nnUNet_results=/path/to/nnUNet_results      # 训练结果目录

# 可选设置
export nnUNet_def_n_proc=8      # 默认进程数（默认 8）
export nnUNet_npp=3             # 推理预处理进程数（默认 3）
export nnUNet_nps=3             # 推理分割导出进程数（默认 3）
export nnUNet_compile=true      # 推理时启用 torch.compile
export CUDA_VISIBLE_DEVICES=0   # GPU 选择

# Windows 额外设置
# 如果使用 mmap 有问题，代码会自动处理（nnUNetDatasetBlosc2 已适配）
```

### 1.2 DTM 模式额外依赖

DTM（距离变换图）模式需要 ANTsPy：

```bash
pip install antspyx
```

标准模式和 Dual Containment 模式**不需要**此依赖。

---

## 2. 数据准备

### 2.1 标准数据集格式

原始数据需按以下结构组织：

```
nnUNet_raw/
├── Dataset001_BrainTumor/
│   ├── imagesTr/
│   │   ├── case001_0000.nii.gz    # 模态 0（如 T1）
│   │   ├── case001_0001.nii.gz    # 模态 1（如 T1ce）
│   │   ├── case002_0000.nii.gz
│   │   └── case002_0001.nii.gz
│   ├── labelsTr/
│   │   ├── case001.nii.gz         # 分割标注
│   │   └── case002.nii.gz
│   ├── imagesTs/                  # 可选：测试图像
│   │   └── case003_0000.nii.gz
│   └── dataset.json              # 数据描述文件
```

### 2.2 dataset.json 格式

使用辅助函数生成：

```python
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

generate_dataset_json(
    output_file="nnUNet_raw/Dataset001_BrainTumor/dataset.json",
    imagesTr_dir="nnUNet_raw/Dataset001_BrainTumor/imagesTr",
    imagesTs_dir="nnUNet_raw/Dataset001_BrainTumor/imagesTs",  # 可选，无测试集用 None
    modalities=("T1", "T1ce"),           # 模态名称
    labels={                              # 标注字典
        "background": 0,
        "tumor": 1,
        # 或 regions-based: "tumor": (1, 2, 3)
    },
    dataset_name="Dataset001_BrainTumor",
    file_ending=".nii.gz",
)
```

### 2.3 数据格式转换

```bash
# 从 Medical Segmentation Decathlon 格式转换
nnUNetv2_convert_MSD_dataset \
    -i /path/to/Task05_Prostate \
    -overwrite_id 001 \
    -np 8

# 从旧版 nnUNet v1 格式转换
nnUNetv2_convert_old_nnUNet_dataset \
    /path/to/old_dataset \
    /path/to/nnUNet_raw/Dataset001_BrainTumor

# 验证数据集完整性
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

---

## 3. 实验规划与预处理

### 3.1 一键执行（推荐）

```bash
# 对数据集 1 执行所有步骤
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# 多个数据集
nnUNetv2_plan_and_preprocess -d 1 2 3 --verify_dataset_integrity

# 仅执行特定配置
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres

# 自定义 planner
nnUNetv2_plan_and_preprocess -d 1 -pl ExperimentPlanner
```

### 3.2 分步执行

```bash
# Step 2a: 提取指纹
nnUNetv2_extract_fingerprint -d 1 --verify_dataset_integrity

# Step 2b: 实验规划
nnUNetv2_plan_experiment -d 1

# 规划时指定 GPU 显存（用于自动确定 batch size / patch size）
nnUNetv2_plan_experiment -d 1 -gpu_memory_target 24

# Step 2c: 预处理
nnUNetv2_preprocess -d 1 -c 2d 3d_fullres 3d_lowres -np 8 4 8
```

### 3.3 DTM 预处理（仅 DTM 模式需要）

**在标准预处理之后**执行：

```bash
# 单数据集
python nnunetv2/nnunetv2/run/run_preprocess_dtm.py \
    Dataset001_BrainTumor 3d_fullres

# 自定义 plans
python nnunetv2/nnunetv2/run/run_preprocess_dtm.py \
    Dataset001_BrainTumor 3d_fullres \
    -p nnUNetPlans \
    -np 8

# verbose 模式
python nnunetv2/nnunetv2/run/run_preprocess_dtm.py \
    Dataset001_BrainTumor 3d_fullres --verbose
```

此步骤会为每个训练样本计算并存储距离变换图（DTM）文件（`_dtm_class0.b2nd`、`_dtm_class1.b2nd` 等）。

---

## 4. 训练

### 4.1 标准训练

```bash
# 基本用法
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD

# 示例：数据集 1，3d_fullres 配置，fold 0
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0

# 训练所有 5 折（顺序执行）
nnUNetv2_train Dataset001_BrainTumor 3d_fullres all

# 使用自定义 plans
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -p nnUNetPlans

# 从断点继续训练
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --c

# 仅验证（无需重新训练）
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --val

# 使用最优 checkpoint 验证
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --val_best

# 加载预训练权重
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
    -pretrained_weights /path/to/checkpoint.pth

# 多 GPU 训练（DDP）
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -num_gpus 4

# CPU 训练
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -device cpu

# 禁用 checkpoint 存储（节省磁盘空间）
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 --disable_checkpointing
```

**输出目录结构**：
```
nnUNet_results/Dataset001_BrainTumor/
└── nnUNetTrainer__nnUNetPlans__3d_fullres/
    ├── fold_0/
    │   ├── checkpoint_latest.pth
    │   ├── checkpoint_best.pth
    │   ├── checkpoint_final.pth
    │   ├── progress.png
    │   └── validation/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    └── fold_4/
```

### 4.2 训练时切换 Trainer（损失函数 / 策略）

```bash
# 使用纯 Dice Loss
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerDiceLoss

# 使用纯 CE Loss
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerCELoss

# 使用 TopK Loss
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerTopkLoss

# 不使用 Deep Supervision
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerNoDeepSupervision

# 禁用数据增强
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerNoDA

# Adam 优化器
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerAdam

# 5 epoch 快速基准测试
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs

# 自定义 epoch 数
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainer_250epochs
```

### 4.3 DTM 训练

```bash
# DTM 监控模式（GSL 仅记录，不参与反向传播 — 安全）
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerDTM

# DTM 完整模式（GSL 参与反向传播，epoch ≥ 250 后 alpha=0.5）
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 -tr nnUNetTrainerDTM_GSL

# DTM + 多 GPU
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
    -tr nnUNetTrainerDTM_GSL -num_gpus 4

# DTM + 从断点继续
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
    -tr nnUNetTrainerDTM_GSL --c
```

**两种 DTM 模式对比**：

| 属性 | `nnUNetTrainerDTM` | `nnUNetTrainerDTM_GSL` |
|------|---------------------|--------------------------|
| 反向传播损失 | 仅 DC+CE | DC+CE + α×GSL |
| GSL 作用 | 仅记录（监控） | 参与训练 |
| 风险 | 低（等同于标准训练） | 中（GSL 可能影响收敛） |
| 推荐场景 | 调试、初次尝试 | 追求更好的表面分割 |

### 4.4 Dual Containment 训练（两网络联合训练）

用于训练两个网络对**不同数据集**的联合约束（如：器官 A 必须在器官 B 内部）。

```bash
# 基本用法
nnUNetv2_train_dual_containment DATASET_A DATASET_B CONFIG FOLD

# 完整示例
nnUNetv2_train_dual_containment \
    Dataset001_Liver Dataset002_Tumor 3d_fullres 0 \
    --warmup_epochs 500 \
    --total_epochs 1000 \
    --lambda_containment 0.2 \
    --ramp_epochs 50 \
    --class_idx_A 1 \
    --class_idx_B 1

# 使用 DTM Trainer 进行 Dual Containment
nnUNetv2_train_dual_containment \
    Dataset001_Liver Dataset002_Tumor 3d_fullres 0 \
    -trA nnUNetTrainerDTM_GSL \
    -trB nnUNetTrainerDTM_GSL \
    --warmup_epochs 300 \
    --total_epochs 500

# 加载预训练权重后联合训练
nnUNetv2_train_dual_containment \
    Dataset001_Liver Dataset002_Tumor 3d_fullres 0 \
    -pretrained_weights_A /path/to/checkpoint_A.pth \
    -pretrained_weights_B /path/to/checkpoint_B.pth

# 从联合训练断点继续
nnUNetv2_train_dual_containment \
    Dataset001_Liver Dataset002_Tumor 3d_fullres 0 --continue_joint
```

**Dual Containment 参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--warmup_epochs` | 500 | 独立预训练 epoch 数（不施加 containment loss） |
| `--total_epochs` | 1000 | 总 epoch 数 |
| `--lambda_containment` | 0.2 | containment loss 的最大权重 |
| `--ramp_epochs` | 50 | containment loss 从 0 到 max 的线性递增 epoch 数 |
| `--class_idx_A` | 1 | 网络 A 中被约束的类别索引 |
| `--class_idx_B` | 1 | 网络 B 中作为容器的类别索引 |
| `--containment_margin` | 0.0 | 容忍边界（A 允许超出 B 的额度） |
| `--save_every` | 50 | 检查点保存间隔 |
| `--skip_final_validation` | False | 跳过最终验证 |

**Containment Loss 原理**：

```
loss_containment = mean(relu(prob_A - prob_B - margin)^2)
```

惩罚网络 A 对 class_idx_A 的预测概率超过网络 B 对 class_idx_B 的情况，从而强制 A ⊂ B 的空间关系。

**要求**：两个数据集必须具有**相同的 case 标识符**（病例一一对应）。

### 4.5 AnisoStent 训练（网络增强模块）

nnUNet 标准 U-Net 之外，本版本新增了两个可插拔的网络增强模块和一个几何一致性损失：

| 模块 | 作用 | 适用场景 |
|------|------|----------|
| **AnisotropicAttention** | 3 路各向异性 3D 卷积 + 通道注意力，抑制方向性条纹伪影 | 有限角度反投影 CT、稀疏视角重建 |
| **TubularEnhancement** | 4 路多尺度膨胀卷积 + 残差，恢复细管状结构 | 支架、血管等薄结构分割 |
| **ProjectionConsistencyLoss** | 可微前向投影 + MSE 损失，强制 3D 几何一致性 | 已知投影几何的 CT 重建分割 |

三个模块都使用残差连接设计，保证即使学不到有用信息时网络也能退化到标准 U-Net。额外参数量仅约 **2,200 个**（不到标准 U-Net 的 0.01%）。

#### 4.5.1 仅网络模块（AnisoStentUNet）

```bash
# 同时启用各向异性注意力 + 管状结构增强
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
    -tr nnUNetTrainerAnisoStent

# 5 折训练
for fold in 0 1 2 3 4; do
    nnUNetv2_train Dataset001_BrainTumor 3d_fullres $fold \
        -tr nnUNetTrainerAnisoStent
done
```

#### 4.5.2 网络模块 + 投影一致性损失

```bash
# 默认投影角度 [0°, 30°, 60°]，权重 0.1
nnUNetv2_train Dataset001_BrainTumor 3d_fullres 0 \
    -tr nnUNetTrainerAnisoStent_ProjCons

# 自定义投影角度和权重的子类示例
python -c "
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_AnisoStent import \
    nnUNetTrainerAnisoStent_ProjCons

class MyProjConsTrainer(nnUNetTrainerAnisoStent_ProjCons):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.projection_angles_deg = [0, 20, 40]   # 3 个视角在 40° 范围内
        self.proj_cons_weight = 0.05               # 较小的投影一致性权重
"
```

#### 4.5.3 AnisoStent + Dual Containment（联合训练）

AnisoStent 的 A/B stub trainer 可直接用于 dual containment：

```bash
# 两个网络都使用 AnisoStentUNet 架构（无 ProjCons）
nnUNetv2_train_dual_containment \
    Dataset001_Organ Dataset002_Stent 3d_fullres 0 \
    -trA nnUNetTrainerAnisoStent_A \
    -trB nnUNetTrainerAnisoStent_B \
    --warmup_epochs 500 --total_epochs 1000 \
    --class_idx_A 1 --class_idx_B 1

# 两个网络都使用 AnisoStentUNet + ProjCons
nnUNetv2_train_dual_containment \
    Dataset001_Organ Dataset002_Stent 3d_fullres 0 \
    -trA nnUNetTrainerAnisoStent_ProjCons_A \
    -trB nnUNetTrainerAnisoStent_ProjCons_B \
    --warmup_epochs 500 --total_epochs 1000 \
    --lambda_containment 0.2 \
    --class_idx_A 1 --class_idx_B 1

# 混合：网络 A 用标准 U-Net，网络 B 用 AnisoStentUNet
nnUNetv2_train_dual_containment \
    Dataset001_Organ Dataset002_Stent 3d_fullres 0 \
    -trA nnUNetTrainerA \
    -trB nnUNetTrainerAnisoStent_B
```

**原理**：`CombinedSegProjLoss` 将 DC+CE 和 ProjCons 封装为一个 `nn.Module`，其 `forward(output, target)` 签名与 dual training 脚本的 `trainer.loss(output, target)` 调用完全兼容。

#### 4.5.4 AnisoStent 推理

推理方式与标准模型完全相同：

```bash
# 基于数据集
nnUNetv2_predict \
    -i input/ -o output/ \
    -d Dataset001_BrainTumor -c 3d_fullres -f 0 \
    -tr nnUNetTrainerAnisoStent_ProjCons

# 基于模型文件夹
nnUNetv2_predict_from_modelfolder \
    -i input/ -o output/ \
    -m nnUNet_results/Dataset001_BrainTumor/\
nnUNetTrainerAnisoStent_ProjCons__nnUNetPlans__3d_fullres
```

AnisoStent 网络模块在推理时同样生效——各向异性注意力模块作用在输入上，管状结构增强模块作用在输出上。

#### 4.6 统一增强配置与迁移

Tasks 1–4 将网络增强和辅助损失收敛到一个 `EnhancementConfig`：

| 配置项 | 作用 |
|------|------|
| `enable_anisotropic_attention` | 在 backbone 输入前启用 AnisotropicAttention |
| `enable_tubular_enhancement` | 在最终分辨率输出上启用 TubularEnhancement |
| `projection_angles_deg` / `projection_consistency_weight` | 启用 ProjectionConsistencyLoss 及其投影角度/权重 |
| `dtm_mode` / `dtm_weight` / `dtm_active_weight` | 选择 DTM 的 `disabled`、`monitor` 或 `active` 模式及权重 |

现有 trainer 名称仍然兼容，并映射到等价预设：

| 旧 trainer | 统一配置等价项 |
|------|------|
| `nnUNetTrainerAnisoStent` | Aniso + Tubular |
| `nnUNetTrainerAnisoStent_ProjCons` | Aniso + Tubular + ProjCons（`[0, 30, 60]`，权重 `0.1`） |
| `nnUNetTrainerDTM` | DTM `monitor`（默认在 epoch 250 后可激活） |
| `nnUNetTrainerDTM_GSL` | DTM `active`（权重 `1.0`，激活后系数 `0.5`） |

新代码可直接构造统一网络和损失；训练循环仍可使用标量兼容的 `forward(output, target)`，需要日志明细时调用 `breakdown(...)`：

```python
from nnunetv2.training.pipeline_config import EnhancementConfig
from nnunetv2.training.pipeline_trainer import build_composite_loss, build_network

config = EnhancementConfig(
    enable_anisotropic_attention=True,
    enable_tubular_enhancement=True,
    projection_angles_deg=(0.0, 30.0, 60.0),
    projection_consistency_weight=0.1,
    dtm_mode="active",
    dtm_weight=1.0,
)
network = build_network(
    architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
    num_input_channels, num_output_channels, enable_deep_supervision, config,
)
loss = build_composite_loss(config, segmentation_loss, label_manager)
breakdown = loss.breakdown(output, target, auxiliary_targets, epoch=epoch)
```

迁移时保留原 trainer 名称即可逐步切换：先用 `EnhancementConfig.from_legacy_preset("nnUNetTrainerAnisoStent_ProjCons")` 验证等价行为，再将配置显式传给 `build_network` 和 `build_composite_loss`。DTM 模式仍需要预先生成 DTM 文件；推理阶段不需要提供 DTM 输入。

---

## 5. 推理

### 5.1 标准推理（基于数据集）

```bash
# 基本用法
nnUNetv2_predict \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -d DATASET \
    -c CONFIG

# 完整示例
nnUNetv2_predict \
    -i /path/to/test_images \
    -o /path/to/predictions \
    -d Dataset001_BrainTumor \
    -c 3d_fullres \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    -chk checkpoint_best.pth \
    -step_size 0.5 \
    -npp 3 \
    -nps 3 \
    -device cuda

# 使用 torch.compile 加速
export nnUNet_compile=true
nnUNetv2_predict -i input/ -o output/ -d 1 -c 3d_fullres -f 0

# 保存 softmax 概率（集成所需）
nnUNetv2_predict -i input/ -o output/ -d 1 -c 3d_fullres -f 0 --save_probabilities

# 禁用 TTA（测试时增强）— 更快，但精度可能略降
nnUNetv2_predict -i input/ -o output/ -d 1 -c 3d_fullres -f 0 --disable_tta

# 继续中断的预测（跳过已有文件）
nnUNetv2_predict -i input/ -o output/ -d 1 -c 3d_fullres --continue_prediction

# 分片预测（在多台机器上并行）
# 机器 1：
nnUNetv2_predict -i input/ -o output/ -d 1 -c 3d_fullres \
    -num_parts 4 -part_id 0
# 机器 2：
nnUNetv2_predict -i input/ -o output/ -d 1 -c 3d_fullres \
    -num_parts 4 -part_id 1
# ...
```

### 5.2 基于模型文件夹推理（无需 nnUNet_results）

```bash
nnUNetv2_predict_from_modelfolder \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -m /path/to/model_folder \
    -f 0 1 2 3 4 \
    -step_size 0.5 \
    --disable_tta \
    --save_probabilities
```

模型文件夹应包含：`fold_0/checkpoint_best.pth`、`dataset.json`、`plans.json`、`dataset_fingerprint.json`。

### 5.3 推理预测的参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i` | 必需 | 输入图像文件夹 |
| `-o` | 必需 | 输出预测文件夹 |
| `-d` | 必需（模式1） | 数据集名称或 ID |
| `-c` | 必需（模式1） | 配置：`2d`, `3d_fullres`, `3d_lowres`, `3d_cascade_fullres` |
| `-m` | 必需（模式2） | 模型文件夹路径 |
| `-f` | `0 1 2 3 4` | 使用的 fold |
| `-tr` | `nnUNetTrainer` | Trainer 类名 |
| `-p` | `nnUNetPlans` | Plans 标识符 |
| `-chk` | `checkpoint_final.pth` | 使用的 checkpoint |
| `-step_size` | `0.5` | 滑动窗口步长（0-1，越小越慢但可能越准） |
| `--disable_tta` | False | 禁用测试时镜像增强 |
| `--save_probabilities` | False | 保存 `.npz` 概率文件 |
| `--continue_prediction` | False | 跳过已存在的预测 |
| `-npp` | `3` | 预处理进程数 |
| `-nps` | `3` | 分割导出进程数 |
| `-num_parts` / `-part_id` | — | 分片预测 |
| `-prev_stage_predictions` | — | 级联模型的上一阶段预测 |
| `-device` | `cuda` | `cuda`, `cpu`, `mps` |

### 5.4 DTM 模型的推理

DTM 模型在推理时**不需要 DTM 输入**（仅训练时使用）。推理方式与标准模型完全相同：

```bash
# 与标准推理完全一致，仅需指定 trainer 类名
nnUNetv2_predict \
    -i input/ \
    -o output/ \
    -d Dataset001_BrainTumor \
    -c 3d_fullres \
    -f 0 \
    -tr nnUNetTrainerDTM_GSL    # 指定 DTM trainer

# 或使用基于模型文件夹的方式
nnUNetv2_predict_from_modelfolder \
    -i input/ \
    -o output/ \
    -m nnUNet_results/Dataset001_BrainTumor/nnUNetTrainerDTM_GSL__nnUNetPlans__3d_fullres
```

### 5.5 Dual Containment 模型的推理

训练完成后，分别对两个网络进行推理：

```bash
# 网络 A
nnUNetv2_predict_from_modelfolder \
    -i input/ \
    -o output/A_predictions/ \
    -m nnUNet_results/Dataset001_Liver_Dataset002_Tumor/nnUNetTrainerA__nnUNetPlans__3d_fullres

# 网络 B
nnUNetv2_predict_from_modelfolder \
    -i input/ \
    -o output/B_predictions/ \
    -m nnUNet_results/Dataset001_Liver_Dataset002_Tumor/nnUNetTrainerB__nnUNetPlans__3d_fullres
```

---

## 6. 后处理

```bash
# 首先确定是否需要后处理（基于验证集性能）
nnUNetv2_determine_postprocessing \
    -m MODEL_FOLDER \
    -f 0 1 2 3 4

# 应用后处理到预测结果
nnUNetv2_apply_postprocessing \
    -i PREDICTIONS_FOLDER \
    -o POSTPROCESSED_FOLDER \
    -pp_pkl_file /path/to/postprocessing.pkl

# 对多个预测文件夹批量应用
nnUNetv2_apply_postprocessing \
    -i predictions_config1 predictions_config2 \
    -o postprocessed_config1 postprocessed_config2 \
    -pp_pkl_file postprocessing.pkl
```

后处理内容：移除除最大连通分量外的所有小连通域（仅在提高验证集分数时启用）。

---

## 7. 评估

```bash
# 评估预测结果 vs 真实标注
nnUNetv2_evaluate_folder \
    -pred_folder PREDICTIONS_FOLDER \
    -gt_folder GROUND_TRUTH_FOLDER \
    -djfile dataset.json \
    -pfile plans.json \
    -o output_summary.json

# 简化评估
nnUNetv2_evaluate_simple \
    -pred_folder PREDICTIONS_FOLDER \
    -gt_folder GROUND_TRUTH_FOLDER \
    -l 1 2 3               # 评估的标签

# 累计交叉验证结果
nnUNetv2_accumulate_crossval_results \
    -d 1 \
    -c 3d_fullres
```

---

## 8. 集成与寻找最佳配置

### 8.1 自动寻找最佳配置

```bash
# 自动测试所有配置并找出最佳
nnUNetv2_find_best_configuration \
    -d 1 \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    -c 2d 3d_fullres 3d_lowres 3d_cascade_fullres
```

### 8.2 手动集成

```bash
# 第一步：分别推理并保存概率
nnUNetv2_predict -i input/ -o output_2d/ -d 1 -c 2d -f 0 --save_probabilities
nnUNetv2_predict -i input/ -o output_3d/ -d 1 -c 3d_fullres -f 0 --save_probabilities

# 第二步：集成（概率平均）
nnUNetv2_ensemble \
    -i output_2d/ output_3d/ \
    -o ensemble_output/
```

---

## 9. 模型导出与共享

```bash
# 导出训练好的模型
nnUNetv2_export_model_to_zip \
    -d DATASET \
    -o /path/to/output.zip \
    -c 3d_fullres \
    -tr nnUNetTrainer \
    -p nnUNetPlans \
    -f 0 1 2 3 4 \
    -chk checkpoint_best.pth

# 从 URL 下载预训练模型
nnUNetv2_download_pretrained_model_by_url \
    https://example.com/model.zip

# 从本地 zip 安装预训练模型
nnUNetv2_install_pretrained_model_from_zip \
    /path/to/model.zip
```

---

## 10. 模式切换速查

### 10.1 训练模式

| 模式 | 命令 | 说明 |
|------|------|------|
| **标准** | `nnUNetv2_train DATASET CONFIG FOLD` | DC+CE Loss，基准训练 |
| **标准 + 自定义损失** | `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerDiceLoss` | 纯 Dice / CE / TopK 等 |
| **DTM 监控** | `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerDTM` | GSL 仅记录，安全 |
| **DTM 活跃** | `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerDTM_GSL` | GSL 参与反向传播 |
| **AnisoStent** | `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerAnisoStent` | 各向异性注意力 + 管状增强 |
| **AnisoStent + ProjCons** | `nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainerAnisoStent_ProjCons` | 网络增强 + 投影一致性 |
| **Dual Containment** | `nnUNetv2_train_dual_containment DA DB CONFIG FOLD` | 两网络联合约束 |
| **Dual + DTM** | `nnUNetv2_train_dual_containment DA DB CONFIG FOLD -trA nnUNetTrainerDTM_GSL -trB nnUNetTrainerDTM_GSL` | 联合训练 + 表面损失 |
| **Dual + AnisoStent** | `nnUNetv2_train_dual_containment DA DB CONFIG FOLD -trA nnUNetTrainerAnisoStent_A -trB nnUNetTrainerAnisoStent_B` | 联合训练 + 网络增强 |
| **Dual + AnisoStent + ProjCons** | `nnUNetv2_train_dual_containment DA DB CONFIG FOLD -trA nnUNetTrainerAnisoStent_ProjCons_A -trB nnUNetTrainerAnisoStent_ProjCons_B` | 联合 + 网络增强 + 投影一致性 |

### 10.2 可用的 Trainer 类名一览

#### 损失函数变体
- `nnUNetTrainer` — 标准 DC+CE Loss
- `nnUNetTrainerDiceLoss` — 纯 Dice Loss
- `nnUNetTrainerCELoss` — 纯 Cross-Entropy Loss
- `nnUNetTrainerTopkLoss` — TopK Loss

#### 数据增强变体
- `nnUNetTrainerNoDA` — 无数据增强
- `nnUNetTrainerDA5` — DA5 增强策略
- `nnUNetTrainerDAOrd0` — 无 spatial 变换
- `nnUNetTrainerNoMirroring` — 无镜像增强
- `nnUNetTrainer_noDummy2DDA` — 无伪 2D 增强

#### 网络架构变体
- `nnUNetTrainerBN` — 带 BatchNorm
- `nnUNetTrainerNoDeepSupervision` — 无深度监督

#### 优化器变体
- `nnUNetTrainerAdam` — Adam 优化器
- `nnUNetTrainerAdan` — Adan 优化器

#### 学习率调度变体
- `nnUNetTrainerCosAnneal` — 余弦退火
- `nnUNetTrainer_warmup` — Warmup

#### 训练长度变体
- `nnUNetTrainerBenchmark_5epochs` — 5 epoch 快速测试
- `nnUNetTrainer_Xepochs` — 自定义 epoch 数（需设置 `num_epochs` 属性）
- `nnUNetTrainer_Xepochs_NoMirroring` — 自定义 epoch + 无镜像

#### DTM 变体
- `nnUNetTrainerDTM` — DTM 监控模式
- `nnUNetTrainerDTM_GSL` — DTM 完整模式

#### AnisoStent 变体（网络增强模块）
- `nnUNetTrainerAnisoStent` — AnisoStentUNet（各向异性注意力 + 管状增强）
- `nnUNetTrainerAnisoStent_ProjCons` — AnisoStentUNet + 投影一致性损失

#### AnisoStent + Dual Containment
- `nnUNetTrainerAnisoStent_A` — AnisoStentUNet 网络 A
- `nnUNetTrainerAnisoStent_B` — AnisoStentUNet 网络 B
- `nnUNetTrainerAnisoStent_ProjCons_A` — AnisoStentUNet + ProjCons 网络 A
- `nnUNetTrainerAnisoStent_ProjCons_B` — AnisoStentUNet + ProjCons 网络 B

#### Dual Containment（基础）
- `nnUNetTrainerA` — 网络 A（空子类，区分输出目录）
- `nnUNetTrainerB` — 网络 B（空子类，区分输出目录）

### 10.3 推理模式总结

| 模式 | 命令 | 特点 |
|------|------|------|
| **标准推理** | `nnUNetv2_predict -i IN -o OUT -d DATASET -c CONFIG` | 依赖 nnUNet_results 目录结构 |
| **模型文件夹推理** | `nnUNetv2_predict_from_modelfolder -i IN -o OUT -m MODELDIR` | 不依赖 nnUNet_results |
| **保存概率** | `... --save_probabilities` | 集成所需 |
| **禁用 TTA** | `... --disable_tta` | 更快（约 8×），精度略降 |
| **分片并行** | `... -num_parts N -part_id M` | 多机并行推理 |
| **断点续推** | `... --continue_prediction` | 跳过已完成文件 |

---

## 11. 完整工作流示例

### 11.1 标准工作流（从头到尾）

```bash
# === 环境设置 ===
export nnUNet_raw=/data/nnUNet_raw
export nnUNet_preprocessed=/data/nnUNet_preprocessed
export nnUNet_results=/data/nnUNet_results
export CUDA_VISIBLE_DEVICES=0,1

# === Step 1: 数据准备 ===
# 假设数据已按 nnUNet 格式放在 nnUNet_raw/Dataset001_Liver/
# 创建 dataset.json
python -c "
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
generate_dataset_json(
    output_file='/data/nnUNet_raw/Dataset001_Liver/dataset.json',
    imagesTr_dir='/data/nnUNet_raw/Dataset001_Liver/imagesTr',
    imagesTs_dir=None,
    modalities=('CT',),
    labels={'background': 0, 'liver': 1, 'tumor': 2},
    dataset_name='Dataset001_Liver',
    file_ending='.nii.gz',
)
"

# === Step 2: 规划与预处理 ===
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# === Step 3: 训练（5折） ===
for fold in 0 1 2 3 4; do
    nnUNetv2_train Dataset001_Liver 3d_fullres $fold -num_gpus 2
done

# === Step 4: 自动寻找最佳配置 ===
nnUNetv2_find_best_configuration -d 1 -f 0 1 2 3 4

# === Step 5: 推理 ===
nnUNetv2_predict \
    -i /data/nnUNet_raw/Dataset001_Liver/imagesTs \
    -o /data/predictions/Liver \
    -d Dataset001_Liver \
    -c 3d_fullres \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -step_size 0.5

# === Step 6: 后处理 ===
MODELDIR=$(ls -d /data/nnUNet_results/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_fullres)
nnUNetv2_determine_postprocessing -m $MODELDIR -f 0 1 2 3 4

nnUNetv2_apply_postprocessing \
    -i /data/predictions/Liver \
    -o /data/predictions/Liver_pp \
    -pp_pkl_file $MODELDIR/crossval_results_folds_0_1_2_3_4/postprocessing.pkl
```

### 11.2 DTM 工作流

```bash
# === 环境设置（同上） ===
# 额外安装：pip install antspyx

# === Step 1-2: 数据准备 + 标准预处理（同上） ===
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# === Step 2b: DTM 额外预处理 ===
python nnunetv2/nnunetv2/run/run_preprocess_dtm.py \
    Dataset001_Liver 3d_fullres -np 8

# === Step 3: DTM 训练 ===
# 先用监控模式测试
nnUNetv2_train Dataset001_Liver 3d_fullres 0 -tr nnUNetTrainerDTM -num_gpus 2

# 确认 gsl_loss 正常下降后，使用完整模式
nnUNetv2_train Dataset001_Liver 3d_fullres 0 -tr nnUNetTrainerDTM_GSL -num_gpus 2

# 5折训练
for fold in 1 2 3 4; do
    nnUNetv2_train Dataset001_Liver 3d_fullres $fold \
        -tr nnUNetTrainerDTM_GSL -num_gpus 2
done

# === Step 4-6: 评估/推理/后处理（与标准相同，改 -tr 即可） ===
nnUNetv2_predict \
    -i /data/nnUNet_raw/Dataset001_Liver/imagesTs \
    -o /data/predictions/Liver_DTM \
    -d Dataset001_Liver -c 3d_fullres -f 0 1 2 3 4 \
    -tr nnUNetTrainerDTM_GSL
```

### 11.3 Dual Containment 工作流

```bash
# === 前提：两个数据集有相同 case 标识符 ===
# Dataset001_Organ: 器官分割（如 liver）
# Dataset002_Lesion: 病灶分割（如 liver tumor）
# 两数据集病例一一对应，identifier 相同

# === Step 1: 分别对两个数据集执行规划与预处理 ===
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity  # Organ
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity  # Lesion

# === Step 2: 联合训练 ===
nnUNetv2_train_dual_containment \
    Dataset001_Organ Dataset002_Lesion 3d_fullres 0 \
    --warmup_epochs 500 \
    --total_epochs 1000 \
    --lambda_containment 0.2 \
    --ramp_epochs 50 \
    --class_idx_A 1 \       # Organ 的 liver 类
    --class_idx_B 1 \       # Lesion 的 tumor 类（必须在 liver 内）
    --containment_margin 0

# === Step 3: 推理两个网络 ===
# 网络 A（Organ）
nnUNetv2_predict_from_modelfolder \
    -i /data/test_images/ \
    -o /data/predictions/Organ/ \
    -m nnUNet_results/Dataset001_Organ/nnUNetTrainerA__nnUNetPlans__3d_fullres

# 网络 B（Lesion）
nnUNetv2_predict_from_modelfolder \
    -i /data/test_images/ \
    -o /data/predictions/Lesion/ \
    -m nnUNet_results/Dataset001_Organ/nnUNetTrainerB__nnUNetPlans__3d_fullres

# === Step 4: 验证 containment 效果 ===
# 对比有无 containment 的病灶预测重叠度
nnUNetv2_evaluate_folder \
    -pred_folder /data/predictions/Organ/ \
    -gt_folder /data/ground_truth/Organ/ \
    -djfile /data/nnUNet_preprocessed/Dataset001_Organ/dataset.json \
    -pfile /data/nnUNet_preprocessed/Dataset001_Organ/nnUNetPlans.json
```

### 11.4 Dual Containment + DTM 联合工作流

```bash
# === Step 1: 数据准备 + 标准预处理 ===
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres
nnUNetv2_plan_and_preprocess -d 2 -c 3d_fullres

# === Step 2: DTM 预处理（两个数据集都需要） ===
python nnunetv2/nnunetv2/run/run_preprocess_dtm.py Dataset001_Organ 3d_fullres -np 8
python nnunetv2/nnunetv2/run/run_preprocess_dtm.py Dataset002_Lesion 3d_fullres -np 8

# === Step 3: Dual Containment + DTM 联合训练 ===
nnUNetv2_train_dual_containment \
    Dataset001_Organ Dataset002_Lesion 3d_fullres 0 \
    -trA nnUNetTrainerDTM_GSL \
    -trB nnUNetTrainerDTM_GSL \
    --warmup_epochs 300 \
    --total_epochs 500 \
    --lambda_containment 0.15 \
    --ramp_epochs 30
```

### 11.5 AnisoStent 工作流（有限角度反投影 CT 分割）

适用于：60° 范围内 3 个视角反投影 → 分割目标结构。

```bash
# === 前提：数据已预处理 ===
# 输入为多通道反投影体积（如 3 个视角各一反投影 → 3 通道）
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# === Step 1: 先用 AnisoStentUNet 训练（无 ProjCons） ===
for fold in 0 1 2 3 4; do
    nnUNetv2_train Dataset001_LimitedAngle 3d_fullres $fold \
        -tr nnUNetTrainerAnisoStent -num_gpus 2
done

# === Step 2: 评估基线性能 ===
nnUNetv2_find_best_configuration -d 1 -f 0 1 2 3 4 \
    -tr nnUNetTrainerAnisoStent

# === Step 3: 如需更强的几何一致性，加上 ProjCons ===
for fold in 0 1 2 3 4; do
    nnUNetv2_train Dataset001_LimitedAngle 3d_fullres $fold \
        -tr nnUNetTrainerAnisoStent_ProjCons -num_gpus 2
done

# === Step 4: 比较两种 trainer 的性能 ===
nnUNetv2_find_best_configuration -d 1 -f 0 1 2 3 4 \
    -tr nnUNetTrainerAnisoStent \
    -tr nnUNetTrainerAnisoStent_ProjCons

# === Step 5: 推理 ===
nnUNetv2_predict \
    -i /data/test_images/ \
    -o /data/predictions/AnisoStent_ProjCons/ \
    -d Dataset001_LimitedAngle -c 3d_fullres -f 0 1 2 3 4 \
    -tr nnUNetTrainerAnisoStent_ProjCons
```

### 11.6 AnisoStent + Dual Containment 工作流（支架分割 + 器官约束）

适用于：从包含支架的脑部反投影中分割支架 mask，同时用器官分割网络提供空间约束。

```bash
# === 数据集准备 ===
# Dataset001_BrainOrgan: 脑部器官分割（如血管/脑区）
# Dataset002_Stent:    支架分割
# 两数据集病例一一对应，identifier 相同

# === Step 1: 分别预处理 ===
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity

# === Step 2: 联合训练（AnisoStent + ProjCons 双网络） ===
nnUNetv2_train_dual_containment \
    Dataset001_BrainOrgan Dataset002_Stent 3d_fullres 0 \
    -trA nnUNetTrainerAnisoStent_ProjCons_A \
    -trB nnUNetTrainerAnisoStent_ProjCons_B \
    --warmup_epochs 500 \
    --total_epochs 1000 \
    --lambda_containment 0.2 \
    --ramp_epochs 50 \
    --class_idx_A 1 \       # 网络 A: 血管/器官类
    --class_idx_B 1 \       # 网络 B: 支架类（必须在血管内）
    --containment_margin 0

# === Step 3: 5 折训练 ===
for fold in 1 2 3 4; do
    nnUNetv2_train_dual_containment \
        Dataset001_BrainOrgan Dataset002_Stent 3d_fullres $fold \
        -trA nnUNetTrainerAnisoStent_ProjCons_A \
        -trB nnUNetTrainerAnisoStent_ProjCons_B \
        --warmup_epochs 500 --total_epochs 1000 \
        --lambda_containment 0.2 --ramp_epochs 50 \
        --class_idx_A 1 --class_idx_B 1
done

# === Step 4: 推理两个网络 ===
# 网络 A（器官/血管）
nnUNetv2_predict_from_modelfolder \
    -i /data/test/ -o /data/pred/Organ/ \
    -m nnUNet_results/Dataset001_BrainOrgan/\
nnUNetTrainerAnisoStent_ProjCons_A__nnUNetPlans__3d_fullres

# 网络 B（支架）
nnUNetv2_predict_from_modelfolder \
    -i /data/test/ -o /data/pred/Stent/ \
    -m nnUNet_results/Dataset001_BrainOrgan/\
nnUNetTrainerAnisoStent_ProjCons_B__nnUNetPlans__3d_fullres

# === Step 5: 后处理（利用 containment 约束精修） ===
# 支架预测 ∩ 器官预测 → 移除器官外的假阳性支架
```

---

## 附录 A：训练日志解读

训练过程中的关键日志指标：

| 日志项 | 含义 |
|--------|------|
| `train_loss` | 训练损失（标准 DC+CE） |
| `val_loss` | 验证损失 |
| `Pseudo dice` | 在线计算的伪 Dice（绿色曲线），近似真实 Dice |
| `gsl_loss` | 仅 DTM 模式：广义表面损失值 |
| `ema_fg_dice` | 前景 Dice 的指数移动平均（用于选择最佳 checkpoint） |
| `Epoch time` | 每 epoch 耗时 |

DTM 监控模式下，`train_loss` 仅反映 DC+CE，`gsl_loss` 独立记录不参与梯度。

## 附录 B：常用路径速查

```bash
# 预处理数据
$nnUNet_preprocessed/DatasetXXX_YYY/
├── dataset.json                      # 数据集元信息
├── dataset_fingerprint.json          # 指纹
├── nnUNetPlans.json                  # 实验计划
└── nnUNetPlans_3d_fullres/           # 预处理后数据
    └── *.{b2nd,pkl,_seg.b2nd}       # Blosc2 格式
    └── *_dtm_class*.b2nd            # 仅 DTM 模式

# 训练结果
$nnUNet_results/DatasetXXX_YYY/
└── {trainer}__{plans}__{config}/
    ├── fold_0/
    │   ├── checkpoint_best.pth
    │   ├── checkpoint_final.pth
    │   ├── checkpoint_latest.pth
    │   ├── progress.png
    │   └── validation/              # 最终验证结果
    └── ...
```

## 附录 C：DTM 模式原理简释

**DTM (Distance Transform Map)** 为每个体素计算到最近边界的带符号距离。

```
DTM(x) = distance(x, nearest_boundary)
```

- 边界体素: DTM ≈ 0
- 远离边界的内部体素: DTM >> 0
- 背景体素: DTM = max_diagonal（相当于无穷远）

**GSL (Generalized Surface Loss)** = `mean(DTM * Predicted_Probability)`

优化 GSL 意味着：鼓励模型对边界附近的体素给出更高的属于正确类别的概率（边界区域的 DTM 小 → 对损失贡献小 → 放松约束；内部区域的 DTM 大 → 预测错误时惩罚大）。

**CombinedLoss** = `DC_CE_Loss + alpha × GSL`

- `epoch < 250`: alpha = 0（仅 DC+CE）
- `epoch ≥ 250`: alpha = 0.5（加入表面损失）
- 先让模型学会大致分割，再精细调整表面

## 附录 D：AnisoStent 模块架构简释

### 数据流

```
输入 Back-projected Volume (B, C, D, H, W)
    │
    ▼
┌─────────────────────────────┐
│  AnisotropicAttention       │  ← 残差连接，抑制条纹伪影
│  3 路各向异性 3D 卷积        │    (1×3×3) axial / (3×1×3) sag / (3×3×1) cor
│  + SE 通道注意力融合 + 门控   │    参数量：~300
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Standard U-Net Backbone    │  ← PlainConvUNet / ResidualEncoderUNet
│  (不变)                      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  TubularEnhancement         │  ← 残差连接，恢复细管状结构
│  4 路膨胀 3D 卷积            │    dil=1,2,3,5 → 1×1×1 融合
│  + 残差                      │    参数量：~1,800
└─────────────────────────────┘
    │
    ▼
输出 Segmentation (B, num_classes, D, H, W)
    │  (训练时)
    ▼
┌─────────────────────────────┐
│  ProjectionConsistencyLoss  │  ← 可微前向投影 (affine_grid + sum)
│  MSE(proj(pred), proj(GT))  │    投影角度可配置，默认 [0°, 30°, 60°]
└─────────────────────────────┘
```

### 模块参数对比

| 模块 | 参数数量 | 占比 |
|------|----------|------|
| AnisotropicAttention | ~300 | < 0.001% |
| TubularEnhancement | ~1,800 | < 0.006% |
| U-Net Backbone | ~33M | 99.99% |
| **总计** | **~33M** | **100%** |

### 设计原则

- **残差连接**：两个模块都使用 `output = module(x) + x`，保证初始状态下网络行为与标准 U-Net 完全一致
- **Wrapper 模式**：不修改 `dynamic_network_architectures` 外部包，用 `AnisoStentUNet` 包裹标准 U-Net
- **Deep Supervision 兼容**：管状增强仅作用于最终（最高分辨率）输出，其余 DS 输出保持不变

## 附录 E：故障排查

| 问题 | 解决方案 |
|------|----------|
| DTM 模式报 `No module named 'ants'` | `pip install antspyx` |
| 多 GPU OOM（显存不足） | 减小 `-gpu_memory_target` 重新规划，或 `-num_gpus 1` |
| Dual 训练时 identifier 不匹配 | 确保两个数据集的 case 名称完全一致 |
| AnisoStent 训练速度慢 | 额外模块参数量 < 0.01%，额外开销可忽略。确认 `nnUNet_compile=false`（Windows 下默认关闭） |
| ProjCons Loss 不下降 | 确认投影角度与实际采集几何一致。默认 `[0,30,60]` 适用于绕 z 轴等间距采样的 3 视角场景 |
| ProjCons 报 shape mismatch | 确保输入体积各向同性（D≈H≈W），或修改 `differentiable_forward_projection` 中的旋转轴 |
| `nnUNetv2_train` 报 `No such file` | 检查环境变量是否设置：`echo $nnUNet_preprocessed` |
| Windows 下 batchgenerators 慢 | 正常现象，mmap 在 Windows 上禁用 |
| 推理时 `CUDA out of memory` | 调大 `-step_size`（如 0.8）或使用 `-device cpu` |
