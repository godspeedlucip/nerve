# 消融实验 — AnisoStentUNet

## 数据预处理（仅一次）

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 3d_fullres
```

## 消融实验（6 组，以 fold 0 为例）

```bash
# E0: Baseline — 标准 PlainConvUNet
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans

# E1: AnisotropicAttention only — 仅条纹伪影抑制
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerAnisoStent_AnisoAttnOnly -p nnUNetPlans

# E2: TubularEnhancement only — 仅管状结构细化
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerAnisoStent_TubEnhOnly -p nnUNetPlans

# E3: ProjCons only — 仅投影一致性损失
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerAnisoStent_ProjConsOnly -p nnUNetPlans

# E4: AnisoAttn + TubEnh — 注意力 + 管状增强
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerAnisoStent -p nnUNetPlans

# E5: Full pipeline — 注意力 + 管状增强 + 投影一致性
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerAnisoStent_ProjCons -p nnUNetPlans
```

## 5 折交叉验证完整命令

```bash
for fold in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres $fold -tr nnUNetTrainer -p nnUNetPlans
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres $fold -tr nnUNetTrainerAnisoStent_AnisoAttnOnly -p nnUNetPlans
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres $fold -tr nnUNetTrainerAnisoStent_TubEnhOnly -p nnUNetPlans
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres $fold -tr nnUNetTrainerAnisoStent_ProjConsOnly -p nnUNetPlans
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres $fold -tr nnUNetTrainerAnisoStent -p nnUNetPlans
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_ID 3d_fullres $fold -tr nnUNetTrainerAnisoStent_ProjCons -p nnUNetPlans
done
```

## 消融矩阵

| 实验 | Trainer | AnisoAttn | TubEnh | ProjCons |
|:---:|---------|:---:|:---:|:---:|
| E0 | `nnUNetTrainer` | | | |
| E1 | `nnUNetTrainerAnisoStent_AnisoAttnOnly` | ✓ | | |
| E2 | `nnUNetTrainerAnisoStent_TubEnhOnly` | | ✓ | |
| E3 | `nnUNetTrainerAnisoStent_ProjConsOnly` | | | ✓ |
| E4 | `nnUNetTrainerAnisoStent` | ✓ | ✓ | |
| E5 | `nnUNetTrainerAnisoStent_ProjCons` | ✓ | ✓ | ✓ |
