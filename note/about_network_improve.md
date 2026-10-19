# nnUNet 网络改进总结

## 架构总览

```
输入体积 (B, C, D, H, W)
   ↓
AnisotropicAttention  ← 预处理：去条纹伪影（3路各向异性卷积 + SE融合 + 门控）
   ↓
标准 U-Net (encoder → bottleneck → decoder)  ← backbone 不变
   ↓
TubularEnhancement   ← 后处理：多尺度膨胀卷积精炼细管状结构（仅作用于最终输出）
   ↓
分割输出 mask
```

两个模块分别在 U-Net 的**两端**，不侵入 backbone 内部。这样解耦后 deep supervision 不受影响——TubEnh 只作用于 `output[0]`，辅助输出不经过。

训练时额外路径（仅 ProjCons 变体）：

```
分割输出 → 可微前向投影 → 投影图
                              │
原始投影图 ────────────→ [MSE Loss] ←─┘
                              │
                      加到 DC+CE Loss
```

---

## 模块 1: AnisotropicAttention

**文件**: `network_modules/anisotropic_attention.py`
**作用**: 抑制有限角度反投影产生的方向性条纹伪影
**参数量**: ~42（in_channels=1 时）

```
输入 → 3路各向异性卷积分支 → concat → SE通道注意力 → Sigmoid门控 → 残差输出
       ├─ Conv3d(1×3×3)  轴向 (XY平面)
       ├─ Conv3d(3×1×3)  矢状 (XZ平面)
       └─ Conv3d(3×3×1)  冠状 (YZ平面)
```

- 三条分支分别沿三个解剖轴提取方向性特征
- SE 模块自适应融合三路信息
- 残差设计：学不到东西时退化为恒等变换

## 模块 2: TubularEnhancement

**文件**: `network_modules/tubular_enhancement.py`
**作用**: 恢复支架、血管等细薄管状结构，防止 U-Net 下采样丢失
**参数量**: ~1,832（num_classes=2 时）

```
输入 → 4路膨胀3D卷积 → concat → 1×1融合 + InstanceNorm + LeakyReLU → 残差输出
       ├─ dil=1 (小管)
       ├─ dil=2
       ├─ dil=3
       └─ dil=5 (大管)
```

- 多尺度膨胀率覆盖不同粗细的管状结构
- 只在 U-Net 最终输出上作用，不影响中间层

## 模块 3: ProjectionConsistencyLoss

**文件**: `network_modules/projection_consistency.py`
**作用**: 利用原始投影图做自监督约束，确保分割的 3D 几何形状与多视角投影一致

核心组件：
- `build_rotation_matrix_3d()` — 构建绕 Z 轴的旋转矩阵
- `differentiable_forward_projection()` — 通过 `F.affine_grid` + `F.grid_sample` 旋转体积后沿投影轴积分，全程可微
- `ProjectionConsistencyLoss` — MSE(预测投影, 真值投影)
- `CombinedSegProjLoss` — 包装 DC+CE + ProjCons 为单一 `nn.Module`，签名兼容 `trainer.loss(output, target)`

## 包装网络: AnisoStentUNet

**文件**: `network_modules/aniso_stent_unet.py`

```python
class AnisoStentUNet(nn.Module):
    def forward(self, x):
        x = self.aniso_attn(x)        # 1. 前置：去条纹
        out = self.backbone(x)        # 2. 标准 U-Net
        if isinstance(out, list):     # 3. deep supervision 兼容
            out[0] = self.tub_enh(out[0])
            return out
        return self.tub_enh(out)
```

## 自定义 Trainer

**文件**: `training/nnUNetTrainer/nnUNetTrainer_AnisoStent.py`

| Trainer | 网络 | Loss | 用途 |
|---|---|---|---|
| `nnUNetTrainerAnisoStent` | AnisoStentUNet | DC+CE | 基础版 |
| `nnUNetTrainerAnisoStent_ProjCons` | AnisoStentUNet | DC+CE+ProjCons | 带投影一致性 |
| `nnUNetTrainerAnisoStent_A/B` | AnisoStentUNet | DC+CE | Dual training A/B 桩 |
| `nnUNetTrainerAnisoStent_ProjCons_A/B` | AnisoStentUNet | DC+CE+ProjCons | Dual+ProjCons 桩 |

A/B 桩类仅类名不同，使 nnUNet 输出 checkpoint 到不同文件夹，兼容 `run_training_dual_containment.py`。

## 关键设计原则

1. **残差贯穿** — AnisoAttn、TubEnh 均用残差连接，确保退化安全
2. **不侵入外部包** — AnisoStentUNet 是 wrapper，不修改 `dynamic_network_architectures`
3. **两端解耦** — 模块在 U-Net 前后两端，不影响 backbone 内部和 deep supervision
4. **可开关** — 多个 Trainer 变体独立，方便 ablation 实验
5. **完全兼容 dual training** — CombinedSegProjLoss 的 `forward(output, target)` 签名与 `trainer.loss()` 一致

## 文件清单

```
nnunetv2/nnunetv2/
├── network_modules/
│   ├── __init__.py
│   ├── anisotropic_attention.py
│   ├── tubular_enhancement.py
│   ├── projection_consistency.py
│   └── aniso_stent_unet.py
├── training/
│   └── nnUNetTrainer/
│       └── nnUNetTrainer_AnisoStent.py
```
