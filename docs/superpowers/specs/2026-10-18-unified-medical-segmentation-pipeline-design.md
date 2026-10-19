# Unified Medical Segmentation Enhancement Pipeline Design

## Goal

将 DTM/GSL、投影一致性损失、各向异性条纹伪影抑制和管状结构增强整合为可组合、可测试、配置驱动的训练管线，同时保持现有 nnU-Net trainer 名称、训练命令、deep supervision、双网络训练和已有 checkpoint 的兼容性。

## Scope and non-goals

本次重构只处理自定义增强能力的组织方式，不修改 nnU-Net backbone、数据集格式、投影几何定义或已有损失的数学含义。

包含：

- 统一网络增强模块的装配入口；
- 统一分割、ProjCons、DTM/GSL 的 loss 组合和组件日志；
- 统一 DTM/`valid_mask` 的可选 batch 契约；
- 将旧 trainer 变体改为预设配置的兼容入口；
- 为组合、deep supervision、忽略标签、无辅助数据和 A/B trainer 增加单元测试。

不包含：

- 改变模型结构或重新设计 AnisotropicAttention/TubularEnhancement 内部算法；
- 改变 ProjectionConsistencyLoss 的角度、旋转和积分语义；
- 改变 DTM 预处理文件格式或重新计算已有 DTM；
- 删除旧 trainer 类名或迁移既有 checkpoint。

## Current problems

当前功能分散在 `network_modules`、`training/loss`、DTM 专用 dataloader/preprocessor 和多个 trainer 文件中。Aniso/ProjCons/DTM trainer 各自复制初始化、训练步、验证步和日志逻辑，功能组合主要通过类继承和类名表达，导致：

1. 新增一种组合需要新增 trainer 类；
2. 组件 loss 计算和日志拆解与总 loss 逻辑重复；
3. DTM 辅助数据只在 DTM trainer 分支中显式处理；
4. deep supervision、ignore label、regions 等分支容易在不同 trainer 中出现行为漂移。

## Proposed architecture

### 1. Configuration as the composition boundary

新增不可变配置对象 `EnhancementConfig`，集中描述增强管线的开关和超参数。配置至少包含：

- `enable_anisotropic_attention: bool`，默认 `False`；
- `enable_tubular_enhancement: bool`，默认 `False`；
- `projection_angles_deg: tuple[float, ...]`，空元组表示关闭；
- `projection_consistency_weight: float`，必须大于等于 0；
- `dtm_mode: Literal["disabled", "monitor", "active"]`，默认 `"disabled"`；
- `dtm_weight: float`，必须大于等于 0；
- `dtm_activate_after_epoch: int`，必须大于等于 0；
- `dtm_active_weight: float`，默认保持现有 GSL active 模式的权重语义。

配置对象负责校验值和表达组合，不负责创建网络或损失。旧 trainer 通过类级预设返回配置；未来可在不新增 trainer 类的情况下增加配置组合。

### 2. Model enhancement pipeline

新增 `EnhancementPipeline`，职责是包装 backbone 并按配置执行：

```python
EnhancementPipeline(
    backbone: nn.Module,
    in_channels: int,
    num_classes: int,
    config: EnhancementConfig,
) -> nn.Module
```

行为：

- 启用 AnisotropicAttention 时，在 backbone 前执行；
- 启用 TubularEnhancement 时，只对最终最高分辨率输出执行；
- deep supervision 的辅助输出保持原样；
- 两个模块关闭时，包装器应等价于 backbone；
- 暴露 `decoder` 属性，保持 nnU-Net 的 `set_deep_supervision_enabled` 和相关训练流程兼容；
- 保持 state dict 前缀稳定：兼容现有 `aniso_attn`、`backbone`、`tub_enh` 命名，避免旧 checkpoint 无法加载。

现有 `AnisoStentUNet` 保留为兼容类，内部委托到该 pipeline 或与其共享实现。

### 3. Composite loss pipeline

新增 `CompositeSegmentationLoss`，统一封装基础分割 loss、ProjCons 和 DTM/GSL：

```python
CompositeSegmentationLoss(
    segmentation_loss: nn.Module,
    config: EnhancementConfig,
    label_manager,
    projection_loss: Optional[nn.Module] = None,
    dtm_loss: Optional[nn.Module] = None,
) -> nn.Module
```

统一入口：

```python
forward(
    output,
    target,
    auxiliary: Optional[AuxiliaryTargets] = None,
    epoch: Optional[int] = None,
) -> LossBreakdown
```

其中 `LossBreakdown` 至少提供 `total`、`segmentation`、`projection`、`dtm` 四个标量，并支持转换为现有 trainer 需要的标量 loss。为兼容外部 dual training，提供兼容调用路径，使 `trainer.loss(output, target)` 仍然返回可反向传播的 scalar；没有 DTM 辅助数据时 DTM 项为零且不报错。

统一处理：

- tensor/list 两种 deep supervision 输出；
- regions 与普通 class-index target；
- ignore label mask；
- ProjCons 只使用最终输出；
- DTM/GSL 的 monitor 与 active 两种模式；
- 各组件的独立日志值，避免 trainer 手动重复计算 ProjCons。

现有 `CombinedSegProjLoss` 和 DTM loss 类保留兼容导出；兼容层可以将旧接口适配到新组件，但不改变旧调用签名。

### 4. Auxiliary target contract

新增轻量 `AuxiliaryTargets` 数据对象，统一表示可选训练辅助数据：

```python
AuxiliaryTargets(
    dtm: Optional[Tensor | list[Tensor]],
    valid_mask: Optional[Tensor | list[Tensor]],
)
```

提供：

- `to(device)`：递归移动 tensor/list；
- `final_resolution()`：取得与最终输出匹配的 DTM 和 mask；
- `empty()`：表示普通训练 batch；
- 形状和 batch size 的明确校验。

现有 DTM dataloader 继续产出 `dtm` 和 `valid_mask` 字段，由 trainer adapter 转成该对象；普通 dataloader 不需要改数据格式。

### 5. Trainer assembly and compatibility presets

新增通用装配组件，集中负责：

- 构建 backbone 和 `EnhancementPipeline`；
- 构建基础 segmentation loss 与 deep supervision wrapper；
- 根据配置创建 Composite loss；
- 统一 train/validation step 的辅助数据移动、loss 计算和组件日志；
- 保持现有 Dice/TP-FP-FN 统计和 optimizer 行为。

旧类映射固定如下：

| 旧 trainer | 兼容预设 |
|---|---|
| `nnUNetTrainerAnisoStent` | Aniso + Tubular，ProjCons disabled，DTM disabled |
| `nnUNetTrainerAnisoStent_ProjCons` | Aniso + Tubular + ProjCons |
| `nnUNetTrainerDTM` | 普通网络 + DTM monitor |
| `nnUNetTrainerDTM_GSL` | 普通网络 + DTM active |
| `*_A` / `*_B` | 对应预设的 A/B 命名兼容类 |

现有消融 trainer 继续可用，但改为复用同一 pipeline/Composite loss 装配逻辑，而不是在文件内定义匿名 wrapper 和重复 loss 构造代码。

### 6. Backward compatibility

- 不删除旧模块导出和旧 trainer 类名；
- 不修改已有 trainer 的默认超参数和默认启用能力；
- 保持 checkpoint 中已有模块名称，必要时使用兼容属性或 state-dict 映射；
- `CombinedSegProjLoss.forward(output, target)` 继续可调用；
- dual containment 训练脚本继续能够调用 `trainer.loss(output, target)`；
- 旧 DTM 文件命名 `_dtm_class*.b2nd` 和预处理入口保持不变。

## Error handling and validation

- 配置在创建时校验负权重、空角度与非空权重的组合、非法 DTM mode 和非法 epoch；
- Composite loss 在启用 DTM 时缺少辅助数据：训练中明确返回零 DTM 项并记录一次 warning，不能静默把 DTM 当作 active；
- Composite loss 在启用 ProjCons 时缺少 projection loss：初始化阶段直接抛出带上下文的 `ValueError`；
- target、output、DTM 的 batch 或空间维度不一致时抛出 `ValueError`，错误信息包含组件名和实际形状；
- 无辅助项时不创建不必要的投影缓存或 DTM 计算。

## Testing strategy

测试集中放在 `nnunetv2/tests/unit/`，不依赖真实数据集或 GPU：

1. 配置默认值、非法值和组合校验；
2. pipeline 开关组合、deep supervision 输出结构、decoder 委托和 state-dict 命名；
3. Composite loss 的单项/多项数值组成、monitor/active DTM、ignore label、regions、无辅助数据；
4. AuxiliaryTargets 的 list/tensor 设备移动和形状校验；
5. 旧 trainer 预设映射和旧兼容接口；
6. 最小 CPU forward/backward smoke test，证明 Aniso、Tubular、ProjCons 和 DTM 组合可同时反向传播。

验收标准：

- 旧 trainer 名称仍可导入，已有默认能力不变；
- 新增组合不需要复制 trainer 的 train/validation 循环；
- 单元测试覆盖全部配置分支并在 CPU 上通过；
- 现有用户改动 `.gitignore` 和 `TRAINING_INFERENCE_GUIDE.md` 不被覆盖。

## Migration order

1. 先引入配置、辅助数据对象和 pipeline，保留旧类；
2. 引入 Composite loss，并给旧 loss 入口加适配；
3. 抽取 trainer 装配和通用 step；
4. 将旧 trainer 改为预设兼容入口；
5. 更新文档和验证脚本，最后运行单元测试与已有模块验证脚本。
