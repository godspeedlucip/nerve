# AnisoStentUNet 论文级网络架构图 — AI 生成 Prompt

## 架构说明（用于理解上下文）

```
输入 3D Volume (B, C, D, H, W)
        │
        ▼
┌─────────────────────────┐
│  AnisotropicAttention   │  ← 方向感知注意力，抑制条纹伪影
│  (Axial + Sag + Cor)    │
└─────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│              PlainConvUNet Backbone           │
│                                               │
│  Encoder (下采样)              Decoder (上采样) │
│  ├─ Stage 0: 32ch    ──skip──►  Stage 5      │
│  ├─ Stage 1: 64ch    ──skip──►  Stage 4      │
│  ├─ Stage 2: 128ch   ──skip──►  Stage 3      │
│  ├─ Stage 3: 256ch   ──skip──►  Stage 2      │
│  ├─ Stage 4: 320ch   ──skip──►  Stage 1      │
│  └─ Stage 5: 320ch   ──skip──►  Stage 0      │
│       (Bottleneck)                            │
│                                               │
│  + Deep Supervision (多尺度输出)                │
└──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  TubularEnhancement     │  ← 多尺度空洞卷积细化管状结构
│  (dilation=1,2,3,5)     │
└─────────────────────────┘
        │
        ▼
输出 Segmentation Map (B, num_classes, D, H, W)
```

---

## Prompt 版本 1：DALL-E / GPT-4o（推荐）

```
Generate a scientific paper-style neural network architecture diagram for a 3D medical image segmentation model called "AnisoStentUNet". The diagram should be clean, publication-quality, suitable for a Nature Methods or MICCAI paper.

Layout: Top-to-bottom pipeline flow.

Components (from top to bottom):

1. **Input** — A 3D CT volume block labeled "Input Volume (D×H×W)", rendered as a semi-transparent blue cube with grid lines.

2. **AnisotropicAttention Module** — A rectangular block with three parallel sub-blocks labeled "Axial Conv (1×3×3)", "Sagittal Conv (3×1×3)", "Coronal Conv (3×3×1)". These three merge into a "Channel Attention (SE Block)" → "Sigmoid Gate". Show a curved residual skip connection going around the entire block. Label this section "Anisotropic Attention (Streak Artifact Suppression)". Use orange/warm tones.

3. **PlainConvUNet Backbone** — A classic U-Net shape with:
   - Left arm (Encoder): 6 blocks stacked vertically, each block labeled with "Conv-InstanceNorm-LeakyReLU ×N" and feature map dimensions [32, 64, 128, 256, 320, 320]. Use progressively darker blue.
   - Down-sampling arrows (↓2) between encoder blocks.
   - Right arm (Decoder): 5 blocks stacked vertically, mirroring encoder. Use progressively lighter blue.
   - Up-sampling arrows (↑2) between decoder blocks.
   - Horizontal gray dashed lines connecting encoder→decoder at each level, labeled "Skip Connection".
   - Bottom: a block labeled "Bottleneck (320ch)".
   - Multiple output branches from decoder labeled "Deep Supervision (DS1, DS2, ...)".

4. **TubularEnhancement Module** — A rectangular block showing 4 parallel dilated convolution branches: "Dilated Conv (d=1)", "Dilated Conv (d=2)", "Dilated Conv (d=3)", "Dilated Conv (d=5)". These merge into a "1×1×1 Fusion" block. Show a curved residual skip connection. Label this section "TubularEnhancement (Stent/Vessel Refinement)". Use green tones.

5. **Output** — A segmentation map block labeled "Output Segmentation (N classes)", colored as a multi-colored 3D volume.

Style requirements:
- White/light gray background, clean lines, professional academic style
- Color palette: muted blues (#3A6B9B, #5B8DB8, #7EB0D4) for backbone, warm orange (#D4783B) for attention, green (#4B8B6B) for enhancement
- Sans-serif font (Arial/Helvetica style) for all labels
- Thin connecting arrows (↓) between modules, slightly thicker for data flow
- Dashed lines for skip connections and residual connections
- Add a small legend in the bottom-right corner
- No cartoon effects, no 3D shading on blocks (flat design with subtle shadows)
- Surrounding annotations in gray italic text explaining key design choices
- Aspect ratio approximately 3:4 (width:height)
```

---

## Prompt 版本 2：精简短版（适合快速迭代）

```
A clean academic paper figure showing a U-Net neural network architecture for 3D medical image segmentation. The figure shows a vertical pipeline: Input 3D CT volume → AnisotropicAttention (3 directional conv branches with attention gate) → U-Net encoder-decoder backbone with skip connections → TubularEnhancement (4 dilated conv branches for vessel refinement) → Output segmentation map. Flat design, white background, blue color scheme for the U-Net, orange for attention module, green for enhancement module. Sans-serif labels. Professional scientific diagram style like Nature Methods figures. No 3D shading, clean vector aesthetic.
```

---

## Prompt 版本 3：Midjourney（带参数）

```
A scientific paper figure illustrating a neural network architecture for 3D medical image segmentation. The diagram shows a top-to-bottom pipeline: a 3D input volume, an anisotropic attention block with three directional branches merging into an attention gate, a U-Net encoder-decoder with skip connections across six resolution levels, a tubular enhancement block with four parallel dilated convolutions, and a segmentation output. Clean academic style, white background, muted colors (navy blue, slate blue, warm orange accent, forest green accent), flat vector design with subtle shadows, sans-serif text labels, professional publication quality. --ar 3:4 --style raw --no cartoon, photo, realistic, gradient, 3d render
```

---

## 使用建议

| 工具 | 推荐版本 | 技巧 |
|------|---------|------|
| GPT-4o / DALL-E 3 | 版本 1（长版） | 可以分段对话，先让它理解结构，再要求绘图 |
| Midjourney | 版本 3 | 加 `--style raw` 减少艺术化；可能需要多次 reroll |
| Stable Diffusion | 版本 2（简版）| 配合 controlnet 效果更好 |

**强烈建议**: 用 GPT-4o + DALL-E 3 先跑版本 1，如果不满意结构精确度，再考虑用 Python (matplotlib + drawconv) 或 TikZ 做精确矢量图。
