# Interior

这是脑血管/支架仿真、医学影像预处理和 3D U-Net 实验代码的独立 Git 仓库。

## 当前主线

- `data/corr_data/gen_stent_and_vessel.py`：当前推荐的 DTE 3D 支架/血管生成入口。
- `data/corr_data/README_DTE_GENERATOR.md`：DTE 生成器的参数、输入、输出和 marker 说明。
- `data/corr_data/utils/`：中心线、血管、支架、投影和 marker 辅助模块。
- `data/gen_data_utils/`：NIfTI/TIFF 预处理、ROI、截断和 nnU-Net 格式整理工具。
- `unet/`：3D U-Net 模型、训练、测试和数据加载代码。
- `train_gpu.py`：GPU 占用/计算辅助工具。

## 运行提示

代码仍保留原工作区的相对导入和部分绝对数据路径。运行前请先确认对应数据、helper、PLY 模板和 Python/CUDA 依赖已经存在；推荐在脚本所在目录执行旧脚本，例如：

```bash
cd data/corr_data
python gen_stent_and_vessel.py --help

cd ../gen_data_utils/nib
python get_resample_roi.py --help

cd ../../../unet
python train.py --help
```

上面的命令只检查入口参数，不会替代完整的数据生成或训练流程。

## 数据与版本控制

`data/` 和 `real_data/` 同时承担代码目录和本地数据工作区的角色。原始影像、投影结果、helper/resample/truncate 产物、PLY 模板、模型权重和运行输出均保留在本机，但不纳入 Git；具体规则见 `.gitignore`。克隆仓库后，需要按实验环境自行准备这些外部数据。

## 历史内容

明确的旧版本、复制版本和早期实验位于 `data/legacy/`；探索性 notebook 位于 `notebooks/legacy/`。它们用于追溯实验，不属于当前推荐入口。
