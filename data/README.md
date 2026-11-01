# `data/`

这里同时放置数据处理代码和本地数据工作区。

## 代码入口

- `corr_data/gen_stent_and_vessel.py`：当前 DTE 支架/血管生成主线。
- `corr_data/utils/`、`corr_data/gen_stent/`：主线所需的辅助模块和支架生成器。
- `gen_data_utils/nib/`：NIfTI 数据的重采样、ROI、截断和格式转换。
- `gen_data_utils/tif/`：TIFF 数据的 ROI 和截断处理。
- `gen_data_utils/other/`：距离图、骨架和检查工具。

## 数据边界

`src_data/`、`corr_data/data/` 以及各数据集下的 helper、resample、truncate 目录是本地工作数据，不进入 Git。脚本中的历史绝对路径也默认这些目录已经存在；迁移到其他机器时请通过命令行参数或按实验配置调整路径。

## 历史代码

`legacy/` 保存旧版本、复制版本和早期实验。当前流程优先使用上面列出的主线入口。
