# Interior 代码独立仓库整理设计

## Intent

在 `/data1/liuyang/stent/interior` 创建独立 Git 仓库，用于版本管理支架/血管数据生成、预处理和 3D U-Net 训练代码。仓库应能说明当前主线入口，同时不把约 893 GB 的医学影像、投影结果、helper 缓存和模型输出纳入版本控制。

## Goals

- 在 `interior` 根目录初始化独立 Git 仓库，不嵌套或依赖上层仓库。
- 保留现有脚本依赖的主要路径，整理后仍能从原工作区运行。
- 明确当前推荐入口、数据目录、运行前提和历史代码位置。
- 将明显的重复版本、早期实验和临时 notebook 归档，而不是删除。
- 用 `.gitignore` 排除原始数据、生成数据、模型权重、缓存和运行输出。
- 完成源码语法检查、忽略规则检查和首次 Git 提交。

## Non-goals

- 不迁移、重命名或重算现有医学影像和生成结果。
- 不在本次工作中重构所有脚本的绝对路径、参数接口或算法逻辑。
- 不安装新的 Python/CUDA 依赖，也不承诺在当前环境之外完整运行训练或 TIGRE 投影流程。
- 不删除历史代码、原始数据或已有实验结果。

## Anti-goals

- 不把 `data/`、`real_data/` 或模型输出整体 `git add` 进仓库。
- 不将历史实验版本误标成当前生产入口。
- 不为了“看起来整齐”而移动会破坏当前绝对路径和相对导入的核心源码。

## Constraints and current state

- `interior` 当前没有 `.git`，但 `/data1/liuyang/stent` 是另一个上层仓库；本次在 `interior` 建立独立仓库。
- `data/` 约 893 GB，主要空间来自 `src_data/`、`corr_data/data/` 和各数据集 helper 目录。
- 当前 DTE 主线是 `data/corr_data/gen_stent_and_vessel.py`，配套使用 `data/corr_data/utils/`、`data/corr_data/gen_stent/` 和工作区中的 helper/data 目录。
- 数据预处理代码主要位于 `data/gen_data_utils/`；3D U-Net 代码位于 `unet/`，GPU 占用工具位于 `train_gpu.py`。
- 代码中存在绝对路径、当前工作目录假设以及多个历史复制版本，因此本次以仓库整理为主，不做全面可移植化改造。

## Approaches considered

1. **在 `interior` 原位建立代码仓库（采用）**：保留当前主线目录和运行路径，新增根 README、忽略规则和文档，把历史/临时内容归档。兼容性最好，改动范围可控。
2. **抽出全新的 `src/` 项目布局**：代码结构更现代，但需要同步修改大量导入、绝对路径和脚本参数，容易影响已有实验，超出本次整理范围。
3. **只初始化 Git，不移动任何文件**：风险最低，但无法解决当前目录中重复版本和临时文件难以辨认的问题。

采用方案 1，因为用户需要独立仓库，同时现有脚本仍依赖当前工作区布局。

## Planned organization

- 新增 `README.md`：说明项目用途、推荐入口、目录结构、数据不入库原则和典型运行方式。
- 新增 `.gitignore`：忽略 `data/src_data/`、`data/corr_data/data/`、helper/resample/truncate 产物、`real_data/`、模型权重、日志、缓存、压缩包和大体积二进制数据；源码、README、配置和小型工具仍可跟踪。
- 保留当前主线源码路径：
  - `data/corr_data/gen_stent_and_vessel.py`
  - `data/corr_data/utils/`
  - `data/corr_data/gen_stent/`
  - `data/gen_data_utils/`
  - `unet/`、`train_gpu.py`
- 将 `data/corr_data/to_down/`、`data/gen_data_utils/nib/before/`、`corr_data` 中明确标为 copy/diffR/旧实验的脚本，以及临时 notebook 放入 `data/legacy/` 或 `notebooks/legacy/`；移动前保留其内容，并在 README 中标明不属于当前主线。
- 不移动数据目录本身；支架 PLY 等大体积运行资产保留在原工作区并由 README 说明获取/准备方式。
- 在仓库内保留本设计文档，便于后续继续整理。

## Verification and handoff

- 检查整理后的文件树和 Git 状态。
- 对可纳入仓库的 Python 文件运行 `python -m compileall`，不执行需要 GPU/数据集的任务。
- 用 `git check-ignore` 验证大型数据和输出被忽略、关键源码未被忽略。
- 初始化 `main` 分支并创建一次初始提交；不配置远程仓库，不上传数据。

## Builder decisions

- 本次不做完整依赖锁定：现有代码横跨 PyTorch、nibabel、SimpleITK、TIGRE、Open3D 等环境，先记录用途和入口，避免生成一份不可靠的统一 requirements 文件。
- 历史代码优先移动而非删除；如果某个历史目录被当前入口实际引用，则保留原路径并仅在文档中标注。
- Git 仓库根目录固定为 `/data1/liuyang/stent/interior`，数据工作区仍使用 `/data1/liuyang/stent/interior/data`。
- 上层 `/data1/liuyang/stent` 仓库保持不变；其中已跟踪的两个文件不从上层仓库移除，新仓库会独立记录它们当前的内容。
- 兼容性承诺只覆盖 README 标出的 DTE、预处理和 U-Net 主入口；归档脚本移动后不保证仍可从原路径直接执行。
- 首次提交只接收源码、文档、小型文本配置和可通过检查的 notebook；影像、投影、helper、PLY、权重和其他大二进制不进入 Git。
- 所有进入首次提交的 Python 文件都必须通过语法编译检查；无法通过检查的历史文件留在磁盘上，但不进入首次提交。
