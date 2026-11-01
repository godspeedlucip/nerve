# Interior Code Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn `/data1/liuyang/stent/interior` into an independent Git repository that versions the useful code and documentation while leaving large data workspaces untracked.

**Architecture:** Keep the existing `data/` and `unet/` paths so the documented entry points continue to resolve their current imports and absolute data paths. Move only clearly historical or temporary code into `data/legacy/` and `notebooks/legacy/`; add root documentation and precise ignore rules; initialize and commit only the curated source tree.

**Tech Stack:** Git, Python 3 syntax compilation, Markdown documentation, existing Python/PyTorch/TIGRE/nibabel/Open3D scripts.

**Spec:** `docs/superpowers/specs/2026-10-29-interior-code-repository-design.md`

## Global Constraints

- Git repository root is `/data1/liuyang/stent/interior`.
- The upper `/data1/liuyang/stent` repository must remain unchanged.
- Do not move or version medical images, projections, helper outputs, PLY assets, model weights, or other large binary data.
- Preserve the documented DTE, preprocessing, and U-Net mainline paths.
- Historical files are moved rather than deleted; archived files are not guaranteed to run from their old paths.
- Every Python file included in the initial commit must pass syntax compilation.
- Do not install dependencies or run GPU/data-consuming workflows.

## Review Focus

- Ignored data directories: `git add` must not stage the multi-gigabyte `data/src_data/`, `data/corr_data/data/`, `real_data/`, or helper outputs.
- Mainline entry points: `data/corr_data/gen_stent_and_vessel.py`, `data/gen_data_utils/`, `unet/`, and `train_gpu.py` must remain present and staged.
- Legacy relocation: duplicate `to_down` and `before` trees must exist under `data/legacy/`, with no accidental deletion.
- Syntax integrity: all staged `*.py` files must pass `python -m compileall`.
- Repository boundary: the new `.git` must exist only at `interior/.git`; no parent repository files or indexes may be changed.

### Task 1: Move historical and temporary material without touching data outputs

**Files:**
- Move: `data/corr_data/to_down/` -> `data/legacy/corr_data/to_down/`
- Move: `data/gen_data_utils/nib/before/` -> `data/legacy/gen_data_utils/nib/before/`
- Move: `data/corr_data/gen_stent_and_vessel_copy.py`
- Move: `data/corr_data/gen_stent_and_vessel_copy_diffR.py`
- Move: `data/corr_data/gen_stent_and_vessel_with_no_proj.py`
- Move: `data/corr_data/gen_correspond_stent.py`
- Move: `data/corr_data/gen_correspond_vessel.py`
- Move: `data/corr_data/gen_proj_stent.py`
- Move: `data/corr_data/gen_truncate_data.py`
- Move: `data/corr_data/gen_truncate_data_copy.py`
- Move: `data/corr_data/gen_truncate_data_dte.py`
- Move: `data/corr_data/gen_truncate_data_yes_fdk.py`
- Move: `data/corr_data/get_2d_from_3d.py`
- Move: `data/corr_data/nnUnet_format_train_data.py`
- Move: `data/corr_data/nnUNet_format_test_data.py`
- Move: `data/temp.ipynb` -> `notebooks/legacy/data-temp.ipynb`
- Move: `data/occupy_gpu.ipynb` -> `notebooks/legacy/data-occupy-gpu.ipynb`
- Move: `main.ipynb` -> `notebooks/legacy/interior-main.ipynb`

**Interfaces:**
- Preserves: `data/corr_data/gen_stent_and_vessel.py`, `data/corr_data/utils/`, `data/corr_data/gen_stent/`, `data/gen_data_utils/` except `nib/before/`, `unet/`, and `train_gpu.py`.
- Produces: a visible `data/legacy/` and `notebooks/legacy/` boundary for historical material.

- [ ] **Step 1: Confirm all source and destination paths before moving.**

  Run:

  ```bash
  test -f data/corr_data/gen_stent_and_vessel.py
  test -d data/corr_data/to_down
  test -d data/gen_data_utils/nib/before
  test -f main.ipynb
  ```

  Expected: every command exits successfully.

- [ ] **Step 2: Move only the listed historical and temporary paths.**

  Keep `data/corr_data/data/`, `data/src_data/`, `real_data/`, and `data/corr_data/gen_stent/ply/` in their existing locations.

- [ ] **Step 3: Verify canonical paths and archive contents.**

  Run:

  ```bash
  test -f data/corr_data/gen_stent_and_vessel.py
  test -d data/corr_data/utils
  test -d data/corr_data/gen_stent
  test -d data/gen_data_utils
  test -d data/legacy/corr_data/to_down
  test -d data/legacy/gen_data_utils/nib/before
  test ! -e data/corr_data/to_down
  test ! -e data/gen_data_utils/nib/before
  ```

  Expected: canonical paths exist, archive paths exist, and the old archive paths are absent.

### Task 2: Add repository documentation and ignore policy

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `data/README.md`
- Create or modify: `data/legacy/README.md`
- Create or modify: `notebooks/README.md`

**Interfaces:**
- Consumes: the current DTE generator documentation at `data/corr_data/README_DTE_GENERATOR.md` and the retained source tree.
- Produces: a documented project map, current entry points, legacy boundary, data policy, and commands that do not assume Git tracks data.

- [ ] **Step 1: Write `README.md` with the repository purpose and map.**

  It must identify `data/corr_data/gen_stent_and_vessel.py` as the current DTE generator, link to its existing detailed README, describe `data/gen_data_utils/`, identify `unet/` and `train_gpu.py`, and state that raw/generated data stays in the workspace and is ignored.

- [ ] **Step 2: Write `.gitignore` for code-repository boundaries.**

  Include Python/editor caches, logs, outputs, checkpoints/weights, `real_data/`, `data/src_data/`, `data/corr_data/data/`, helper/resample/truncate result directories, `*.nii*`, `*.npy`, `*.npz`, `*.tif*`, `*.pickle`, `*.pkl`, `*.ply`, archives, and other large binary artifacts. Do not ignore retained `*.py`, `*.sh`, `*.md`, or small JSON/text configuration files.

- [ ] **Step 3: Add focused README files for data and archive conventions.**

  Document that `data/` contains scripts plus an external working dataset, that `data/legacy/` is historical and not the active path, and that notebooks are exploratory records rather than tested entry points.

- [ ] **Step 4: Verify ignore rules and repository documentation links.**

  Run:

  ```bash
  git check-ignore -v data/src_data real_data data/corr_data/data data/corr_data/gen_stent/ply/train/stent1.ply
  test -f README.md
  test -f .gitignore
  rg -n "gen_stent_and_vessel|gen_data_utils|unet|legacy|not tracked|不纳入" README.md data/README.md data/legacy/README.md notebooks/README.md
  ```

  Expected: each large-data path is ignored and each README contains the expected project map terms.

### Task 3: Perform source-only verification

**Files:**
- Verify: all staged candidate `*.py` files under `data/`, `unet/`, and the repository root.

**Interfaces:**
- Consumes: the post-move source tree and ignore policy.
- Produces: evidence that the curated Python tree is syntactically valid without importing optional dependencies or executing data/GPU workflows.

- [ ] **Step 1: Enumerate the source-only candidate set.**

  Run:

  ```bash
  find data unet -type f -name '*.py' -not -path '*/__pycache__/*' -print | sort
  ```

- [ ] **Step 2: Compile every candidate Python file.**

  Run:

  ```bash
  python -m compileall -q data unet train_gpu.py
  ```

  Expected: exit code 0. Do not run imports, training, projection, or dataset generation.

- [ ] **Step 3: Remove or ignore generated bytecode and verify no source was modified.**

  Run:

  ```bash
  find data unet -type d -name __pycache__ -prune -print
  git diff -- data unet train_gpu.py
  ```

  Expected: bytecode directories are ignored; the only intentional changes are moves and README/documentation changes.

### Task 4: Initialize the independent Git repository and create the first commit

**Files:**
- Create: `.git/` at `/data1/liuyang/stent/interior`
- Stage: `README.md`, `.gitignore`, `docs/`, `data/` source/docs/legacy files, `notebooks/`, `unet/`, and `train_gpu.py` subject to ignore rules.

**Interfaces:**
- Consumes: verified source tree and `.gitignore`.
- Produces: a standalone `main` repository with a code-only initial commit and no remote configuration.

- [ ] **Step 1: Confirm there is no nested repository and initialize `main`.**

  Run:

  ```bash
  test ! -e .git
  git init -b main
  ```

  Expected: a new `.git` exists only under `interior`.

- [ ] **Step 2: Stage the curated tree explicitly.**

  Run:

  ```bash
  git add .gitignore README.md docs data unet notebooks train_gpu.py
  git status --short
  ```

  Expected: source/docs/legacy paths are staged; raw/generated data, PLY files, caches, and outputs are absent.

- [ ] **Step 3: Inspect staged size and paths before committing.**

  Run:

  ```bash
  git diff --cached --stat
  git diff --cached --name-only | rg '(^|/)(src_data|real_data|corr_data/data|__pycache__|.*\\.(nii|nii\\.gz|npy|npz|tif|tiff|ply|pth|pt|ckpt))($|/)'
  ```

  Expected: the second command prints nothing; staged content remains code/documentation sized.

- [ ] **Step 4: Create the initial commit.**

  Run:

  ```bash
  git commit -m "chore: organize interior code repository"
  ```

  Expected: one root commit is created on `main`.

- [ ] **Step 5: Verify the new repository and the outer repository boundary.**

  Run:

  ```bash
  git status --short --branch
  git log -1 --oneline
  git rev-parse --show-toplevel
  git -C /data1/liuyang/stent status --short -- interior
  ```

  Expected: the new repo is clean on `main`, its root is `/data1/liuyang/stent/interior`, and the outer repository has no new changes caused by this work.
