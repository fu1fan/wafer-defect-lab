# Wafer Defect Lab

面向晶圆缺陷研究的深度学习实验仓库。核心任务是利用 wafer map 上 die 的空间分布模式进行 **wafer-level 异常/模式识别**，并为后续 region-level normality-based anomaly localization 预留接口。

---

## 项目目标

1. **Wafer-level pattern classification baseline** ← 当前阶段
   - 输入：224×224 单通道 wafer map（原始 ~52×52 → nearest resize）
   - 输出：整张 wafer 的 normal/abnormal 判别（binary），或 failure-type pattern 分类（multiclass 9 类）
   - 解释：GradCAM 响应图（仅为解释性热力图，不等同于真正的 anomaly localization）

2. **Normality-based anomaly localization**（Phase 2）
   - 方法：PaDiM / PatchCore 等基于正常样本特征分布的方法
   - 输出：pixel/region-level anomaly score map
   - 目标：定位偏离正常分布的局部空间模式

3. **混合缺陷多标签分类 & 持续学习**（Phase 3+）

> **重要区分**：wafer map 中 `2=defect die` 是已知输入信息，不是本项目要预测的目标。
> 本项目关心的是这些 die 的空间分布模式（pattern recognition）以及分布偏移定位（distribution-shift localization）。

---

## 当前状态（Phase 1）

- [x] WM-811K 数据读取 → interim artifacts → processed 224×224 HDF5
- [x] PyTorch Dataset / DataLoader（支持 filter、按 split_label 划分）
- [x] ResNet18 wafer-level classifier（binary / multiclass 可切换）
- [x] 训练 / 评估 / GradCAM 可视化完整链路
- [ ] PaDiM / PatchCore anomaly localization（Phase 2）
- [ ] MixedWM38 多标签分类（Phase 3）

---

## 数据说明

### Wafer Map 语义

| 值 | 含义 |
|----|------|
| `0` | blank / 背景 |
| `1` | normal die |
| `2` | defect die |

### WM-811K 标注

- `failure_type`：none, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch
- 本阶段仅使用 **labeled** 子集（172,950 张），按原始 `split_label` 分为 Training / Test
- Processed HDF5 shape: `[N, 1, 224, 224]`, dtype `uint8`

---

## 项目结构

```text
wafer-defect-lab/
├── configs/
│   ├── data/wm811k.yaml              # 数据处理配置
│   └── train/wm811k_resnet_baseline.yaml  # 训练配置
├── data/
│   ├── raw/                           # 原始数据（不入 Git）
│   ├── interim/                       # 中间处理产物
│   └── processed/wm811k/             # 训练用 HDF5 + index
├── outputs/                           # 训练输出、checkpoint、评估结果
├── scripts/
│   ├── prepare_data.py                # 下载 + 构建 interim
│   ├── process_data.py                # interim → processed 224×224
│   ├── train_classifier.py            # 训练入口
│   ├── eval_classifier.py             # 评估入口
│   └── visualize_cam.py               # GradCAM 可视化
└── src/waferlab/
    ├── data/                          # Dataset、DataLoader、数据处理
    ├── models/                        # WaferClassifier (ResNet backbone)
    ├── engine/                        # Trainer、Evaluator
    ├── metrics/                       # 分类指标计算
    └── visualize/                     # GradCAM
```

---

## 快速开始

### 环境

```bash
conda activate torch
pip install -r requirements.txt
```

这里的 `requirements.txt` 只包含项目基础依赖，不再写死 `torch` / `torchvision` 版本：
- 本地开发建议直接在你已有的 CUDA/conda 环境里安装合适的 PyTorch
- 远端机器则交给 `scripts-remote/deploy.py` 自动选择并安装兼容的最新组合

本地 `make` 目标会直接使用当前 shell 里的 `python3` / `pip`，不会自动切换 conda 环境，所以执行 `make train`、`make eval` 等命令前请先激活好对应环境。

本地训练默认采用 `device=auto`：
- 本机有 CUDA 时优先使用 GPU
- 没有 CUDA 时自动降级到 CPU

也可以通过环境变量切换数据与输出根目录：

```bash
export WAFERLAB_DATA_ROOT=/abs/path/to/data
export WAFERLAB_OUTPUT_ROOT=/abs/path/to/outputs
```

### 数据准备（如已有 processed artifacts 可跳过）

```bash
python scripts/prepare_data.py       # 下载 + 构建 interim
python scripts/process_data.py       # 生成 224×224 processed HDF5
```

### 训练

```bash
# Binary baseline（默认）
python scripts/train_classifier.py

# Multi-class failure-type 分类
python scripts/train_classifier.py --task-mode multiclass

# 快速验证（1 epoch, 256 samples）
python scripts/train_classifier.py --smoke-test
```

也可以使用封装脚本：

```bash
bash scripts/run_train.sh --smoke-test
```

### 评估

```bash
python scripts/eval_classifier.py --checkpoint outputs/wm811k_resnet_baseline/best.pt
```

### GradCAM 热力图

```bash
python scripts/visualize_cam.py --checkpoint outputs/wm811k_resnet_baseline/best.pt
```

> ⚠️ GradCAM 热力图展示的是分类器的注意力区域（interpretability），
> 不等同于真正的 anomaly localization（需要 PaDiM/PatchCore 等方法）。

---

## 配置说明

训练配置 `configs/train/wm811k_resnet_baseline.yaml`：

| 参数 | 说明 |
|------|------|
| `task_mode` | `binary` (normal/abnormal) 或 `multiclass` (9 类 failure type) |
| `model.arch` | ResNet 变体：`resnet18` / `resnet34` / `resnet50` |
| `model.in_channels` | `1` = 单通道原始 wafer map；`3` = 三通道复制 |
| `model.pretrained` | 是否加载 ImageNet 预训练权重 |
| `model.dropout` | FC 层前的 dropout |
| `data.augment` | 仅支持 wafer-safe 的空间增强（flip, rotate90） |

---

## 双线开发工作流

项目现在支持两条并行工作线，共用同一套 Python 入口和路径约定。

### 1. 本地原生开发线

适合日常写代码、跑 smoke test、快速评估：

```bash
conda activate torch
make train
make smoke-test
make eval
```

### 2. 远程 GPU 训练线

远程控制脚本已经迁移到 `scripts-remote/`，并改成 Python CLI。现在按三类职责拆分：
- `scripts-remote/deploy.py`：部署代码与环境，并可选远端下载/处理数据
- `scripts-remote/remote_run.py`：先同步代码，再在远端执行任意 `scripts/` 下脚本，并把小文件结果直接回写到本地 `outputs/`
- `scripts-remote/fetch_all_output.py`：按需补拉完整输出目录，包括大文件

开始之前，请先配置好 SSH 密钥免密登录。当前远程工作流默认通过 `ssh` / `rsync` 直接访问远端主机，如果还需要手动输入密码，部署、同步代码和回传输出时都容易卡在交互提示上。

建议先确认下面这条命令可以直接登录成功且不要求输入密码：

```bash
ssh -p 20277 root@host
```

推荐用 `remote-deploy` 一步完成部署 + 数据下载 + 数据预处理，再运行训练：

```bash
make remote-deploy \
  HOST=root@host \
  PORT=20277 \
  PREPARE_DATA=1 \
  PROCESS_SUBSETS="labeled"
```

`PREPARE_DATA=1` 会在部署结束后自动在远端执行 `scripts/prepare_data.py`（下载数据集，默认 `DATASET=WM-811K`）；`PROCESS_SUBSETS="labeled"` 则继续执行 `scripts/process_data.py --subset labeled`，生成 224×224 processed HDF5。

如果数据已经就绪、或想单独重跑某一步，也可以分步执行：

```bash
# 单独跑数据下载
make remote-prepare-data
```

```bash
# 单独跑处理
make remote-process-data
```

```bash
make remote-train \
  CONFIG=configs/train/wm811k_resnet_baseline.yaml \
  ARGS="--smoke-test"
```

```bash
make remote-run \
  SCRIPT=scripts/eval_classifier.py \
  ARGS="--checkpoint outputs/runs/run-20260330-120000/best.pt"
```

```bash
make remote-fetch-all-output RUN_ID=run-20260330-120000
```

这套 Python CLI 现在优先针对 Vast.ai / Runpod 这类“SSH 进去已经在容器/工作空间里”的远端 shell 模式，不再依赖 Docker in Docker。

`remote-deploy` 现在默认使用 `system` 部署模式：
- 不创建 `venv`
- 不自动安装 `torch` / `torchvision`
- 直接使用系统默认 `python3`
- 默认要求你先探测并选择远端解释器，避免误用没有 torch 的系统 Python
- 依赖会安装到你最终选中的 `python_bin`
- 回传远端当前的 Python / torch / torchvision / CUDA 探测结果

默认 bootstrap 模板是：

```bash
{python_bin} -m pip install --no-cache-dir -r requirements.txt
```

例如如果你选择的是 `/opt/miniforge3/bin/python`，实际执行的就是：

```bash
/opt/miniforge3/bin/python -m pip install --no-cache-dir -r requirements.txt
```

如果你明确想走隔离环境，也可以切回 `venv` 模式：

```bash
make remote-deploy \
  HOST=root@host \
  PORT=20277 \
  REMOTE_DEPLOYMENT_MODE=venv
```

只有在 `venv` 模式下，`remote-deploy` 才会继续按 CUDA 版本自动安装当前支持矩阵里的 PyTorch 组合：
- CUDA 13.0: `torch==2.10.0`, `torchvision==0.25.0`
- CUDA 12.8: `torch==2.10.0`, `torchvision==0.25.0`
- CUDA 12.6: `torch==2.10.0`, `torchvision==0.25.0`
- CUDA 12.4: `torch==2.6.0`, `torchvision==0.21.0`
- 未检测到可用 GPU: CPU 版 `torch==2.10.0`, `torchvision==0.25.0`

也可以通过 `REMOTE_BOOTSTRAP_CMD=...` 覆盖，或在 `venv` 模式下加 `SKIP_TORCH_INSTALL=1` 跳过自动安装。

如果你已经知道远端要用哪个解释器，也可以直接指定：

```bash
make remote-deploy \
  HOST=root@host \
  PORT=20277 \
  REMOTE_PYTHON_BIN=/opt/miniforge3/bin/python
```

`remote-train` 会：
- 先把本地代码与配置同步到远端项目目录
- 根据你选择的训练配置和附加参数运行 `scripts/train_classifier.py`
- 在本地实时跟随远端训练日志
- 训练结束后按 `SYNC_MAX_SIZE` 限制把远端 `outputs/` 中的小文件直接同步回本地 `outputs/`

`remote-prepare-data` 和 `remote-process-data` 也都是通过 `remote_run.py` 执行，因此会共享同一套远端 `python_bin`、数据根目录和输出回传逻辑。

如果你想执行任意远端脚本，可以直接调用：

```bash
python scripts-remote/remote_run.py scripts/train_classifier.py -- --smoke-test
python scripts-remote/remote_run.py scripts/eval_classifier.py -- --checkpoint outputs/runs/run-20260330-120000/best.pt
```

默认情况下，`remote_run.py` 会把小于 `32m` 的文件从远端 `outputs/` 回写到本地 `outputs/`。需要补拉大文件时，再执行：

```bash
python scripts-remote/fetch_all_output.py --run-id run-20260330-120000
python scripts-remote/fetch_all_output.py --all
```

这套自动选择策略的依据来自 PyTorch 官方 previous versions 页面，当前已经覆盖你手头的两类主机：
- `CUDA 12.8`
- `CUDA 13.0`

## 后续计划

- [ ] PaDiM / PatchCore 基于正常样本特征分布的异常定位
- [ ] MixedWM38 多标签分类
- [ ] 更精细的 class-balanced sampling / focal loss
- [ ] 持续学习实验框架

---

## 致谢

本项目面向晶圆缺陷检测与模式识别研究，参考了 wafer map 缺陷分析、工业异常检测与可扩展实验组织的相关工作，并结合当前课题需求进行实现与扩展。

---

## License

当前默认仅供科研与学习使用。正式开源前请根据数据集和项目要求补充许可证信息。
