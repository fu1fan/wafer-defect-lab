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
pip install -r requirements.txt -r requirements-cu128.txt
```

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
make train
make smoke-test
make eval
```

### 2. Docker / 远程 GPU 训练线

容器基座固定为官方镜像：

```text
pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
```

项目现在分成两层镜像：
- `ghcr.io/<owner>/wafer-defect-lab-base:cu128`：稳定的 PyTorch + CUDA + Python 依赖层
- `ghcr.io/<owner>/wafer-defect-lab:<tag>`：轻量的项目代码层

运行镜像不包含 `data/` 与 `outputs/`。这两部分通过挂载目录注入。

本地构建镜像：

```bash
make docker-build
```

在本机通过 Docker 运行训练：

```bash
make docker-train-local
```

这个本地 Docker 入口也会优先申请 GPU；如果宿主机没有可见的 NVIDIA 环境，则直接以 CPU 方式启动容器。

远程控制脚本已经迁移到 `scripts-remote/`，并改成 Python CLI。现在按三类职责拆分：
- `scripts-remote/deploy.py`：部署代码与环境，并可选远端下载/处理数据
- `scripts-remote/train.py`：按配置启动远端训练，训练完成后自动拉回报告
- `scripts-remote/fetch_weights.py`：按需下载权重文件

推荐先部署，再训练：

```bash
make remote-deploy \
  HOST=root@host \
  PORT=20277 \
  PREPARE_DATA=1 \
  PROCESS_SUBSETS="labeled"
```

```bash
make remote-train \
  HOST=root@host \
  PORT=20277 \
  CONFIG=configs/train/wm811k_resnet_baseline.yaml \
  ARGS="--smoke-test"
```

```bash
make remote-fetch-weights PATTERN="best.pt"
```

这套 Python CLI 现在优先针对 Vast.ai / Runpod 这类“SSH 进去已经在容器/工作空间里”的远端 shell 模式，不再依赖 Docker in Docker。

如果远端当前环境还没有装好依赖，`remote-deploy` 会默认执行：

```bash
python3 -m venv /workspace/waferlab-venv
/workspace/waferlab-venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-cu128.txt
```

也可以通过 `REMOTE_BOOTSTRAP_CMD=...` 覆盖。

`remote-train` 会：
- 根据你选择的训练配置和附加参数运行 `scripts/train_classifier.py`
- 在本地实时跟随远端训练日志
- 训练结束后自动同步小报告到本地 `outputs/remote/<run_id>/`

当前远端兼容基线已回退到更保守的组合：
- `torch==2.9.1`
- `torchvision==0.24.1`
- `CUDA 12.8`

这样比 `torch 2.11 + CUDA 13` 更容易匹配现有的 R570 驱动环境。

### GitHub Actions 镜像发布

仓库包含 GitHub Actions 工作流，会在 `main` 分支推送或手动触发时构建并推送镜像到 GHCR：

```text
.github/workflows/docker-image.yml
```

镜像标签包含分支、tag、commit SHA 和默认分支的 `latest`。

---

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
