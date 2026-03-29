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