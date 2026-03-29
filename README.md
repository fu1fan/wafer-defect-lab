# Wafer Defect Lab

一个面向晶圆缺陷研究的深度学习实验仓库。当前阶段聚焦于将 52×52 的 wafer map 扩展为 224×224 输入，完成基于 CNN backbone 的异常检测与定位 baseline；后续逐步扩展到混合缺陷多标签分类与持续学习。

---

## 项目目标

本项目围绕晶圆缺陷检测与识别展开，按以下路线逐步推进：

1. **异常检测与定位 baseline**
   - 输入：52×52 wafer map
   - 处理：扩展/转换为 224×224 图像输入
   - 输出：异常分数、异常热力图、异常位置

2. **混合缺陷多标签分类**
   - 针对 mixed-type wafer defect pattern
   - 支持一张 wafer map 同时包含多种缺陷类型

3. **持续学习**
   - 在新缺陷模式不断加入时，尽量减少遗忘
   - 为后续 Prompt / Prototype / Replay 等方法预留接口

---

## 当前阶段

当前优先任务是：

- 完成 WM-811K / Wafer map 数据读取与整理
- 实现 52×52 到 224×224 的输入转换
- 建立 CNN backbone 的异常检测 baseline
- 输出异常热力图与定位结果
- 搭建统一、可扩展的实验代码结构

当前推荐的第一版 baseline：

- Backbone: `ResNet18`
- Anomaly method: `PaDiM`
- 后续对照：`PatchCore`、`MobileNetV3`

---

## 数据说明

### 1. Wafer Map 表示

每张 wafer map 本质上是一个 52×52 矩阵，常见取值如下：

- `0`: blank / 背景
- `1`: normal die / 正常芯片
- `2`: defect die / 缺陷芯片

为了适配视觉模型，本项目会将其转换为 224×224 图像表示，用于 CNN backbone 的特征提取与异常检测。

### 2. 当前计划使用的数据

- `WM-811K`
  - 用于单缺陷模式分析、异常检测 baseline
- `MixedWM38`
  - 用于后续混合缺陷多标签分类实验

> 注意：原始数据文件不直接提交到 Git 仓库，请放到 `data/raw/` 目录下。

---

## 项目结构

```text
wafer-defect-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── exp/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
├── notebooks/
├── outputs/
│   ├── heatmaps/
│   ├── localization/
│   ├── predictions/
│   └── metrics/
├── scripts/
├── tests/
└── src/
    └── waferlab/
        ├── data/
        ├── models/
        ├── engine/
        ├── metrics/
        ├── utils/
        └── visualize/
```

### 目录说明

- `configs/`：实验配置文件，保存数据、模型、训练与实验组合参数
- `data/raw/`：原始数据
- `data/interim/`：中间处理结果，如 52×52 到 224×224 的转换结果
- `data/processed/`：按任务组织好的训练/验证/测试集
- `data/splits/`：数据划分文件
- `notebooks/`：探索分析、可视化与错误分析
- `outputs/`：模型输出结果，包括热力图、定位结果和评估指标
- `scripts/`：可直接运行的训练、推理、评估与数据处理脚本
- `src/waferlab/`：核心源码
- `tests/`：基本单元测试

---

## 开发路线

### Phase 1: 异常检测与定位

- [ ] 读取 WM-811K 数据
- [ ] 构建 52×52 wafer map 可视化脚本
- [ ] 实现 52×52 → 224×224 转换
- [ ] 构建 anomaly detection 数据划分
- [ ] 实现 ResNet18 + PaDiM baseline
- [ ] 输出 anomaly heatmap
- [ ] 输出异常位置/定位结果
- [ ] 完成第一版评估脚本

### Phase 2: 混合缺陷多标签分类

- [ ] 接入 MixedWM38
- [ ] 设计多标签标注格式
- [ ] 实现多标签分类 head
- [ ] 处理类别不平衡问题
- [ ] 增加错误分析和混淆模式分析

### Phase 3: 持续学习

- [ ] 设计任务序列划分
- [ ] 实现 replay / prototype / prompt 等模块接口
- [ ] 评估遗忘与增量性能
- [ ] 建立统一 continual learning benchmark

---

## 安装

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd wafer-defect-lab
```

### 2. 创建环境

推荐使用 `conda` 或 `venv`：

```bash
conda create -n waferlab python=3.10
conda activate waferlab
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 数据准备

请将原始数据放到以下目录：

```text
data/raw/wm811k/
data/raw/mixedwm38/
```

建议后续统一通过脚本生成中间数据和处理后数据，而不是手动改文件夹。

---

## 快速开始

### 1. 预处理 WM-811K

```bash
python scripts/prepare_wm811k.py --config configs/data/wm811k.yaml
```

### 2. 将 52×52 wafer map 转换为 224×224 输入

```bash
python scripts/convert_52_to_224.py --config configs/data/wm811k.yaml
```

### 3. 构建异常检测数据集

```bash
python scripts/build_anomaly_dataset.py --config configs/train/anomaly_baseline.yaml
```

### 4. 训练异常检测 baseline

```bash
python scripts/train_anomaly.py --config configs/exp/exp_001_resnet18_padim.yaml
```

### 5. 推理并生成热力图

```bash
python scripts/infer_anomaly.py --config configs/exp/exp_001_resnet18_padim.yaml
```

### 6. 评估结果

```bash
python scripts/eval_anomaly.py --config configs/exp/exp_001_resnet18_padim.yaml
```

---

## 第一阶段实验规范

当前建议统一采用以下约定：

- 输入尺寸：`224×224`
- 输入形式：3 通道图像
- 第一版 backbone：`ResNet18`
- 第一版 anomaly model：`PaDiM`
- 对照方法：`PatchCore`
- 输出内容：
  - image-level anomaly score
  - pixel/region-level anomaly heatmap
  - localization result
  - quantitative metrics

---

## 评估内容

第一阶段重点评估以下指标：

### 图像级

- AUROC
- Precision / Recall
- F1-score

### 定位级

- heatmap 可视化质量
- threshold 后的异常区域定位效果
- IoU / region overlap（如果有定位标注）

### 工程级

- 推理速度
- 显存占用
- 模型参数量

---

## 编码规范

建议遵循以下原则：

- 配置与代码分离，参数尽量写入 yaml
- 尽量避免 notebook 中堆积核心逻辑
- 核心训练和推理流程都应能脚本化运行
- 每次实验固定随机种子
- 输出目录按实验编号保存，避免覆盖旧结果

---

## 命名规则

### 实验命名

建议采用统一实验编号：

- `exp_001_resnet18_padim`
- `exp_002_resnet18_patchcore`
- `exp_003_mobilenetv3_padim`

### 输出目录命名

```text
outputs/
├── heatmaps/exp_001_resnet18_padim/
├── localization/exp_001_resnet18_padim/
├── predictions/exp_001_resnet18_padim/
└── metrics/exp_001_resnet18_padim/
```

---

## 后续计划

本仓库将逐步扩展以下内容：

- [ ] PatchCore baseline
- [ ] MobileNetV3 lightweight baseline
- [ ] MixedWM38 多标签分类
- [ ] 持续学习实验框架
- [ ] 多模态缺陷理解与问答接口
- [ ] 更完整的实验记录与报告生成

---

## 注意事项

- 不要将大体积原始数据直接提交到 Git
- 不要把训练得到的 checkpoint 默认加入版本管理
- 重要实验请保留配置文件、日志和结果图
- 数据划分文件应固定保存，保证结果可复现

---

## 致谢

本项目面向晶圆缺陷检测与模式识别研究，参考了 wafer map 缺陷分析、工业异常检测与可扩展实验组织的相关工作，并结合当前课题需求进行实现与扩展。

---

## License

当前默认仅供科研与学习使用。正式开源前请根据数据集和项目要求补充许可证信息。