# Phase A：ResNet18 二分类基线

## 背景与问题

项目在完成数据处理与训练闭环后，需要先建立一个可复现的 wafer-level 二分类基线，确认标准 CNN 在 `normal / abnormal` 任务上的基本上限与主要瓶颈。

## 采取措施

- 使用标准 `ResNet18` 作为 backbone。
- 采用默认 `CE loss`，不额外做类别重加权。
- 保留工程默认训练设置：`AdamW + cosine scheduler + wafer-safe augmentation`。

## 实验设置

- 配置文件：`configs/modal/baseline/experiments/wm811k_resnet18_baseline.yaml`
- 模型路径：`configs/modal/baseline/models/resnet18.yaml`
- 基础配置：`configs/modal/base/wm811k_classifier.yaml`
- 本地输出目录：`outputs/resnet18_binary_20260405_015930/`
- checkpoint：`outputs/resnet18_binary_20260405_015930/best.pt`
- commit / 日期范围：`3b340c4`（2026-04-05，配置整理后形成基线实验入口）
- 训练轮数：`20`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| A0_resnet18_ce | 标准 ResNet18 + CE | 0.9666 | 0.8379 | 0.7813 | 0.5676 | — | 整体精度高，但 abnormal 召回偏低 |

## 分析与决策

这个阶段确认了两件事：

1. 标准 ResNet18 足以把整体 accuracy 拉到较高水平，说明数据处理与训练链路是通的。
2. abnormal 类 recall 只有 `0.5676`，明显暴露出长尾二分类下的偏置问题，单纯依赖 CE 会更偏向 normal 类。

因此后续决策不是继续围绕“提高整体 accuracy”做小修补，而是明确转向**异常类召回优先**的优化路线。

## 下一步

围绕 abnormal recall 改造模型头部与损失函数，引入更适合少数类的结构和训练策略。
