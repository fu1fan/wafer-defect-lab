# ResNet18 二分类线程总览

## 研究目标

在 `WM-811K` 有标签数据上先建立稳定的 wafer-level 二分类基线，并重点解决 abnormal 类在严重类不平衡下召回率偏低的问题。

## 当前最佳方案

- 方案名称：`resnet18_recall_opt + focal + class_weights`
- 关键结构/训练策略：`GeM Pooling + SE attention + 两层分类头 + Focal Loss`
- 对应配置：`configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_focal.yaml`

## 当前最佳指标

- accuracy：`0.9727`
- macro_f1：`0.8945`
- macro_recall：`0.9102`
- minority_recall_mean：`0.8381`
- score：`—`

说明：本线程是二分类任务，这里的 `minority_recall_mean` 等同于 abnormal 类 recall。

## 阶段列表

1. Phase A：标准 `ResNet18 + CE` 二分类基线
2. Phase B：围绕 abnormal recall 的结构与 loss 优化

## 当前结论

标准 ResNet18 已经能达到较高整体精度，但会明显偏向 normal 类。  
在二分类场景下，`GeM + SE + Focal Loss + class weights` 能显著提升 abnormal recall，是后续迁移到多分类与更强 backbone 设计时的重要经验来源。

## 下一步计划

将“召回优先”的设计经验迁移到多分类与更强 backbone 的基线构建中，而不是继续单独扩展二分类线程。
