# Phase B：异常类召回率优化

## 背景与问题

Phase A 说明标准 ResNet18 在二分类场景下已经有较高整体精度，但 abnormal 类 recall 明显偏低。  
这一阶段的核心问题是：如何在不平衡数据集上显著提升异常类召回，同时尽量保持整体精度和可用的 precision。

## 采取措施

- 将 backbone 升级为 `resnet18_recall_opt`。
- 用 `GeM Pooling` 替代全局平均池化，增强对稀疏异常模式的响应。
- 在 pooled feature 后加入 `SE attention`。
- 使用两层分类头（`Linear + BN + GELU`）。
- 训练上引入 `Focal Loss` 和类别权重 `class_weights=[0.25, 1.0]`。
- 同时保留 `CE + class_weights` 作为对照实验。

## 实验设置

- 配置文件：`configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_focal.yaml`
- 模型路径：`configs/modal/baseline/models/resnet18_recall_opt.yaml`
- 代码实现：`src/waferlab/models/resnet_recall_opt.py`
- 本地输出目录：
  - `outputs/recall_opt_short1_focal_025/`
  - `outputs/recall_opt_short2_ce_w8/`
  - `outputs/resnet18_recall_opt_final/`
- checkpoint：`outputs/resnet18_recall_opt_final/best.pt`
- commit / 日期范围：`bfa1f79`（2026-04-05，ResNet18 recall-oriented 优化）
- 训练轮数：
  - 短实验：`5`
  - 最终实验：`20`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| B1_short_focal | Focal + alpha=[0.25,1.0] + recall-opt 结构 | 0.9465 | 0.8326 | 0.9307 | 0.9125 | — | abnormal recall 首次超过 0.90，但 precision 较低 |
| B2_short_ce_w8 | CE + class_weights=[1,8] + recall-opt 结构 | 0.9694 | 0.8714 | 0.8554 | 0.7240 | — | precision 更稳，但 recall 不如 focal |
| B3_final_recall_opt | recall-opt 结构 + Focal + class_weights | 0.9727 | 0.8945 | 0.9102 | 0.8381 | — | 在 recall 与 precision 之间取得最好平衡 |

## 分析与决策

这一阶段的主要结论非常明确：

- `Focal Loss` 比单纯 `CE + class weights` 更适合当前异常召回目标。
- `GeM + SE + 两层分类头` 的 recall-oriented 结构设计是有效的。
- 最终版本在 abnormal recall 提升到 `0.8381` 的同时，accuracy 也提升到 `0.9727`，说明这不是单纯拿 precision 换 recall。

因此本线程的最终保留方案是：`resnet18_recall_opt + Focal + class weights`。

## 下一步

将二分类阶段验证有效的“召回优先”经验迁移到 9 类多分类场景，观察其在长尾类别上的可迁移性。
