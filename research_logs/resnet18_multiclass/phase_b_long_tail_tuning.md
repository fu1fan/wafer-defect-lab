# Phase B：长尾类别处理与策略筛选

## 背景与问题

Phase A 说明直接把 `resnet18_recall_opt` 迁移到 9 类多分类后，少数类仍然明显吃亏。  
这一阶段的目标是系统比较不同的类别重平衡策略，确认哪种方法最值得作为 ResNet18 多分类的长期保留方案。

## 采取措施

- 加入 `class-balanced sampler`。
- 加入 `class-balanced focal` 权重。
- 对比 sampler 与 loss 单独使用、以及 sampler + class-balanced focal 的组合。
- 使用 8 epoch 筛选实验和 20 epoch 完整训练两条线并行观察。

## 实验设置

- 配置文件：
  - `configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_cb_sampler_multiclass.yaml`
  - `configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_cb_focal_multiclass.yaml`
  - `configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_cb_combined_multiclass.yaml`
- recipe：
  - `configs/modal/baseline/recipes/cb_sampler_focal_multiclass.yaml`
  - `configs/modal/baseline/recipes/cb_focal_multiclass.yaml`
  - `configs/modal/baseline/recipes/cb_sampler_cb_focal_multiclass.yaml`
- 本地输出目录：
  - `outputs/screen_cb_focal_8ep/`
  - `outputs/screen_cb_sampler_8ep/`
  - `outputs/screen_cb_combined_8ep/`
  - `outputs/baseline_cb_sampler_multiclass_full/`
- checkpoint：`outputs/baseline_cb_sampler_multiclass_full/best.pt`
- commit / 日期范围：`7d3c61f`（2026-04-06，multiclass imbalance recipes）
- 训练轮数：
  - 筛选实验：`8`
  - 完整训练：`20`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| B1_cb_focal_8ep | class-balanced focal | 0.9649 | 0.7391 | 0.7487 | 0.6909 | — | 能改善部分类别，但整体不如 sampler 稳定 |
| B2_cb_sampler_8ep | class-balanced sampler | 0.9626 | 0.7413 | 0.7799 | 0.7672 | — | 少数类召回最激进，但仅为短程筛选 |
| B3_cb_combined_8ep | sampler + class-balanced focal | 0.9589 | 0.7440 | 0.7939 | 0.7477 | — | recall 很高，但 accuracy 损失更明显 |
| B4_cb_sampler_full | class-balanced sampler + 完整 20 epoch | 0.9666 | 0.7692 | 0.7759 | 0.7084 | — | 兼顾稳定性与长尾表现，是本线程最可保留方案 |

## 分析与决策

这组实验给出的结论相对稳定：

- `class-balanced sampler` 是最关键的改动，不保留它会显著削弱少数类。
- `class-balanced focal` 单独使用有帮助，但不如 sampler 稳定。
- `sampler + class-balanced focal` 虽然能进一步推高部分 recall，但代价是整体稳定性和 accuracy 更差。

因此本线程最终保留的是 `cb_sampler_full`，也就是以 sampler 为主、而不是继续叠加更重的 loss 修饰。

## 下一步

接受 ResNet18 在 9 类任务上的阶段性上限，转向更强 backbone 或结构性新思路，而不是继续在该线程内部过度打磨。
