# Phase A：增强版 ResNet50 基线复现

## 背景与问题

在 ResNet18 多分类线程中已经看到 backbone 容量不足的问题，因此这一阶段的任务是：先用更强的 `ResNet50` 建立一个面向少数类的增强版基线，作为后续搜索和结构增强的共同起点。

## 采取措施

- 将 backbone 升级为 `ResNet50`。
- 保留 `class-balanced sampler`。
- 使用 `Focal Loss`。
- 使用更强的 wafer-safe spatial augmentation：
  - `random_translate_frac=0.08`
  - `random_scale_min=0.95`
  - `random_scale_max=1.05`

## 实验设置

- 配置文件：`configs/modal/baseline/experiments/wm811k_resnet50_cb_sampler_focal_heavy_aug_multiclass.yaml`
- 模型路径：`configs/modal/baseline/models/resnet50.yaml`
- recipe：`configs/modal/baseline/recipes/cb_sampler_focal_multiclass.yaml`
- 本地输出目录：`outputs/r0_resnet50_cb_focal_heavy_aug/`
- checkpoint：`outputs/r0_resnet50_cb_focal_heavy_aug/best.pt`
- commit / 日期范围：`48d0825`（2026-04-08，ResNet50 multiclass + CB sampler + focal + heavy aug）
- 训练轮数：`20`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| A0_r0 | ResNet50 + CB sampler + focal + heavy aug | 0.9373 | 0.6738 | 0.7282 | 0.6866 | 0.7295 | 形成可用起点，但训练后期震荡明显 |

## 分析与决策

R0 的意义不在于它已经足够强，而在于它明确暴露了下一步优化方向：

- backbone 升级是有效的，少数类已经比很多 ResNet18 版本更有希望。
- 但当前 `lr=1e-3 + focal_gamma=2.0` 组合仍然偏激进，训练后期波动明显。
- 因此下一步应该优先做**超参数搜索**，而不是一上来继续叠结构模块。

## 下一步

围绕 `learning rate`、`focal gamma`、`dropout`、`weight_decay` 和 sampler 依赖关系做分阶段筛选，找出更稳定的训练组合。
