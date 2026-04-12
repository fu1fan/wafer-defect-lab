# Phase A：ResNet18 多分类基线

## 背景与问题

二分类线程已经证明 `resnet18_recall_opt` 在异常召回上有效，因此下一步自然是把这套结构迁移到 9 类 failure type 多分类，观察其在长尾类别上的表现。

## 采取措施

- 将任务模式切换为 `multiclass`。
- 延续 `resnet18_recall_opt` 结构。
- 训练上采用 `Focal Loss`，先不加入额外采样策略，作为多分类起点。

## 实验设置

- 配置文件：`configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_focal_multiclass.yaml`
- 模型路径：`configs/modal/baseline/models/resnet18_recall_opt.yaml`
- recipe：`configs/modal/baseline/recipes/focal_multiclass.yaml`
- 本地输出目录：`outputs/research_nest_recall_opt_focal_multiclass_manual/`
- checkpoint：`outputs/research_nest_recall_opt_focal_multiclass_manual/best.pt`
- commit / 日期范围：`7d3c61f`（2026-04-06，baseline multiclass configs）
- 训练轮数：`20`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| A0_recall_opt_multiclass | recall-opt ResNet18 + focal，多分类直迁移 | 0.9696 | 0.7458 | 0.7065 | 0.6153 | — | 整体 accuracy 高，但少数类仍明显受限 |

## 分析与决策

这一阶段最关键的观察是：

- `accuracy=0.9696` 并不代表多分类已经解决，因为 `none` 类占比极大。
- `Donut / Loc / Scratch` 等少数类 recall 依然偏低，说明二分类里有效的 recall-oriented 结构不能直接解决多分类长尾问题。

因此下一步必须把重点从“模型结构小修补”转移到“采样与类别重平衡策略”上。

## 下一步

系统比较 `class-balanced sampler`、`class-balanced focal` 与组合策略，验证哪种方案最值得保留为 ResNet18 多分类阶段的稳定版本。
