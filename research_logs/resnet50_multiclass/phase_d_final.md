# Phase D：最终组合定版

## 背景与问题

Phase B 已经找到了最优超参数组合，Phase C 已经确认 `GeM` 是唯一值得保留的结构增强。  
因此这一阶段的目标不再是发散搜索，而是把两个阶段最有效的部分组合起来，形成当前线程的最终版本。

## 采取措施

- 采用 `ResNet50 + GeM` 结构。
- 采用 Phase B 最优训练设置：
  - `lr=5e-4`
  - `focal_gamma=1.5`
  - `class-balanced sampler`
- 保留 heavy augmentation。
- 将训练轮数提升到 `25`，作为正式定版训练。

## 实验设置

- 配置文件：`configs/modal/baseline/experiments/phase_d/resnet50_gem_tuned.yaml`
- 模型路径：`configs/modal/baseline/models/resnet50_gem.yaml`
- 本地输出目录：`outputs/phase_d/D1_gem_tuned/`
- checkpoint：`outputs/phase_d/D1_gem_tuned/best.pt`
- 关键文件：
  - `outputs/phase_d/D1_gem_tuned/run_summary.json`
  - `outputs/phase_d/D1_gem_tuned/eval_metrics_Test.json`
- commit / 日期范围：
  - 配置：未追踪（当前工作区新增文件）
  - 结果创建时间：`2026-04-09T08:18:58`
- 训练轮数：`25`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| D1_gem_tuned | GeM + `lr=5e-4` + `gamma=1.5` + class-balanced sampler | 0.9541 | 0.7485 | 0.7988 | 0.7975 | 0.8017 | 当前线程最优方案 |

## 分析与决策

`D1_gem_tuned` 是当前线程的最终保留版本，原因有三：

1. 它把 Phase B 最有效的训练改动和 Phase C 最有效的结构改动真正整合到了一起。
2. 它在当前定义的综合评分上达到 `0.8017`，高于所有前序方案。
3. 它最符合本线程的目标，即优先兼顾少数类召回，而不是只追整体 accuracy。

与 Phase A 的 `R0` 相比，`D1_gem_tuned` 的提升非常明确：

- `score`：`0.7295 -> 0.8017`
- `macro_f1`：`0.6738 -> 0.7485`
- `macro_recall`：`0.7282 -> 0.7988`
- `minority_recall_mean`：`0.6866 -> 0.7975`

当前阶段的最终交付物如下：

- 最优配置：`configs/modal/baseline/experiments/phase_d/resnet50_gem_tuned.yaml`
- 最优输出目录：`outputs/phase_d/D1_gem_tuned/`
- 最优 checkpoint：`outputs/phase_d/D1_gem_tuned/best.pt`
- 最终评估：`outputs/phase_d/D1_gem_tuned/eval_metrics_Test.json`

## 下一步

继续围绕损失函数和采样策略做更细粒度微调，而不是继续叠加更重的注意力模块。
