# Phase C：结构增强对比

## 背景与问题

Phase B 已经把训练超参数稳定在较好的区间，因此这一阶段转向回答一个更具体的问题：  
在当前任务上，值得保留的结构增强到底是哪一个？

## 采取措施

- 对比 `ResNet50 + GeM`
- 对比 `ResNet50 + CBAM`
- 对比 `ResNet50 + GeM + CBAM`
- 使用统一的评分函数与本地评估流程，避免只看单一 accuracy。

## 实验设置

- 配置文件：
  - `configs/modal/baseline/experiments/phase_c/resnet50_gem.yaml`
  - `configs/modal/baseline/experiments/phase_c/resnet50_cbam.yaml`
  - `configs/modal/baseline/experiments/phase_c/resnet50_gem_cbam.yaml`
- 模型路径：
  - `configs/modal/baseline/models/resnet50_gem.yaml`
  - `configs/modal/baseline/models/resnet50_cbam.yaml`
  - `configs/modal/baseline/models/resnet50_gem_cbam.yaml`
- 自动化脚本：`scripts/run_phase_c.py`
- 本地输出目录：`outputs/phase_c/`
- 关键文件：
  - `outputs/phase_c/phase_c_results.json`
  - `outputs/phase_c/phase_c_log.txt`
- commit / 日期范围：
  - 配置、模型与脚本：未追踪（当前工作区新增文件）
  - 结果日期：2026-04-09（按本地输出整理）
- 训练轮数：按本阶段对比设置执行

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| C1_gem | 在 ResNet50 上加入 GeM Pooling | 0.9484 | 0.7452 | 0.7809 | 0.7576 | 0.7871 | 最有价值的结构增强 |
| C2_cbam | 在 ResNet50 上加入 CBAM | 0.9346 | 0.6972 | 0.7264 | 0.6860 | 0.7379 | 不值得保留 |
| C3_gem_cbam | GeM 与 CBAM 同时使用 | 0.9234 | 0.7015 | 0.8137 | 0.7794 | 0.7784 | 没有超过 GeM 单独方案 |

## 分析与决策

这组对比的结论很清晰：

- `GeM` 是当前任务上最有效的结构改进。
- `CBAM` 单独使用的收益不足，甚至拖累整体表现。
- `GeM + CBAM` 虽然能提高部分 recall，但综合评分仍不如 `GeM` 单独使用。

因此 Phase C 的最终决策是：**保留 GeM，放弃 CBAM。**

## 下一步

把 Phase B 的最佳训练参数与 Phase C 的最佳结构 `GeM` 合并，形成最终组合并做定版训练。
