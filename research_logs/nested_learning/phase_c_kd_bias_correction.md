# Phase C：KD 与偏差校正实验

## 背景与问题

Phase B 虽然把 continual 的内部指标推高到了 `0.60` 左右，但结果仍然被 `none` 类和任务顺序强烈牵制。  
因此这一阶段希望进一步回答两个问题：

1. Knowledge Distillation 能否稳定降低 forgetting？
2. Weight Alignment / Cosine Classifier 等偏差校正手段能否缓解新旧类失衡？

## 采取措施

- 加入 `masked KD`、`replay-only KD`、`additive KD`。
- 加入 `weight alignment`。
- 加入 `cosine classifier`。
- 加入 `teacher snapshot` 与按 task 控制 KD 的逻辑。

## 实验设置

- 相关配置：
  - `configs/modal/research_nest/experiments/wm811k_nested_selfmod_continual_v4.yaml`
  - `configs/modal/research_nest/experiments/wm811k_nested_selfmod_continual_v5_final_d.yaml`
- 相关实现：
  - `scripts/train_continual.py`
  - `src/waferlab/engine/trainer.py`
- 结果汇总文档：`outputs/continual_v5_kd_analysis_report.md`
- 本地输出目录（代表性）：
  - `outputs/continual_v4/`
  - `outputs/continual_v5d/`
  - `outputs/continual_v5_final_d/`
  - `outputs/continual_v5_final_f/`
- commit / 日期范围：
  - `dbd516b`（2026-04-07，KD 变体、masking 控制与实验配置）
  - `0c81fdf`（2026-04-07，去 token 化结构与新 loss）
- 训练轮数：
  - 小规模筛选：每 task `5`
  - 全规模对比：每 task `15`

## 结果汇总

说明：本阶段 `accuracy` 统一记录为 continual 的 `avg_accuracy_final`。

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| C1_v4_ref | 无 KD 参考基线 | 0.4430 | — | — | — | — | 仍是当前任务顺序下最平衡的参考点 |
| C2_v5d_short | Masked KD + WA，小规模筛选 | 0.2300 | — | — | — | — | forgetting 明显下降，但整体效果不足 |
| C3_v5_final_d | Additive replay-only KD，全规模 | 0.2550 | — | — | — | — | forgetting 最低，但 `none` 类严重受损 |
| C4_v5_final_f | KD 只作用于 task 1 | 0.4020 | — | — | — | — | 比全局 KD 更实用，但仍不优于 V4 |

### 持续学习补充指标

| exp_id | avg_accuracy_final | avg_forgetting | macro_accuracy | none_accuracy |
|--------|--------------------|----------------|----------------|---------------|
| C1_v4_ref | 0.443 | 0.338 | 0.389 | 0.998 |
| C2_v5d_short | 0.230 | 0.255 | 0.197 | — |
| C3_v5_final_d | 0.255 | 0.147 | 0.183 | 0.000 |
| C4_v5_final_f | 0.402 | 0.782 | 0.281 | 0.998 |

## 分析与决策

这一阶段的结论非常关键：

- KD 的确能降低 forgetting，尤其 `replay-only KD` 在遗忘控制上最干净。
- 但由于 `none` 类出现在最后一个 task，KD 会与 `none` 的学习目标直接冲突。
- 结果就是：要么旧类保留更好但 `none` 崩掉，要么保住 `none` 但整体改进有限。

因此本线程最终没有把 KD 方案作为主线保留。  
这里得到的核心经验是：当前问题已经不只是“继续堆技术”，而是**任务顺序与数据分布本身决定了很多方法的上限**。

## 下一步

暂停继续深挖当前 continual 设定，先回到更强 backbone 的多分类基线构建，再决定是否重新引入 Nested Learning 机制。
