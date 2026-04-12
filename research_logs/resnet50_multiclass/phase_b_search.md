# Phase B：超参数搜索

## 背景与问题

Phase A 的 `R0` 虽然已经形成较强起点，但训练后期震荡明显，说明当前超参数仍偏激进。  
这一阶段的目标是用一批可比的 15 epoch 搜索实验，找出更稳定且更偏向少数类召回的训练组合。

## 采取措施

- 固定 ResNet50 主干和 `class-balanced sampler` 主路线。
- 系统比较不同的：
  - `learning rate`
  - `weight_decay`
  - `dropout`
  - `focal_gamma`
  - augmentation 强度
  - 是否移除 sampler
- 使用统一评分函数：
  - `0.40 * macro_f1`
  - `0.25 * macro_recall`
  - `0.20 * minority_recall_mean`
  - `0.15 * accuracy`

## 实验设置

- 配置目录：`configs/modal/baseline/experiments/search/`
- 参考模型路径：`configs/modal/baseline/models/resnet50.yaml`
- 自动化脚本：`scripts/run_resnet50_search.py`
- 本地输出目录：`outputs/phase_b_search/`
- 关键文件：
  - `outputs/phase_b_search/search_results.json`
  - `outputs/phase_b_search/search_log.txt`
- commit / 日期范围：
  - 配置与脚本：未追踪（当前工作区新增文件）
  - 搜索结果日期：2026-04-08 ~ 2026-04-09（按本地输出与日志整理）
- 训练轮数：`15`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| B1_r0_ref | R0 参考复现，15 epoch 对齐比较 | 0.9342 | 0.6654 | 0.7613 | 0.7767 | 0.7519 | 可作为统一对照，但不是最终最佳 |
| B2_lr5e4_gamma1p5 | `lr=5e-4, gamma=1.5` | 0.9395 | 0.7064 | 0.8096 | 0.8169 | 0.7893 | 本阶段最佳，整体最稳 |
| B3_lr3e4_wd5e4_drop03 | 更低 lr、更低 wd、更高 dropout | 0.9374 | 0.6970 | 0.8079 | 0.8162 | 0.7846 | 接近 B2，但综合略弱 |
| B4_wd5e4_drop03_gamma2p5 | 更高 gamma | 0.9286 | 0.5902 | 0.6155 | 0.6026 | 0.6498 | gamma 过高明显伤害效果 |
| B5_lr5e4_wd1e3_drop02 | 中等 lr + gamma=2.0 | 0.9234 | 0.7097 | 0.7940 | 0.7977 | 0.7804 | 比 R0 稳，但不如 B2 |
| B6_gamma1p0_wd1e3_drop01 | `gamma=1.0` 最弱 focal | 0.9365 | 0.7116 | 0.8035 | 0.8087 | 0.7877 | 很接近 B2，说明降低 gamma 有价值 |
| B7_strong_aug | 更强平移/缩放增强 | 0.9443 | 0.6884 | 0.7435 | 0.6900 | 0.7409 | accuracy 上升，但少数类不够理想 |
| B8_no_sampler | 移除 class-balanced sampler | 0.9580 | 0.6283 | 0.5974 | 0.5179 | 0.6479 | 少数类性能明显崩坏 |
| B9_lr5e4_gamma2p5_drop03 | 低 lr + 高 gamma | 0.9536 | 0.6517 | 0.6738 | 0.6293 | 0.6980 | 高 gamma 依旧不适合当前任务 |

## 分析与决策

这一阶段把后续主线几乎完全定下来了：

- `lr=5e-4` 或 `3e-4` 明显比 `1e-3` 更稳定。
- `focal_gamma=1.5` 比 `2.0` 更适合当前少数类。
- 去掉 `class-balanced sampler` 后，少数类会明显崩掉，说明 sampler 必须保留。
- 高 gamma 方案（B4 / B9）整体都不理想，说明 focal 不能再继续加重。

因此 Phase B 的明确结论是：**B2 是最佳超参数组合，B3 和 B6 可以作为备选，但主线进入下一阶段时应以 B2 为基础。**

## 下一步

在 B2 的训练设置上叠加结构增强，对比 `GeM`、`CBAM` 与 `GeM + CBAM` 的实际增益。
