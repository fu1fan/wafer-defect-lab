# ResNet50 多分类线程总览

## 研究目标

在确认 ResNet18 已不足以支撑 9 类多分类长期主线后，以 `ResNet50` 为新 backbone 重建更强的 `WM-811K` 多分类基线，并用分阶段方式系统完成基线复现、超参数搜索、结构增强和最终定版。

## 当前最佳方案

- 方案名称：`G2_gamma1.0_logitadj`
- 基础：`D1_gem_tuned` → cRT (冻结 backbone, focal γ=1.0, LR=1e-3, 4ep) + 推理时 logit adjustment
- 关键策略：`ResNet50 + GeM + cRT(focal_gamma=1.0, CB sampler, 4ep, AdamW) + LogitAdj(tau=-0.05)`
- 对应 checkpoint：`outputs/phase_g/G2_gamma1.0/best.pt`
- 推理时后处理：`logits -= (-0.05) × log(class_prior)`
- 训练脚本：`scripts/crt_sweep.py`（配置 `G2_gamma1.0`）

## 当前最佳指标

- accuracy：`0.9591`
- macro_f1：`0.7486`
- macro_recall：`0.8401`
- minority_recall_mean：`0.8399`
- score：`0.8213`

## 阶段列表

1. Phase A：增强版 `R0` 基线复现
2. Phase B：9 组超参数搜索
3. Phase C：`GeM / CBAM / GeM+CBAM` 结构增强
4. Phase D：最佳超参与最佳结构组合定版（原主线 score=0.8017）
5. Phase F：后校准与解耦重平衡（score=0.8113）
6. Phase G：cRT 超参数搜索与交叉验证（新主线 score=0.8213，+0.0196）

## 当前结论

- `GeM Pooling` 依然是最有效的结构改动。
- `class-balanced sampler` + `focal loss` 是必须保留的训练前提。
- 后校准（temperature scaling / bias correction / tau-norm 等）在当前设置下无法超过基线。
- **解耦重训练（cRT）是最有效的改进路径**：冻结 backbone 重训 head 可大幅改善少数类 recall。
- **cRT 最优配置：focal γ=1.0 + LR=1e-3 + 4 epochs**（比默认 γ=1.5 更好）。
- **cRT + post-hoc logit adjustment 可以 stacking**：最优 τ=-0.05，额外提升 +0.0018。
- 最终最优方案 score=0.8213（+0.0196），改进主要来自 Center/Edge-Ring/Scratch 的 recall 大幅提升。
- accuracy 从 0.9541 提升至 0.9591，macro_recall 从 0.7988 提升至 0.8401。
- 代价：Loc (-0.057)、Edge-Loc (-0.045)、Donut (-0.027) 三个类有轻微 recall 回退。

## 下一步计划

- Loc 类 recall 恢复（当前 0.636，从 0.693 下降）是最大的改进空间
- 可考虑 class-wise threshold tuning 或 Loc-specific 数据增强
- 进一步探索是否可在 cRT 阶段使用 mixed precision 或更长训练
