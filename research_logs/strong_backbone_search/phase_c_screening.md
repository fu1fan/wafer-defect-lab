# Phase C：中小规模筛选实验

## 背景与问题

Phase B 完成了三个新 backbone（ConvNeXt-Tiny、EfficientNetV2-S、ConvNeXt-Small/Nano）的工程接入和冒烟验证。本阶段目标是通过 5 epoch 短训练，低成本筛选出最有前景的 backbone + 不平衡处理组合，确定进入 Phase D 完整训练的 Top-2 方案。

核心假设：
1. ConvNeXt-Tiny/EfficientNetV2-S 凭借更强的预训练表征，应能在 5 epoch 内展现优于 ResNet50 的趋势
2. Focal loss + CB sampler 组合在 ResNet50 上已验证有效，需确认是否适配新 backbone
3. Balanced Softmax / Logit Adjustment 等损失函数是否在新 backbone 上提供额外增益
4. CB sampler 是否仍然是必要组件

## 采取措施

- 实验 E1：ConvNeXt-Tiny + focal(γ=1.5) + CB sampler（与 D1 同配方，换 backbone）
- 实验 E2：ConvNeXt-Tiny + Balanced Softmax + CB sampler（测试损失层双重纠偏效应）
- 实验 E3：EfficientNetV2-S + focal(γ=1.5) + CB sampler（第二候选 backbone）
- 实验 E5：ConvNeXt-Tiny + Logit Adjustment + CB sampler（测试另一种损失层纠偏）
- 实验 E8：ConvNeXt-Tiny + LDAM + CB sampler（测试 margin-based 损失）
- 实验 E9：ConvNeXt-Tiny + focal(γ=1.5) 无 CB sampler（CB sampler 消融实验）
- 实验 E10：ConvNeXt-Tiny + Logit Adjustment 无 CB sampler（纯损失层纠偏消融）

共 7 组实验，覆盖了 backbone 选择、损失函数对比、采样策略消融三个维度。

跳过的实验及理由：
- E4（EffNetV2-S + Balanced Softmax + CB）：E2 已证明 Balanced Softmax + CB 灾难性失败，无需在另一 backbone 上重复
- E6（EffNetV2-S + Logit Adjustment + CB）：E5 已证明 Logit Adj + CB 少数类 recall 差，优先级低
- E7（ConvNeXt-Small + focal + CB）：ConvNeXt-Small 参数量更大但 5 epoch screening 下 ConvNeXt-Tiny 已有清晰信号，算力有限故跳过

## 实验设置

- 配置文件目录：`configs/modal/baseline/experiments/phase_e_backbone/`
- 模型配置：`configs/modal/baseline/models/{convnext_tiny,efficientnetv2_s}.yaml`
- Recipe 配置：`configs/modal/baseline/recipes/{cb_sampler_focal,cb_sampler_balanced_softmax,cb_sampler_logit_adj,cb_sampler_ldam,logit_adj}_multiclass.yaml`
- 输出目录：`outputs/phase_e/E{1,2,3,5,8,9,10}_*/`
- 训练轮数：5 epochs（筛选用，非最终训练）
- 统一超参：lr=5e-4, wd=1e-3, cosine schedule, batch_size=64
- 评估集：Test split（118595 samples）

评分函数：`score = 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall_mean + 0.15 * accuracy`

少数类（minority）：Center, Donut, Loc, Near-full, Scratch

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| E1 | ConvNeXt-Tiny + focal + CB | 0.8833 | 0.6436 | 0.7440 | 0.7055 | 0.7170 | ✅ **最优，进入 Phase D** |
| E3 | EffNetV2-S + focal + CB | 0.8978 | 0.6060 | 0.7572 | 0.7185 | 0.7100 | ✅ **第二候选，进入 Phase D** |
| E5 | ConvNeXt-Tiny + logit_adj + CB | 0.9576 | 0.6593 | 0.6623 | 0.5556 | 0.6841 | ❌ 高精度但少数类差 |
| E2 | ConvNeXt-Tiny + bal_softmax + CB | 0.4148 | 0.5175 | 0.7003 | 0.7551 | 0.5953 | ❌ 灾难性失败 |
| E9 | ConvNeXt-Tiny + focal 无CB | 0.9476 | 0.4650 | 0.4887 | 0.2691 | 0.5041 | ❌ 无CB灾难性少数类缺失 |
| E8 | ConvNeXt-Tiny + LDAM + CB | 0.7283 | 0.5889 | 0.7523 | 0.7305 | 0.6790 | ❌ 高 recall 但 acc 崩塌 |
| E10 | ConvNeXt-Tiny + logit_adj 无CB | 0.9581 | 0.5479 | 0.4957 | 0.3334 | 0.5535 | ❌ 无CB少数类差 |

### 少数类 Recall 详细

| exp_id | Center | Donut | Loc | Near-full | Scratch | mean |
|--------|--------|-------|-----|-----------|---------|------|
| E1 | 0.617 | 0.660 | 0.605 | 1.000 | 0.646 | 0.706 |
| E3 | 0.611 | 0.281 | 0.737 | 1.000 | 0.964 | 0.719 |
| E5 | 0.565 | 0.589 | 0.379 | 0.958 | 0.287 | 0.556 |
| E2 | 0.785 | 0.822 | 0.666 | 1.000 | 0.503 | 0.755 |
| E8 | 0.724 | 0.719 | 0.486 | 1.000 | 0.724 | 0.731 |
| E9 | 0.325 | 0.034 | 0.060 | 0.926 | 0.000 | 0.269 |
| E10 | 0.579 | 0.171 | 0.232 | 0.684 | 0.000 | 0.333 |

### 训练时间

| exp_id | ~秒/epoch | 总时间 | 备注 |
|--------|-----------|--------|------|
| E1 | ~1060 | ~88min | CB sampler 有额外开销 |
| E3 | ~731 | ~61min | EffNetV2 更快 (30%) |
| E5 | ~1060 | ~88min | 同 E1 |
| E9 | ~550-1060 | ~59min | 无 CB 更快 |

## 分析与决策

### 1. CB Sampler 是必要条件

E9（无 CB）结果灾难性：Scratch recall = 0.0, Donut = 0.034, Loc = 0.060。Focal loss 作为硬样本挖掘机制并不改变采样分布，在极端不平衡（none 占 67.6%）数据集上，仅靠 focal 无法让模型充分学习少数类。CB sampler 通过数据层重采样确保少数类在每个 epoch 内有足够曝光。

**结论：所有后续实验必须使用 CB sampler。**

### 2. Focal Loss 是最佳损失函数

| 损失函数 | 与 CB 搭配效果 | 机制 | 问题 |
|----------|---------------|------|------|
| Focal | ✅ 最优 | 纯硬样本挖掘，不调整类频率 | 与 CB sampler 正交互补 |
| LDAM | ⚠️ 高 recall 但精度崩 | 按类频率设 margin，scale=30 放大 logits | acc=0.73, Scratch prec=0.023 |
| Logit Adjustment | ⚠️ 可用但偏差 | 减去 τ·log(prior)，推测时倾向多数类 | 少数类 recall 不足 |
| Balanced Softmax | ❌ 灾难 | 加 log(N_j) 到 logits + CB 重采样 = 双重过矫 | acc 降至 0.41 |

关键发现：**损失层的类频率调整与数据层的 CB sampler 不可叠加。** Focal loss 成功的关键在于它不做类频率调整，只做难度加权，因此与 CB sampler 正交互补。LDAM（margin-based）虽然少数类 recall 高，但 scale=30 导致输出过度放大，整体精度崩塌（acc=0.73, Scratch precision=0.023），大量 none 样本被错分为少数类。

### 3. ConvNeXt-Tiny 略优于 EfficientNetV2-S

两者在 5 epoch 下表现接近（score 0.717 vs 0.710），但有差异：
- ConvNeXt-Tiny：macro_f1 更高（0.644 vs 0.606），各类 recall 更均衡
- EfficientNetV2-S：训练快 30%，Scratch recall 极高（0.964 vs 0.646）但 Donut 极低（0.281 vs 0.660）

ConvNeXt-Tiny 的少数类表现更均衡，更适合追求"全面提升"而非个别类别突出。

### 4. 5 epoch vs 25 epoch 的预期

5 epoch screening scores（0.71-0.72）不可直接对比 D1 baseline 25 epoch 的 score（0.8017）。ConvNeXt-Tiny 在 E1 中的 val_acc 轨迹为 0.314→0.718→0.691→0.876→0.883，显示其仍在快速收敛中。25 epoch 下预期有显著提升。

### Phase D 决策

**进入 Phase D 完整训练的 Top-2 方案：**

1. **ConvNeXt-Tiny + focal(γ=1.5) + CB sampler**（E1 配方，25 epochs）
   - 理由：screening score 最高，少数类 recall 最均衡
   - 配置：`phase_d/D_convnext_tiny_focal_full.yaml`

2. **EfficientNetV2-S + focal(γ=1.5) + CB sampler**（E3 配方，25 epochs）
   - 理由：screening score 接近第一，训练快 30%，有实用价值
   - 配置：`phase_d/D_efficientnetv2_s_focal_full.yaml`

### 淘汰方案

| 方案 | 淘汰原因 |
|------|----------|
| Balanced Softmax + CB | 双重过矫导致 acc=0.41，不可用 |
| Logit Adjustment + CB | 少数类 recall 严重不足（Loc=0.38, Scratch=0.29） |
| LDAM + CB | margin-based 造成 acc=0.73，Scratch precision=0.023 |
| 任何无 CB sampler 的方案 | 少数类 recall 灾难性（Scratch=0.0） |
| ConvNeXt-Small | 参数更大但无额外收益预期，算力受限跳过 |

## 下一步

Phase D：对 Top-2 方案进行 25 epoch 完整训练，与 D1 baseline（ResNet50+GeM, score=0.8017）做最终对比，决定是否替换当前主线。
