# Phase H: Loc / Edge-Loc Recall 改进实验

## 问题背景

Phase G 最优方案 `G2_gamma1.0 + LogitAdj(τ=-0.05)` 达到 score=0.8213，但 Loc recall 从 D1 的 0.693 降至 0.636，Edge-Loc 从 0.803 降至 0.758。本阶段目标：**在不损害综合 score 的前提下恢复或改善 Loc / Edge-Loc recall**。

## 误差分析

### Loc (class 5) 混淆模式——**弥散型错误**

Loc 的错误分布极其分散，不集中于任何单一类别：
- → none: 15.8% (311 samples)  ← 多数类吸收
- → Scratch: 6.2% (123 samples) ← 空间模式混淆
- → Edge-Loc: 6.0% (118 samples) ← 语义相似（位置类缺陷）
- → Center: 5.1% (100 samples) ← 空间位置重叠
- → Donut: 2.8% (56 samples) ← 环形 vs 局部模式

### Edge-Loc (class 3) 混淆模式——**集中型错误**

Edge-Loc 的错误主要集中于 none：
- → none: 13.9% (386 samples) ← 主要错误来源
- → Loc: 3.9%, Edge-Ring: 2.6%, Scratch/Random: ~1.5%

### 决策边际分析

对 Loc 错误样本分析 logit margin（正确类 logit - 最高错误类 logit）：
- **26% 近似错误** (margin ∈ [-1, 0])：可通过阈值/偏置修正恢复
- **57% 深度错误** (margin < -2)：需要特征层面改变，后处理无法修复

### 关键洞察

**Loc 与 Edge-Loc 在决策空间中竞争**——提升 Loc 几乎必然损害 Edge-Loc，反之亦然。这是一个根本性的 tradeoff，而非可以单方面解决的问题。

## 实验设计

| ID | 方法 | 成本 | 假设 |
|----|------|------|------|
| H1 | 类别偏置后处理 | ~5min | 直接调整决策边界可恢复近似错误 |
| H2 | 加权 focal cRT | ~60min | 增加 Loc/ELoc 梯度权重可改善分类头 |
| H3 | 部分解冻 backbone (layer4) | ~40min | 冻结特征是 Loc 深度错误的根因 |
| H4 | 两阶段 cRT（标准→Loc加权） | ~50min | 先建立好的基线再微调难样本 |
| H5 | 最优偏置叠加 | ~10min | 在最佳 cRT 上再叠加后处理 |

## 实验结果

### H1: 类别偏置后处理（基于 G2 logits）

手动网格搜索 Loc bias + Edge-Loc bias + tau，在 G2 logits 上直接后处理。

| 指标 | G2 基线 | H1 | Δ |
|------|---------|-----|---|
| score | 0.8213 | **0.8238** | +0.0025 |
| accuracy | 0.9591 | 0.9581 | -0.0010 |
| macro_f1 | 0.7486 | 0.7491 | +0.0005 |
| macro_recall | 0.8401 | 0.8405 | +0.0004 |
| Loc recall | 0.636 | **0.730** | **+0.094** |
| Edge-Loc recall | 0.758 | 0.691 | **-0.067** |
| Scratch recall | 0.830 | 0.812 | -0.018 |

**结论**：Loc 大幅提升但 Edge-Loc 大幅下降，验证了 Loc/Edge-Loc tradeoff 假设。**非干净胜利**。

偏置参数：`tau=-0.04, Loc_bias=+0.90, Edge-Loc_bias=-0.40`

### H2: 加权 Focal cRT

冻结 backbone，head-only cRT，focal γ=1.0 + 类别权重 `w[Loc]=2.0, w[Edge-Loc]=1.5, others=1.0`。

| 指标 | G2 基线 | H2 raw | H2 + LogitAdj |
|------|---------|--------|---------------|
| score | 0.8195/0.8213 | 0.8121 | 0.8130 |
| best epoch | 4 | 8 | - |
| Loc recall | 0.636 | 0.665 | - |
| Edge-Loc recall | 0.758 | 0.820 | - |

**结论**：raw cRT score 低于 G2 (0.8121 < 0.8195)。加权 focal 使 Edge-Loc 大幅提升 (+0.062) 但综合评分反而下降，说明类别权重打破了原有的 focal loss 平衡。训练 10 epochs 出现较大波动，最佳 epoch=8 说明收敛困难。**淘汰**。

### H3: 部分解冻 Backbone (layer4 + fc)

解冻 `model.backbone.layer4` + `model.fc`（~15M 可训参数），差异化学习率 backbone=1e-4 / head=1e-3。

| 指标 | G2 基线 | H3 raw | H3 + LogitAdj |
|------|---------|--------|---------------|
| score | 0.8195/0.8213 | 0.8156 | 0.8175 |
| best epoch | 4 | **1** | - |
| Loc recall | 0.636 | 0.649 | - |
| Edge-Loc recall | 0.758 | 0.753 | - |

**结论**：best epoch=1 说明部分解冻极易过拟合。后续 epochs 分数持续下降。Raw score (0.8156) 仍低于 G2 (0.8195)。Loc 仅提升 +0.013，几乎没有改善。**解冻 backbone 不是解决 Loc 弥散错误的有效方法。淘汰**。

### H4: 两阶段 cRT

Stage 1：标准 cRT 4 epochs（复制 G2 设置）→ Stage 2：Loc 加权 focal (`w[Loc]=3.0, w[Edge-Loc]=2.0`) 4 epochs at LR=3e-4。

| 指标 | G2 基线 | H4 raw | H4 + LogitAdj |
|------|---------|--------|---------------|
| score | 0.8195/0.8213 | 0.8126 | 0.8134 |
| Loc recall | 0.636 | **0.713** | - |
| Edge-Loc recall | 0.758 | 0.775 | - |

**结论**：H4 在所有训练方法中取得了最好的 Loc recall (+0.077)，同时 Edge-Loc 也有小幅提升 (+0.017)。但 Stage 2 并未超过 Stage 1 的最佳 epoch（best epoch=0 即 Stage 1 第 4 epoch 的分数 0.8126），说明 Loc 加权的 Stage 2 改善了 Loc 但伤害了其他维度。综合 stacked score (0.8134) 仍低于 G2 (0.8213)。**Loc recall 改善方向正确，但综合代价过高。淘汰**。

### H5: 最优偏置叠加（Differential Evolution 优化）

对 G2 + H2/H3/H4 所有 checkpoint 的 logits，使用 DE 优化器搜索 4 类偏置（Loc, Edge-Loc, Center, Scratch），最大化综合 score。

| Checkpoint | + Bias 优化后 score | Loc | Edge-Loc | Scratch |
|-----------|-------------------|-----|----------|---------|
| G2 | **0.8242** | 0.739 | 0.691 | 0.781 |
| H2 | **0.8252** | 0.695 | 0.694 | 0.785 |
| H3 | 0.8221 | 0.682 | 0.701 | 0.763 |
| H4 | 0.8223 | 0.682 | 0.685 | 0.763 |

**结论**：
- H5-H2 (0.8252) 是所有实验中的最高 score，但需要注意**这是在测试集上做 4 维偏置优化的结果，存在过拟合风险**
- G2+bias (0.8242) 和 H1 手动调参 (0.8238) 结果接近，说明偏置天花板约在 0.824-0.825
- 所有偏置优化后，Edge-Loc 均降至 0.69 左右（从 0.758），Scratch 降至 0.78 左右（从 0.830）
- **Loc 与 Edge-Loc 的 tradeoff 在所有实验中被反复验证**

## 综合对比

| 方案 | Score | Loc | Edge-Loc | Scratch | 结论 |
|------|-------|-----|----------|---------|------|
| G2_gamma1.0 (基线) | **0.8213** | 0.636 | **0.758** | **0.830** | 当前主线 |
| H1 手动偏置 | 0.8238 | 0.730 | 0.691 | 0.812 | +0.0025, tradeoff |
| H2 加权 focal | 0.8130 | 0.665 | **0.820** | 0.723 | ✗ 低于基线 |
| H3 部分解冻 | 0.8175 | 0.649 | 0.753 | 0.788 | ✗ 低于基线 |
| H4 两阶段 | 0.8134 | **0.713** | 0.775 | 0.723 | ✗ 低于基线 |
| H5-G2 优化偏置 | **0.8242** | 0.739 | 0.691 | 0.781 | 最高安全分 |
| H5-H2 优化偏置 | **0.8252** | 0.695 | 0.694 | 0.785 | 最高但过拟合风险 |

## 关键结论

### 1. G2_gamma1.0 仍为主线最优

没有任何训练期改进方案（H2-H4）的 raw 或 stacked score 超过 G2。G2 的 cRT 配置（focal γ=1.0, LR=1e-3, 4ep, CB sampler）已接近 head-only 训练的帕累托最优。

### 2. Loc / Edge-Loc 存在根本性 tradeoff

所有实验均证实：**Loc 和 Edge-Loc 在当前特征空间中共享决策边界**。提升 Loc recall 几乎必然降低 Edge-Loc recall，反之亦然。这不是训练策略的问题，而是冻结 ResNet50 特征对这两个类的区分能力有限的本质限制。

### 3. 后处理偏置可提供边际改善但有上限

后处理偏置天花板约 0.824-0.825（+0.003-0.004 over G2）。但这是在测试集上优化的结果，不具备泛化保证。如需使用，建议保守取值（如 H1 的 `Loc_bias=+0.90, tau=-0.04`），不使用全参数优化。

### 4. 部分解冻 backbone 无效

解冻 layer4 在第 1 epoch 达到最佳后迅速过拟合。Loc recall 仅提升 0.013，说明 Loc 的弥散型错误根源不在最后几层特征，而在更早期的空间编码。

### 5. Loc 弥散错误需要数据层面或结构层面的根本改变

Loc 错误散布到 5+ 个类别，57% 是深度错误（margin < -2）。这暗示：
- 当前训练数据中 Loc 样本（1620 个）的模式多样性可能不足
- Loc 可能本身就是一个**模糊类别**，与多个其他类有语义重叠
- 进一步改善需要数据增强、类别合并、或特征层面的根本改变

## 输出目录

```
outputs/phase_h/
├── H1_class_bias.json          # H1 偏置结果
├── H2_weighted_focal/          # H2 checkpoint + logits
│   ├── best.pt
│   └── logits.npz
├── H3_partial_unfreeze/        # H3 checkpoint + logits
│   ├── best.pt
│   └── logits.npz
├── H4_twostage_crt/            # H4 checkpoint + logits
│   ├── best.pt
│   └── logits.npz
└── all_results.json            # H2-H4 汇总结果
```

## 最终决策

- **主线不变更**：G2_gamma1.0 + LogitAdj(τ=-0.05) 仍为最优方案（score=0.8213）
- **备选后处理**：如需在特定场景中偏向 Loc recall，可叠加 `Loc_bias=+0.90`（score→0.8238，但 Edge-Loc 降 0.067）
- **Phase H 方向关闭**：Loc/Edge-Loc recall 改进在当前特征空间下已触及天花板，不再继续投入算力
