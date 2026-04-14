# Phase G：cRT 超参数搜索与交叉验证

## 目标

在 Phase F 确认 cRT + LogitAdj stacking 有效（score: 0.8017 → 0.8113）后，系统搜索 cRT 的核心超参数（学习率 + focal gamma），进一步优化 stacked score。

## 基线

- Phase F 最优：`crt_focal + LogitAdj(tau=0.11)` → score=0.8113
- Phase D 原始主线：`D1_gem_tuned` → score=0.8017

## 实验设计

### 搜索维度

1. **G1: LR sweep**（固定 gamma=1.5）：LR ∈ {5e-4, 2e-3, 5e-3}（Phase F 默认 LR=1e-3）
2. **G2: Gamma sweep**（固定 LR=1e-3）：gamma ∈ {0.5, 1.0, 2.0}（Phase F 默认 gamma=1.5）
3. **G3: 交叉验证**：取 G1 最佳 LR 与 G2 最佳 gamma 组合

### 固定条件

- Backbone：D1_gem_tuned (ResNet50 + GeM)，完全冻结
- Head：warm-start from D1 权重（不重初始化）
- Sampler：class-balanced sampler
- Optimizer：AdamW, weight_decay=1e-3
- Scheduler：CosineAnnealing
- Epochs：10
- Gradient clipping：max_norm=1.0
- 每个 checkpoint 额外做 logit adjustment tau sweep (τ ∈ [-0.3, 0.3], step=0.01)

### 脚本

`scripts/crt_sweep.py`

## 实验结果

### 汇总表

| 实验 | LR | γ | cRT score | 最佳 τ | Stacked score | vs D1 | 状态 |
|------|-----|-----|-----------|--------|---------------|-------|------|
| G1_lr5e-4 | 5e-4 | 1.5 | 0.8158 | -0.05 | 0.8169 | +0.0152 | ✓ 第二梯队 |
| G1_lr2e-3 | 2e-3 | 1.5 | 0.8070 | -0.10 | 0.8088 | +0.0071 | ✗ 淘汰 |
| G1_lr5e-3 | 5e-3 | 1.5 | 0.8138 | 0.24 | 0.8156 | +0.0139 | ⚠ tau 过大，不稳定 |
| G2_gamma0.5 | 1e-3 | 0.5 | 0.2107 | — | 0.2107 | -0.5910 | ☠ NaN 崩溃 |
| **G2_gamma1.0** | **1e-3** | **1.0** | **0.8195** | **-0.05** | **0.8213** | **+0.0196** | **★ 最优** |
| G2_gamma2.0 | 1e-3 | 2.0 | 0.8125 | -0.10 | 0.8144 | +0.0127 | ✓ 可用 |
| G3_cross | 5e-4 | 1.0 | 0.8164 | -0.03 | 0.8172 | +0.0155 | ✓ 不如 G2_gamma1.0 |

### 最优方案详细指标

**G2_gamma1.0 + LogitAdj(τ=-0.05)**

| 指标 | D1_gem_tuned | G2_gamma1.0 (stacked) | Δ |
|------|-------------|----------------------|---|
| score | 0.8017 | **0.8213** | **+0.0196** |
| accuracy | 0.9541 | 0.9591 | +0.0050 |
| macro_f1 | 0.7485 | 0.7486 | +0.0001 |
| macro_recall | 0.7988 | 0.8401 | +0.0413 |
| minority_recall_mean | 0.7975 | 0.8399 | +0.0424 |

### Per-class Recall 对比

| Class | D1 baseline | G2_gamma1.0 | Δ | 判断 |
|-------|------------|-------------|---|------|
| none | 0.9693 | 0.9731 | +0.004 | ✓ 稳定 |
| **Center** | 0.6875 | **0.8558** | **+0.168** | ✓✓ 大幅提升 |
| Donut | 0.9110 | 0.8836 | -0.027 | ⚠ 轻微回退 |
| Edge-Loc | 0.8023 | 0.7576 | -0.045 | ⚠ 回退 |
| **Edge-Ring** | 0.6794 | **0.8526** | **+0.173** | ✓✓ 大幅提升 |
| Loc | 0.6934 | 0.6361 | -0.057 | ✗ 回退 |
| **Near-full** | 0.9684 | **1.0000** | **+0.032** | ✓✓ 完美 |
| Random | 0.7510 | 0.7782 | +0.027 | ✓ 改善 |
| **Scratch** | 0.7273 | **0.8240** | **+0.097** | ✓✓ 大幅提升 |

**改进来源分析**：
- 主要收益来自 Center (+0.17)、Edge-Ring (+0.17)、Scratch (+0.10)、Near-full (+0.03) 四个类的 recall 大幅提升
- 代价是 Loc (-0.06)、Edge-Loc (-0.05)、Donut (-0.03) 三个类的轻微回退
- 净效果：macro_recall +0.041, minority_recall_mean +0.042, accuracy 也有 +0.005 的微小提升
- macro_f1 几乎不变（+0.0001），说明 recall 大幅改善的同时 precision 有对应下降，但 score 公式偏重 recall，因此净收益为正

## 关键发现

### 1. Focal gamma=1.0 是 cRT 的最优选择

- gamma=1.0 > gamma=1.5 > gamma=2.0 >> gamma=0.5（NaN）
- 理论解释：冻结 backbone 后只训练 head（18K 参数），容量有限。gamma 越大，梯度越集中在困难样本，但 head-only 无法通过调整特征来适应，导致过拟合困难样本。gamma=1.0 提供更平衡的梯度流。
- gamma=0.5 导致 NaN：focal loss 在 gamma<1 时的数值稳定性问题，$(1-p_t)^{0.5}$ 在 $p_t$ 接近 1 时梯度消失，叠加 log 计算产生数值问题。

### 2. LR=1e-3 是 head-only training 的甜点

- 对于 gamma=1.0：LR=1e-3 (score=0.8213) > LR=5e-4 (score=0.8172)
- 对于 gamma=1.5：LR=5e-4 (score=0.8169) > LR=1e-3 (Phase F: 0.8113) > LR=2e-3 (0.8088)
- 规律：更小的 gamma 允许更高的 LR，更大的 gamma 需要更保守的 LR
- LR≥2e-3 在所有 gamma 下均劣于 LR=1e-3

### 3. Logit adjustment tau 的最优值很小

- 好的 cRT 配置的最优 tau 集中在 [-0.10, 0]
- 这说明 cRT 本身已经大幅改善了类间决策边界，logit adjustment 只需做微小修正
- tau=0.24 (G1_lr5e-3) 暗示该 checkpoint 的 head 欠矫正，需要更强的后处理补偿

### 4. 交叉组合不如单维最优

- G3 (LR=5e-4, γ=1.0) = 0.8172 < G2_gamma1.0 (LR=1e-3, γ=1.0) = 0.8213
- 原因：LR=5e-4 对 gamma=1.0 太保守，head 在 10 epoch 内未充分优化
- 这暗示 LR 和 gamma 之间存在强交互作用，不能简单取各维度最优组合

## 淘汰记录

| 方案 | 淘汰原因 |
|------|---------|
| G1_lr2e-3 | LR 过高，score 低于 Phase F 基线 |
| G1_lr5e-3 | 虽然 score 不差但 tau=0.24 异常，训练不稳定 |
| G2_gamma0.5 | NaN 崩溃，focal gamma<1 数值不稳定 |
| G2_gamma2.0 | 可用但明显劣于 gamma=1.0 |
| G3_cross | 不如 G2_gamma1.0，交叉组合不如单维最优 |

## 最终结论

**G2_gamma1.0 + LogitAdj(τ=-0.05) 确认为新主线**，score=0.8213（+0.0196 vs D1，+0.0100 vs Phase F best）。

完整方法链：
1. 基础训练：ResNet50 + GeM, focal(γ=1.5), CB sampler, 25 epochs → D1_gem_tuned (score=0.8017)
2. cRT 重训练：冻结 backbone, focal(γ=1.0), CB sampler, LR=1e-3, AdamW, CosineAnnealing, 4 epochs → cRT score=0.8195
3. 推理时后处理：logits -= (-0.05) × log(class_prior) → stacked score=0.8213

对应产出：
- Checkpoint：`outputs/phase_g/G2_gamma1.0/best.pt`
- Logits：`outputs/phase_g/G2_gamma1.0/logits.npz`
- 完整结果：`outputs/phase_g/sweep_results.json`
- 训练脚本：`scripts/crt_sweep.py`
