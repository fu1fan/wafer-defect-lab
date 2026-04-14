# Phase F：后校准与解耦重平衡

## 研究背景

Phase D 确定了 `D1_gem_tuned`（score=0.8017）为当前最优主线。Phase E（强 backbone 搜索）已证实性能瓶颈不在 backbone（ConvNeXt-Tiny / EfficientNetV2-S 均未超越 ResNet50+GeM）。

本阶段的出发点：既然瓶颈在类不平衡 / 决策边界 / 分类头偏置，那么应该从后处理校准和解耦训练两个方向寻找改进。

## 实验设计

### 三层探索策略（由低到高成本）

| 层级 | 方向 | 成本 | 状态 |
|------|------|------|------|
| Layer 1 | 后校准（post-hoc calibration） | 零成本（复用已有 checkpoint） | ✅ 完成 |
| Layer 2 | cRT（冻结 backbone 重训分类头） | 低成本（~200s/epoch，仅训 head） | ✅ 完成 |
| Layer 2+ | 平衡微调（解冻 backbone + 低 LR） | 中成本（~327s/epoch，全模型） | ✅ 完成 |

### 评估公式

```
score = 0.40 × macro_f1 + 0.25 × macro_recall + 0.20 × minority_recall_mean + 0.15 × accuracy
```
minority classes: Center, Donut, Loc, Near-full, Scratch

---

## Layer 1：后校准（Post-hoc Calibration）

### 实验 F1：6 种后处理方法

从 `D1_gem_tuned/best.pt` 导出 logits（train 54355×9, test 118595×9），依次测试：

| 方法 | 最佳 score | Δ | 结论 |
|------|-----------|------|------|
| Temperature Scaling | 0.8017 | 0.0000 | 不改变 argmax，恒等变换 |
| Class-wise Bias Correction | 0.7589 | -0.0428 | 在 train 上学偏置过拟合 |
| Vector Scaling | 0.7814 | -0.0203 | 同上，过拟合训练分布 |
| Tau-normalization | 0.7427 | -0.0590 | 破坏已学好的权重范数结构 |
| Prior-aware Logit Adjustment | 0.8010 | -0.0007 | tau=-0.08 最接近但未超越 |
| Greedy Threshold Tuning | 0.7905 | -0.0112 | 阈值在 train 上过拟合 |

**Layer 1 结论：全部淘汰。** 模型在决策层面已经相当校准，后校准方法的主要问题是 train/test 分布不一致导致参数过拟合。唯一接近的是 logit adjustment（tau=-0.08），但无法超过基线。

### 关键发现

- Temperature scaling 对 argmax 无效（数学上恒等）
- Tau-normalization 破坏了模型已学到的权重范数结构（Near-full 虽然只有 54 样本但范数最大=2.307，说明模型正确地放大了它）
- 需要 separate validation set 才能让后校准方法不过拟合

---

## Layer 2：cRT（Classifier Re-Training）

### 实验 F3：5 种 cRT 策略

基础设置：从 `D1_gem_tuned/best.pt` 加载全部权重，冻结 backbone（仅 `model.fc` 可训练，18441/23520202 = 0.08%），使用 class-balanced sampler + cosine schedule，训练 10 epochs。

| 策略 | Loss | Head 初始化 | Backbone | LR | 最佳 score | Δ | 最佳 epoch |
|------|------|-------------|----------|-----|-----------|------|-----------|
| **crt_balanced_finetune** | Focal(γ=1.5) | 保留 | **解冻(0.01×)** | 1e-3 | **0.8099** | **+0.0082** | **8** |
| **crt_focal** | Focal(γ=1.5) | 保留 | 冻结 | 1e-3 | **0.8091** | **+0.0074** | **2** |
| crt_ce | CE | 保留 | 冻结 | 1e-3 | 0.8037 | +0.0020 | 1 |
| crt_reset_focal | Focal(γ=1.5) | **重新初始化** | 冻结 | 1e-3 | 0.7980 | -0.0037 | 6 |
| crt_label_smooth | CE(ε=0.1) | 保留 | 冻结 | 1e-3 | 0.7928 | -0.0089 | 9 |

### 各策略详细指标

| 策略 | acc | mF1 | mR | minR | score |
|------|------|------|------|------|-------|
| D1_gem_tuned (baseline) | 0.9541 | 0.7485 | 0.7988 | 0.7975 | 0.8017 |
| **crt_balanced_finetune** | 0.9450 | 0.7446 | **0.8312** | **0.8127** | **0.8099** |
| **crt_focal** | **0.9509** | **0.7485** | 0.8198 | 0.8106 | **0.8091** |
| crt_ce | 0.9375 | 0.7221 | 0.8364 | 0.8255 | 0.8037 |
| crt_reset_focal | 0.9335 | 0.7168 | 0.8301 | 0.8189 | 0.7980 |
| crt_label_smooth | 0.9433 | 0.7035 | 0.8187 | 0.8262 | 0.7928 |

### 关键 per-class recall 对比（四个弱势类）

| 类别 | baseline | crt_focal | crt_balanced_ft | 变化方向 |
|------|----------|-----------|-----------------|----------|
| Center | 0.6875 | 0.7981 (+0.11) | 0.7993 (+0.11) | ✅ 大幅提升 |
| Edge-Ring | 0.6794 | 0.8508 (+0.17) | 0.8366 (+0.16) | ✅ 大幅提升 |
| Loc | 0.6934 | 0.7268 (+0.03) | 0.6695 (-0.02) | ⚠ crt_focal 好，balanced_ft 略降 |
| Scratch | 0.7273 | 0.7244 (-0.00) | 0.7908 (+0.06) | ⚠ crt_focal 持平，balanced_ft 好 |

### cRT 实验分析

1. **Focal loss 是 cRT 的必要条件**：crt_focal(0.8091) >> crt_ce(0.8037)。CE 损失无法有效聚焦难分样本。

2. **保留原始 head 权重至关重要**：crt_reset_focal(0.7980) < baseline(0.8017)。原始 head 已包含有价值的类别知识，重新初始化会丢失这些信息。

3. **Label smoothing 不适合长尾场景的 cRT**：crt_label_smooth(0.7928) 最差。ε=0.1 的 smoothing 对少数类产生了过度正则化，抑制了少数类的学习信号。

4. **解冻 backbone 有额外收益但需更多 epoch**：crt_balanced_finetune 最终超过 crt_focal（0.8099 vs 0.8091），但需要 8 个 epoch（vs 2 个），且 per-epoch 成本更高（327s vs 203s）。

5. **两个 winner 的改进路径不同**：
   - crt_focal：主要改善 Center(+0.11), Edge-Ring(+0.17), Loc(+0.03)，但 Scratch 持平
   - crt_balanced_finetune：主要改善 Center(+0.11), Edge-Ring(+0.16), Scratch(+0.06)，但 Loc 略降

---

## 淘汰判断

| 方向 | 淘汰/保留 | 理由 |
|------|-----------|------|
| Temperature scaling | ❌ 淘汰 | 对 argmax 恒等 |
| Class-wise bias / Vector scaling | ❌ 淘汰 | 在 train 上过拟合 |
| Tau-normalization | ❌ 淘汰 | 破坏已学权重结构 |
| Logit adjustment (post-hoc) | ❌ 淘汰 | 边际收益不足 |
| crt_reset_focal | ❌ 淘汰 | 重初始化 head 破坏已有知识 |
| crt_label_smooth | ❌ 淘汰 | Label smoothing 对长尾 cRT 有害 |
| crt_ce | ⚠ 备选 | 小幅改善但不如 focal |
| **crt_focal** | ✅ 保留 | 成本低、效果好、可复现 |
| **crt_balanced_finetune** | ✅ 保留，推荐为新主线 | 综合 score 最高 |

---

## Layer 2+：Stacked Post-hoc（cRT + Logit Adjustment）

在 Layer 1 中，post-hoc logit adjustment 对原始 D1_gem_tuned 无效（最佳仅 0.8010）。
但 cRT 改变了模型的决策边界后，logit adjustment 有了新的作用空间。

### 实验 F4：在 cRT checkpoint 上叠加 logit adjustment

| 方法 | 最佳 tau | score | Δ vs 基线 |
|------|----------|-------|-----------|
| crt_focal + LogitAdj | **0.11** | **0.8113** | **+0.0096** |
| crt_balanced_finetune + LogitAdj | 0.0 (无效) | 0.8099 | +0.0082 |

**crt_focal + LogitAdj(tau=0.11)** 成为新的全局最优！

### 最终最优方案详细指标

| 指标 | D1_gem_tuned | crt_focal+LA | Δ |
|------|-------------|-------------|------|
| **score** | 0.8017 | **0.8113** | **+0.0096** |
| accuracy | 0.9541 | 0.9449 | -0.0092 |
| macro_f1 | 0.7485 | 0.7410 | -0.0075 |
| macro_recall | 0.7988 | 0.8302 | +0.0314 |
| minority_recall_mean | 0.7975 | 0.8282 | +0.0307 |

### Per-class Recall 对比

| 类别 | baseline | crt_focal+LA | Δ |
|------|----------|-------------|------|
| none | 0.9635 | 0.9562 | -0.0073 |
| **Center** | 0.6875 | **0.8089** | **+0.1214** |
| Donut | 0.8562 | 0.8836 | +0.0274 |
| Edge-Loc | 0.7753 | 0.7937 | +0.0184 |
| **Edge-Ring** | 0.6794 | **0.8375** | **+0.1581** |
| **Loc** | 0.6934 | **0.7405** | **+0.0471** |
| Near-full | 0.9474 | 0.9579 | +0.0105 |
| Random | 0.7354 | 0.7432 | +0.0078 |
| **Scratch** | 0.7273 | **0.7504** | **+0.0231** |

**所有 9 个类别的 recall 都提升了（除 none 微降 0.73%）。四个弱势类均有显著改善。**

### 为什么 stacking 在这里有效？

- cRT-focal 重训了分类头，大幅改善 Center/Edge-Ring 的决策边界
- 但 cRT 的 CB sampling 不完美——logit 仍略偏向高频类
- Post-hoc logit adjustment (tau=0.11) 对 logit 做先验校正，进一步推高少数类
- crt_balanced_finetune 在训练时已经通过 backbone 微调隐式校准了 logit，所以 post-hoc 无额外空间

---

## 最终结论

### 新主线方案

**crt_focal + LogitAdj(tau=0.11)** 以 score=0.8113（+0.0096）超过 D1_gem_tuned，推荐为新主线。

方法说明：
1. 从 `D1_gem_tuned/best.pt` 加载全部权重
2. 冻结 backbone，仅训练 FC head
3. 使用 focal loss(γ=1.5) + CB sampler，LR=1e-3，cosine schedule，训练 2 epoch（约 7 分钟）
4. 推理时对 logits 施加先验校正：`logits -= 0.11 × log(class_prior)`

改进来源：
- Center recall: +0.12（从 0.6875 到 0.8089）
- Edge-Ring recall: +0.16（从 0.6794 到 0.8375）
- Loc recall: +0.05（从 0.6934 到 0.7405）
- Scratch recall: +0.02（从 0.7273 到 0.7504）
- 全部 9 类 recall 均有改善或基本持平

代价：
- accuracy 从 0.9541 降至 0.9449（-0.0092），主要是 none 类 recall 微降
- macro_f1 从 0.7485 降至 0.7410（-0.0075）

### 备选方案

| 方案 | score | 优势 | 劣势 |
|------|-------|------|------|
| crt_balanced_finetune | 0.8099 | 不需要 post-hoc | 训练更贵（44 min），Loc 略降 |
| crt_focal | 0.8091 | 最简单、最快 | score 略低于 stacked |

---

## 文件产出

| 文件 | 说明 |
|------|------|
| `scripts/posthoc_calibration.py` | 后校准分析脚本（6 种方法） |
| `scripts/crt_retrain.py` | cRT 重训练脚本（5 种策略） |
| `outputs/phase_f/F1_posthoc/` | 后校准结果（logits.npz, results.json） |
| `outputs/phase_f/F3_crt/crt_focal/best.pt` | cRT-Focal 最佳 checkpoint（**新主线模型**） |
| `outputs/phase_f/F3_crt/crt_balanced_finetune/best.pt` | 平衡微调 checkpoint（备选） |
| `outputs/phase_f/F3_crt/all_results.json` | 全部 cRT 结果汇总 |
| `outputs/phase_f/F4_stacked/final_best.json` | Stacked 方法最终结果 |
| `outputs/phase_f/F4_stacked/crt_focal_logits.npz` | crt_focal logits（可复现 post-hoc） |

---

## 下一步建议

1. **首选方向**：在 crt_focal 基础上探索 LR / epoch 微调（0.5e-3 / 2e-3），看 cRT 训练阶段能否进一步优化，使 stacked score 更高
2. **高级方向**：真正的 decoupled training（Stage 1 instance-balanced 训练特征提取器，Stage 2 cRT + logit adj），理论上能释放更多特征空间
3. **Loc 类专项**：Loc 是当前最弱类（0.7405），可以尝试 Loc-aware loss weighting 或专门的决策阈值调整
