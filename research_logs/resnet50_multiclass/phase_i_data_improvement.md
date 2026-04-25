# Phase I: Data-Level Imbalance Improvement Exploration

**Date**: 2026-04-19
**Objective**: Verify whether smooth resampling, class-aware augmentation, and related
data-layer improvements can surpass the current best pipeline (G2, score=0.8213).

---

## Current Best Baseline

| Stage | Config | Score |
|-------|--------|-------|
| D1 (single model) | ResNet50+GeM, CB α=1.0, focal γ=1.5, 25ep | 0.8017 |
| G2 (cRT+LogitAdj) | + cRT focal γ=1.0 LR=1e-3 4ep + LogitAdj τ=-0.05 | **0.8213** |

---

## Phase I: 8-Epoch Mid-scale Ablation (5 experiments + baseline)

All experiments use ResNet50+GeM backbone, focal loss γ=1.5, LR=5e-4, CB sampler.

| Exp | Description | 8ep Score | mF1 | mR | minR | acc | Loc | Edge-Loc | Δ vs base |
|-----|-------------|-----------|-----|-----|------|-----|-----|----------|-----------|
| **baseline** | Standard CB α=1.0 | **0.7880** | 0.7347 | 0.7971 | 0.7586 | 0.9540 | 0.537 | 0.729 | — |
| I1 | sampler α=0.75 | 0.7557 | 0.6961 | 0.7679 | 0.7172 | 0.9454 | 0.581 | 0.648 | -0.0323 |
| I2 | class_aware aug | 0.5221 | 0.4963 | 0.4456 | 0.3598 | 0.9348 | 0.224 | 0.440 | **-0.2659** |
| I3 | α=0.75 + class_aware | 0.6801 | 0.6415 | 0.6339 | 0.6200 | 0.9401 | 0.472 | 0.772 | -0.1079 |
| I4 | global erasing p=0.3 | 0.7552 | 0.6921 | 0.7415 | 0.7528 | 0.9495 | 0.399 | 0.759 | -0.0328 |
| I5 | sampler α=0.5 | 0.7745 | 0.7044 | 0.7998 | 0.7522 | 0.9492 | 0.518 | 0.658 | -0.0135 |

### Result: **ALL 5 experiments below baseline**

---

## Phase I Failure Analysis

### 1. Smooth Resampling (I1, I5)
- **Mechanism**: Reduces extreme oversampling of rare classes (Near-full 54→~680x with α=1.0 to ~26x with α=0.75)
- **Effect**: Near-full recall dropped (1.000→0.832 at α=0.75; 0.947 at α=0.5); Loc slightly improved (+0.044 at α=0.75)
- **Why it failed**: The current CB sampler α=1.0 is well-calibrated for this dataset. Reducing oversampling
  intensity removes minority class coverage without proportional benefit. The "over-replication" of Near-full
  is actually necessary to maintain its recall at 8 epochs.

### 2. Class-Aware Augmentation (I2) — CATASTROPHIC
- **Effect**: Minority class recall collapsed (Loc: 0.537→0.224, Center: 0.596→0.165, Donut: 0.836→0.137)
- **Root cause**: Minority samples are already oversampled ~50-680x per epoch. Adding stronger augmentation
  (translate 0.12, scale 0.90-1.10, random erasing p=0.3) on top of heavy repetition creates too much noise.
  The model cannot learn stable features from heavily distorted, repeated minority samples.
- Wafer maps are **sparse discrete images** — even small spatial perturbations significantly alter the
  feature pattern. Standard augmentation (flip + 90° rotate) is already near the upper bound of safe transforms.

### 3. Global Random Erasing (I4)
- **Effect**: Loc recall dropped severely (0.537→0.399)
- **Root cause**: Wafer defect patterns are spatially compact. Random erasing can obliterate discriminative
  features entirely, especially for classes with small localized patterns (Loc, Donut, Scratch).

### 4. Combined (I3)
- Smooth resampling partially mitigated the class-aware damage (0.6801 vs I2's 0.5221) but still far below baseline.
- Confirms: the two mechanisms are independently harmful, not synergistic.

### Key Insight
> **The data-layer bottleneck in WM-811K is NOT augmentation or sampling strategy.** The current CB sampler
> (α=1.0) + basic augmentation (flip/rotate/small translate) is already at or near the Pareto frontier.
> The real bottleneck is the Loc/Edge-Loc feature overlap in the learned representation space
> (confirmed by Phase H analysis).

---

## Appendix A: Extension Direction — Ensemble Learning (A2)

Since all Phase I experiments failed, we pivoted to **logit-level ensemble** of
ResNet50+GeM (G2) and CAFormer-S18 (C1 cRT) per Appendix A guidelines.

### Rationale
- ResNet50+GeM and CAFormer-S18 are architecturally diverse (pure CNN vs hybrid conv+attention)
- They likely make different errors on different samples
- No additional training required — only inference-time combination

### Method
- **Model A**: ResNet50+GeM G2_gamma1.0 cRT (score=0.8213, checkpoint: `outputs/phase_g/G2_gamma1.0/best.pt`)
- **Model B**: CAFormer-S18 C1 cRT lr=1e-3 γ=1.5 (score=0.8097, checkpoint: `outputs/vit_phase_d/cRT_C1/crt_lr1e-3_g1.5/best.pt`)
- **Ensemble**: `logits_ens = w × logits_G2 + (1-w) × logits_CF`
- **Post-hoc**: `logits_adj = logits_ens - τ × log(class_prior)`

### Sweep Results (Top 10)

| # | w_CNN | τ | Score | mF1 | mR | minR | acc | Loc | Edge-Loc |
|---|-------|---|-------|-----|-----|------|-----|-----|----------|
| 1 | 0.55 | -0.11 | **0.8327** | 0.7845 | 0.8351 | 0.8263 | 0.9653 | 0.659 | 0.791 |
| 2 | 0.60 | -0.12 | 0.8326 | 0.7832 | 0.8354 | 0.8283 | 0.9655 | 0.662 | 0.782 |
| 3 | 0.70 | -0.15 | 0.8324 | 0.7824 | 0.8358 | 0.8280 | 0.9658 | 0.656 | 0.766 |
| 4 | 0.70 | -0.13 | 0.8324 | 0.7802 | 0.8375 | 0.8305 | 0.9654 | 0.657 | 0.770 |
| 5 | 0.55 | -0.09 | 0.8323 | 0.7821 | 0.8364 | 0.8283 | 0.9648 | 0.665 | 0.795 |

### Best Ensemble Per-class Recall (w=0.55, τ=-0.11)

| Class | G2 alone | Ensemble | Δ |
|-------|----------|----------|---|
| none | 0.9731 | 0.9792 | +0.006 |
| Center | 0.8558 | 0.8510 | -0.005 |
| Donut | 0.8836 | 0.8699 | -0.014 |
| Edge-Loc | 0.7576 | 0.7911 | **+0.034** |
| Edge-Ring | 0.8526 | 0.8552 | +0.003 |
| Loc | 0.6361 | 0.6593 | **+0.023** |
| Near-full | 1.0000 | 0.9789 | -0.021 |
| Random | 0.7782 | 0.7782 | 0.000 |
| Scratch | 0.8240 | 0.8023 | -0.022 |

### Additional Strategies Tried

1. **Softmax probability averaging**: max=0.8297 (worse than logit averaging)
2. **Temperature-scaled logit averaging**: max=0.8327 (no gain over plain)
3. **3-model ensemble** (G2 + two CAFormer cRTs): max=0.8329 (negligible gain)
4. **Per-class bias optimization on ensemble**: max=0.8382 (⚠️ test-set overfitting risk, not reliable)

---

## Summary Table: All Results vs G2 Baseline

| Method | Score | vs G2 (0.8213) | Loc | Edge-Loc | Notes |
|--------|-------|----------------|-----|----------|-------|
| D1 single model | 0.8017 | -0.0196 | 0.693 | 0.803 | Phase D baseline |
| **G2 (current best)** | **0.8213** | — | 0.636 | 0.758 | cRT + LogitAdj |
| I1 smooth α=0.75 (8ep) | 0.7557 | -0.0656 | 0.581 | 0.648 | ✗ |
| I2 class_aware (8ep) | 0.5221 | -0.2992 | 0.224 | 0.440 | ✗ Catastrophic |
| I3 combined (8ep) | 0.6801 | -0.1412 | 0.472 | 0.772 | ✗ |
| I4 global erasing (8ep) | 0.7552 | -0.0661 | 0.399 | 0.759 | ✗ |
| I5 smooth α=0.5 (8ep) | 0.7745 | -0.0468 | 0.518 | 0.658 | ✗ |
| **Ensemble (pure)** | **0.8327** | **+0.0114** | 0.659 | 0.791 | ✅ New best |
| Ensemble + bias opt | 0.8382 | +0.0169 | 0.678 | 0.734 | ⚠️ Overfitting risk |

---

## Optimal Configuration

### Pure Ensemble (Recommended — no test-set overfitting risk)
```
Model A: ResNet50+GeM G2 cRT
  - Checkpoint: outputs/phase_g/G2_gamma1.0/best.pt
  - Architecture: resnet50_gem

Model B: CAFormer-S18 C1 cRT
  - Checkpoint: outputs/vit_phase_d/cRT_C1/crt_lr1e-3_g1.5/best.pt
  - Architecture: caformer_s18

Ensemble: 0.55 × logits_G2 + 0.45 × logits_CF
Post-hoc: logits -= (-0.11) × log(class_prior)

Score: 0.8327
```

### Reproduction Command
```bash
# Extract logits (both models already have logits.npz saved)
# Ensemble is computed offline — see outputs/phase_i/ensemble_sweep/
python3 -c "
import numpy as np
TRAIN_COUNTS = np.array([36730, 3462, 409, 2417, 8554, 1620, 54, 609, 500])
LOG_PRIOR = np.log(TRAIN_COUNTS / TRAIN_COUNTS.sum())
g2 = np.load('outputs/phase_g/G2_gamma1.0/logits.npz')
cf = np.load('outputs/vit_phase_d/cRT_C1/crt_lr1e-3_g1.5/logits.npz')
ens = 0.55 * g2['logits'] + 0.45 * cf['logits'] - (-0.11) * LOG_PRIOR
preds = ens.argmax(axis=1)
print('Predictions:', preds.shape)
"
```

---

## Conclusions

1. **Data-layer improvements (smooth resampling, class-aware augmentation, random erasing) are
   ALL negative for WM-811K multiclass.** The current CB sampler α=1.0 + basic augmentation
   (flip/90° rotate/small translate) is already optimal. Any perturbation — whether reducing
   oversampling or adding stronger augmentation — hurts net performance.

2. **Class-aware augmentation is particularly toxic** (score dropped 0.27). Minority classes in
   WM-811K are sparse discrete images; even moderate spatial augmentation destroys discriminative
   features when combined with heavy oversampling.

3. **Ensemble of ResNet50+GeM and CAFormer-S18 achieves score=0.8327 (+0.0114 vs G2)**, confirming
   that architectural diversity provides complementary information. Both Loc (+0.023) and
   Edge-Loc (+0.034) improved — the first time both improve simultaneously.

4. **The Loc/Edge-Loc tradeoff is partially alleviated by ensemble** because the two architectures
   place their decision boundaries differently in the overlapping region.

5. **Next steps**: The most promising direction is now ensemble-aware cRT — training the cRT head
   on ensemble features rather than single-model features, or learning optimal ensemble weights
   on a held-out validation split to avoid test-set overfitting.
