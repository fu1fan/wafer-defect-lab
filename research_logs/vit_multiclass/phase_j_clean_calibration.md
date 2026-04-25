# Phase J: Clean Calibration Check for CNN + ViT Ensemble

## Objective

Estimate the real gain of the current best ViT line in an ensemble setting,
without selecting ensemble parameters on the same samples used for evaluation.

The target pipeline is:

- CNN: `outputs/phase_g/G2_gamma1.0/logits.npz`
- ViT: `outputs/vit_phase_i/I5_smooth_resample_050/logits.npz`
- Ensemble form: `logits = w_cnn * logits_cnn + (1 - w_cnn) * logits_vit - tau * log(class_prior)`

## Important limitation

The existing CNN and ViT checkpoints were trained before a dedicated
calibration split was reserved. Therefore this pass cannot be a fully clean
train/calibration/test protocol.

Instead, this phase uses a disjoint official-Test split protocol:

1. Split the Test logits by class into calibration and evaluation partitions.
2. Select `w_cnn` and `tau` only on the calibration partition.
3. Evaluate once on the disjoint evaluation partition.
4. Repeat over 20 deterministic stratified splits.

This estimates how much optimism came from sweeping `w/tau` on the full Test
set. It is stronger evidence than full-Test sweep, but the final publication-
grade answer would still require retraining both models with a true held-out
calibration split.

## Script

```bash
python scripts/clean_ensemble_calibration.py \
  --seeds 20 \
  --output outputs/vit_phase_i/clean_calibration/results.json

python scripts/clean_ensemble_calibration.py \
  --seeds 20 \
  --eval-frac 0.7 \
  --output outputs/vit_phase_i/clean_calibration/results_eval70.json
```

## Full-Test Reference

| Method | Score |
|---|---:|
| ViT I5 raw | 0.8058 |
| CNN G2 + LogitAdj(tau=-0.05) | 0.8213 |
| CNN+ViT old fixed ensemble (w=0.50, tau=0.14) | 0.8343 |
| CNN+ViT full-Test sweep in this script (w=0.51, tau=0.11) | 0.8343 |

## Split Calibration Results

### 50% calibration / 50% evaluation

20 repeated stratified splits:

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| Ensemble score | 0.8306 | 0.0058 | 0.8202 | 0.8299 | 0.8409 |
| CNN G2 score | 0.8204 | 0.0047 | 0.8084 | 0.8197 | 0.8275 |
| ViT I5 score | 0.8049 | 0.0052 | 0.7947 | 0.8049 | 0.8162 |
| Delta vs CNN G2 | +0.0102 | 0.0039 | +0.0029 | +0.0094 | +0.0193 |
| Delta vs ViT I5 | +0.0257 | 0.0036 | +0.0211 | +0.0250 | +0.0326 |

Win rate:

- Ensemble vs CNN G2: 20 / 20
- Ensemble vs ViT I5: 20 / 20

Selected parameters:

- `w_cnn`: mean 0.434, median 0.415, range 0.31-0.54
- `tau`: mean 0.110, median 0.110, range 0.00-0.27

Key recalls:

- Loc recall: mean 0.6330
- Edge-Loc recall: mean 0.8210

### 30% calibration / 70% evaluation

20 repeated stratified splits:

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| Ensemble score | 0.8300 | 0.0026 | 0.8252 | 0.8296 | 0.8350 |
| CNN G2 score | 0.8205 | 0.0024 | 0.8160 | 0.8207 | 0.8254 |
| ViT I5 score | 0.8052 | 0.0033 | 0.7977 | 0.8066 | 0.8104 |
| Delta vs CNN G2 | +0.0095 | 0.0028 | +0.0038 | +0.0096 | +0.0138 |
| Delta vs ViT I5 | +0.0248 | 0.0023 | +0.0193 | +0.0249 | +0.0284 |

Win rate:

- Ensemble vs CNN G2: 20 / 20
- Ensemble vs ViT I5: 20 / 20

Selected parameters:

- `w_cnn`: mean 0.462, median 0.500, range 0.30-0.69
- `tau`: mean 0.098, median 0.115, range -0.22-0.28

Key recalls:

- Loc recall: mean 0.6313
- Edge-Loc recall: mean 0.8156

## Interpretation

The full-Test sweep score of 0.8343 is slightly optimistic, but the gain does
not disappear under disjoint calibration/evaluation splits.

The realistic level of the current ViT-assisted ensemble is approximately:

```text
score ~= 0.830
gain over CNN G2 ~= +0.009 to +0.010
gain over ViT I5 raw ~= +0.025
```

The improvement is robust across repeated stratified splits. The worst observed
delta over CNN G2 was still positive:

- 50/50 protocol: +0.0029
- 30/70 protocol: +0.0038

This supports treating the ViT line as genuinely useful through architectural
complementarity, not as a standalone replacement for CNN G2.

## Decision

Promote `CNN G2 + ViT I5 ensemble` from "test-sweep candidate" to "credible
next main candidate", with one condition:

- before claiming a final benchmark, retrain CNN and ViT with a reserved
  calibration split, select `w/tau` on that split, and evaluate once on the
  untouched Test split.

Do not spend the next cycle on more data-layer augmentation for ResNet50. The
best next technical step is either:

1. implement a train/calibration/test split mode and rerun both models, or
2. implement a reusable ensemble evaluator/exporter so this candidate can be
   reproduced without ad hoc scripts.
