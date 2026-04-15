# Phase D: Full Training and Final Comparison

## Goal
Full 25-epoch training, cRT + LogitAdj stacking, final comparison vs CNN baseline.

## D_V1: CAFormer 25 epochs (C1 config extended)
- Config: D_V1_caformer_full_25ep.yaml
- Same as C1 but 25 epochs instead of 10

**Results:**
- score=0.7973 (mF1=0.7778, mR=0.7791, mrm=0.7328, acc=0.9653)
- **Worse than C1 10ep (0.8054)!** More training hurts minority recall.
- Loc: 0.595 (down from C1 0.613), Scratch: 0.683 (down from 0.769)
- Classic long-tail overfitting: majority classes improve, minorities degrade

**Lesson: More epochs with cosine schedule do NOT help CAFormer on imbalanced data.**
Early stopping around 8-10 epochs is optimal.

## cRT Sweep on C1 (10ep checkpoint)

Frozen backbone, retrain FC head with CB sampler + focal loss.

| Config | cRT score | tau | Stacked | Loc | Scratch |
|---|---|---|---|---|---|
| lr=1e-3, gamma=1.0 | 0.8060 | 0.30 | 0.8081 | 0.629 | 0.788 |
| lr=1e-3, gamma=1.5 | **0.8096** | 0.12 | **0.8097** | **0.676** | 0.798 |
| lr=5e-4, gamma=1.0 | 0.8022 | -0.07 | 0.8041 | 0.625 | 0.704 |
| lr=2e-3, gamma=1.0 | 0.8033 | 0.14 | 0.8039 | 0.631 | 0.810 |

Best: crt_lr1e-3_g1.5, cRT score=0.8096, +LogitAdj(tau=0.12) = 0.8097

## cRT Sweep on D_V1 (25ep checkpoint)

| Config | cRT score | tau | Stacked |
|---|---|---|---|
| lr=1e-3, gamma=1.0 | 0.7949 | 0.27 | 0.7991 |
| lr=1e-3, gamma=1.5 | 0.7991 | -0.18 | 0.8022 |
| lr=5e-4, gamma=1.0 | 0.7976 | 0.29 | 0.7999 |
| lr=2e-3, gamma=1.0 | 0.8007 | 0.17 | 0.8023 |

Best: 0.8023. D_V1 backbone is worse for cRT than C1.

## Final Comparison

| Config | score | vs D1 | vs G2 |
|---|---|---|---|
| D1 ResNet50+GeM (25ep) | 0.8017 | - | -0.0196 |
| G2 D1+cRT+LogitAdj (best CNN) | **0.8213** | +0.0196 | - |
| C1 CAFormer (10ep, no cRT) | 0.8054 | **+0.0037** | -0.0159 |
| C1+cRT+LogitAdj (best ViT) | 0.8097 | +0.0080 | -0.0116 |

## Verdict

**CAFormer-S18 is a viable alternative to ResNet50+GeM:**
- Single model: CAFormer (0.8054) > ResNet50+GeM (0.8017) by +0.0037
- With full pipeline: CNN (0.8213) > ViT (0.8097) by 0.0116

The gap is in cRT effectiveness, not backbone quality. CNN cRT gains +0.0196
while ViT cRT gains only +0.0043. Likely because CAFormer FC head has only
4617 params (Linear 512 to 9), leaving less room for cRT improvement.
