# Phase C: Targeted Optimization

## Goal
Optimize CAFormer-S18 with ViT-specific tuning: lower LR, warmup, drop_path.

## Experiments

### C1: CAFormer lr=1e-4 + warmup 2ep (10 epochs)
- Config: C1_caformer_s18_lowlr_warmup.yaml
- Added cosine_warmup scheduler to registry (LinearLR warmup + CosineAnnealingLR)

**Results:**
- score=0.8054 (mF1=0.7745, mR=0.7946, mrm=0.7598, acc=0.9664)
- **BEATS D1 ResNet50+GeM (0.8017)!** First ViT to surpass CNN baseline.
- Val_acc trajectory: 0.25, 0.85, 0.88, 0.83, 0.94, 0.93, 0.92, 0.96, 0.95, 0.96
- Best val_acc=0.9664 at epoch 8

Per-class: Center=0.740, Donut=0.740, E-Loc=0.764, E-Ring=0.827, Loc=0.613,
           Near-full=0.937, Scratch=0.769
- Loc (0.613) still weak, but E-Ring massively improved (0.827 vs B3's 0.360)

### C3: CAFormer lr=1e-4 + warmup + drop_path=0.2 (10 epochs)
- Config: C3_caformer_s18_lowlr_warmup_dp02.yaml
- Tests whether stronger stochastic depth prevents overfitting

**Results:**
- score=0.7882 (mF1=0.7487, mR=0.7831, mrm=0.7462, acc=0.9581)
- Worse than C1. Drop_path caused training instability (val dip at ep6: 0.857)
- Drop_path regularization not beneficial when combined with CB sampler

## Key Findings
1. **lr=1e-4 + 2ep warmup is optimal for CAFormer on WM-811K**
2. Drop_path regularization hurts rather than helps (possibly conflicts with CB sampler noise)
3. CAFormer with proper LR schedule beats CNN baseline at only 10 epochs

## Decision
C1 is the best config. Proceed to Phase D with C1 as baseline.
