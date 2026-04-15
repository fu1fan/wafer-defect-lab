# Phase B: Mid-scale Screening (8 epochs)

## Goal
Screen all 3 candidates at 8 epochs with identical recipe (CB sampler + focal gamma=1.5,
lr=5e-4, wd=1e-3) to find the best architecture direction.

## Results

| Config | score | mF1 | mR | mrm | acc | Loc | Scratch | E-Ring |
|---|---|---|---|---|---|---|---|---|
| B1 DeiT3-S/16 | 0.6510 | 0.5955 | 0.6462 | 0.5570 | 0.9323 | 0.293 | 0.300 | 0.399 |
| B2 EVA02-S/14 | 0.7126 | 0.6322 | 0.7323 | 0.6856 | 0.9304 | 0.472 | 0.410 | 0.693 |
| B3 CAFormer-S18 | 0.7680 | 0.6872 | 0.7490 | 0.8136 | 0.9540 | 0.665 | 0.677 | 0.360 |
| D1 ResNet50+GeM (ref) | 0.8017 | 0.7485 | 0.7988 | 0.7975 | 0.9541 | - | - | - |

## Training Observations

### B1 DeiT3-S/16 (~5.3 min/epoch)
- Val_acc peaked at epoch 1 (0.9323), then declined
- Severe underperformance on Loc (0.293) and Scratch (0.300)
- Pure ViT with supervised pretraining struggles on wafer maps

### B2 EVA02-S/14 (~9 min/epoch)
- Highly unstable val_acc: oscillated wildly (0.65, 0.62, 0.89, 0.50, 0.79, 0.93...)
- MIM pretraining may need different LR/warmup strategy
- Better minority recall than DeiT3 but unreliable

### B3 CAFormer-S18 (~12 min/epoch)
- Best val at epoch 1 (0.954!), then gradual overfitting
- Significantly better Loc (0.665) and Scratch (0.677)
- Hybrid conv+attn architecture clearly superior for wafer maps

## Key Insight
**Hybrid (conv+attn) >> Pure ViT for wafer defect maps.**
The conv stages in CAFormer capture local texture patterns while attention stages handle
global spatial relationships. Pure ViTs lack effective local feature extraction at 224x224.

## Decision
Proceed with CAFormer-S18 as primary candidate. Try lower LR + warmup in Phase C.
