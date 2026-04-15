# Phase A: ViT Backbone Integration

## Goal
Integrate 3 ViT-family candidates into waferlab, validate trainability via smoke tests.

## Candidate Selection

| Model | Architecture | Pretraining | Params | Feature Dim |
|---|---|---|---|---|
| DeiT III-S/16 | Pure ViT, patch16 | Supervised IN22K to IN1K | 21.7M | 384 |
| EVA-02-S/14 | Pure ViT, patch14 | MIM on IN22K | 21.6M | 384 |
| CAFormer-S18 | Hybrid conv+attn (MetaFormer) | Supervised IN1K | 23.2M | 512 |

### Excluded
- DINOv2 ViT-S/14: Same arch as EVA-02; pretrained at 518x518 needs pos-embed interpolation
- Swin-Tiny: Hierarchical local attention ~ CNN bias; ConvNeXt already tested, no advantage
- MaxViT-Tiny: Multi-axis attention too complex for screening
- TinyViT-21M: KD-based hybrid closer to CNN

## Implementation

### New Files
- src/waferlab/models/vit_backbones.py: TimmViTWrapper class
  - Auto 1-ch adaptation (average RGB weights to single channel)
  - Supports forward/forward_features/get_cam_target_layer API
  - Registers: deit3_small_wafer, eva02_small_wafer, caformer_s18_wafer
- configs/modal/research_vit/models/: 3 model YAMLs
- configs/modal/research_vit/recipes/cb_focal_vit.yaml: CB sampler + focal recipe
- configs/modal/research_vit/experiments/B1-B3: Screening experiment configs

### 1-ch Adaptation
TimmViTWrapper auto-finds the first Conv2d in patch_embed, averages RGB weights across
input channel dim. For CAFormer (MetaFormer arch), traverses stem/downsample_layers.

### Smoke Tests
All 3 models pass: forward, feature extraction, GradCAM, 1-epoch training, eval pipeline.

## Conclusion
Engineering integration complete. All models trainable. Proceed to Phase B.
