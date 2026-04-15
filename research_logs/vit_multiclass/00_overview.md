# ViT 多分类线程总览

## 研究目标

在现有 ResNet50 + GeM 主线已经把训练策略与长尾处理做得较充分的前提下，新开一条 ViT 线程，验证 token-based 全局建模 是否能更好地学习 WM811K 的空间分布模式。

## 当前最佳方案

- 方案名称：**C1 CAFormer-S18 (10ep) + cRT + LogitAdj**
- 关键结构/训练策略：CAFormer-S18 (hybrid conv+attention MetaFormer), lr=1e-4, cosine warmup 2ep, CB sampler + focal(gamma=1.5), 10ep, then cRT(lr=1e-3, focal gamma=1.5, 1ep) + LogitAdj(tau=0.12)
- 对应配置：configs/modal/research_vit/experiments/C1_caformer_s18_lowlr_warmup.yaml
- 检查点：outputs/vit_phase_d/cRT_C1/crt_lr1e-3_g1.5/best.pt

## 当前最佳指标

- accuracy: 0.9516
- macro_f1: 0.7489
- macro_recall: 0.8226
- minority_recall_mean: 0.8089
- **score: 0.8097**

## CNN 基线对比

| Config | score | mF1 | mR | mrm | acc |
|---|---|---|---|---|---|
| D1 ResNet50+GeM (25ep) | 0.8017 | 0.7485 | 0.7988 | 0.7975 | 0.9541 |
| G2 D1+cRT+LogitAdj | **0.8213** | - | - | - | - |
| **C1 CAFormer (10ep)** | **0.8054** | 0.7745 | 0.7946 | 0.7598 | 0.9664 |
| C1+cRT+LogitAdj | 0.8097 | 0.7489 | 0.8226 | 0.8089 | 0.9516 |

**结论：CAFormer 单模型 (0.8054) > ResNet50+GeM 单模型 (0.8017)，但叠加 cRT+LogitAdj 后 CNN 管线 (0.8213) 仍优于 ViT 管线 (0.8097)。**

## 阶段列表

1. Phase A：ViT 基线接入与可训练性验证 - 3 模型全部接入并通过烟雾测试
2. Phase B：中尺度筛选（8ep）- CAFormer >> EVA02 >> DeiT3
3. Phase C：针对性优化 - lr=1e-4 + warmup 最佳；drop_path 无效
4. Phase D：全量训练 + cRT + LogitAdj - 最终 score=0.8097

## 关键发现

1. **Hybrid (conv+attn) >> Pure ViT**：CAFormer-S18 (0.7680 at 8ep) 远超 DeiT3 (0.6510) 和 EVA02 (0.7126)。纯 ViT 在 224x224 wafer map 上缺乏有效的局部特征提取。
2. **更低 LR + Warmup 是 ViT 训练的关键**：lr=1e-4+2ep warmup 比 lr=5e-4 大幅提升 (0.8054 vs 0.7680)。
3. **更多 epoch 不一定更好**：25ep (0.7973) < 10ep (0.8054)，长训练导致少数类 recall 下降（经典长尾过拟合）。
4. **cRT 对 ViT 的提升比 CNN 小**：CNN cRT 提升 +0.0196 (0.8017->0.8213)，ViT cRT 仅提升 +0.0043 (0.8054->0.8097)。可能因为 ViT fc head 只有 4617 参数，可调空间有限。
5. **ViT 单模型已超越 CNN 单模型**：这表明 CAFormer 的特征表征确实更好，但 cRT 这条后处理路径在 ViT 上还未充分发挥。

## 下一步建议

1. **Layer-wise LR Decay (LLRD)**：当前 Trainer 不支持 LLRD。为 ViT 添加 per-layer LR 衰减可能进一步提升。
2. **更大的 cRT head**：当前 fc 只有 Linear(512, 9)。尝试 2 层 MLP head 可增加 cRT 调参空间。
3. **CAFormer-S36 或 CAFormer-M36**：如果 GPU 内存允许，更大的 CAFormer 可能进一步提升。
4. **集成学习**：CAFormer + ResNet50+GeM 的集成可能优于任一单模型。
