# Phase D：Top 方案完整训练与最终定版

## 背景与问题

Phase C 的 5 epoch 筛选实验从 7 组方案中选出了 Top-2：
1. **E1 ConvNeXt-Tiny + CB sampler + focal(γ=1.5)**，筛选 score=0.7170
2. **E3 EfficientNetV2-S + CB sampler + focal(γ=1.5)**，筛选 score=0.7100

本阶段的目标是对这两个候选做完整的 25 epoch 训练，并在 Test 集上做最终评估，与当前项目主线基线 D1_gem_tuned（ResNet50 + GeM, score=0.8017）进行严格对比，给出"是否值得替换"的明确结论。

同时，Phase C 已确认以下关键约束：
- CB sampler 是必须项（无 CB 时少数类 recall 崩塌）
- focal loss 是最适合本任务的损失函数（与 CB sampler 互补而非冲突）
- 损失层的类频率校正（balanced softmax / logit adjustment / LDAM）+ CB sampler = 双重过校正，不适合

## 采取措施

- 对 Top-2 各创建 25 epoch 完整训练配置
- 训练策略与基线 D1 保持一致：lr=5e-4, focal(γ=1.5), CB sampler, cosine schedule, weight_decay=1e-3, dropout=0.2
- 使用 best.pt（按 val_acc 选择最佳 checkpoint）在 Test 集上进行完整评估
- 两个模型均使用 ImageNet-1K 预训练权重

## 实验设置

### D-ConvNeXt（ConvNeXt-Tiny 25 epochs）

- 配置文件：`configs/modal/baseline/experiments/phase_e_backbone/phase_d/D_convnext_tiny_focal_full.yaml`
- 模型路径：`configs/modal/baseline/models/convnext_tiny.yaml`
- Recipe 路径：`configs/modal/baseline/recipes/cb_sampler_focal_multiclass.yaml`
- 本地输出目录：`outputs/phase_e/D_convnext_tiny_focal_full/`
- checkpoint：`outputs/phase_e/D_convnext_tiny_focal_full/best.pt`（epoch 21, val_acc=0.9485）
- 评估结果：`outputs/phase_e/D_convnext_tiny_focal_full/eval_metrics_Test.json`
- 训练轮数：25
- 训练时间：~546s/epoch, 总计约 3.8 小时

### D-EffNetV2（EfficientNetV2-S 25 epochs）

- 配置文件：`configs/modal/baseline/experiments/phase_e_backbone/phase_d/D_efficientnetv2_s_focal_full.yaml`
- 模型路径：`configs/modal/baseline/models/efficientnetv2_s.yaml`
- Recipe 路径：`configs/modal/baseline/recipes/cb_sampler_focal_multiclass.yaml`
- 本地输出目录：`outputs/phase_e/D_efficientnetv2_s_focal_full/`
- checkpoint：`outputs/phase_e/D_efficientnetv2_s_focal_full/best.pt`（epoch 12, val_acc=0.9549）
- 评估结果：`outputs/phase_e/D_efficientnetv2_s_focal_full/eval_metrics_Test.json`
- 训练轮数：25
- 训练时间：~377s/epoch, 总计约 2.6 小时

### 复现命令

```bash
source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch

# ConvNeXt-Tiny 训练
python scripts/train_classifier.py --config configs/modal/baseline/experiments/phase_e_backbone/phase_d/D_convnext_tiny_focal_full.yaml

# ConvNeXt-Tiny 评估
python scripts/eval_classifier.py --config configs/modal/baseline/experiments/phase_e_backbone/phase_d/D_convnext_tiny_focal_full.yaml --checkpoint outputs/phase_e/D_convnext_tiny_focal_full/best.pt --split Test

# EfficientNetV2-S 训练
python scripts/train_classifier.py --config configs/modal/baseline/experiments/phase_e_backbone/phase_d/D_efficientnetv2_s_focal_full.yaml

# EfficientNetV2-S 评估
python scripts/eval_classifier.py --config configs/modal/baseline/experiments/phase_e_backbone/phase_d/D_efficientnetv2_s_focal_full.yaml --checkpoint outputs/phase_e/D_efficientnetv2_s_focal_full/best.pt --split Test
```

## 结果汇总

### 总体对比

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| D1_gem_tuned (基线) | ResNet50+GeM, focal, CB | 0.9541 | 0.7485 | 0.7988 | 0.7975 | 0.8017 | 当前主线 |
| D-ConvNeXt | ConvNeXt-Tiny, focal, CB | 0.9485 | 0.7473 | 0.8120 | 0.7861 | 0.8014 | 与基线持平 |
| D-EffNetV2 | EffNetV2-S, focal, CB | 0.9549 | 0.7415 | 0.8152 | 0.7905 | 0.8017 | 与基线持平 |

**三个模型的 score 在 0.8014–0.8017 之间，差异仅 0.0003，在统计意义上完全等价。**

### 逐类 Recall 对比

| 类别 | D1 基线 | D-ConvNeXt | D-EffNetV2 | 最优 |
|------|---------|------------|------------|------|
| none | 0.9693 | 0.9623 | 0.9693 | 基线 / EffNet |
| Center | 0.6875 | 0.7139 | 0.6550 | ConvNeXt |
| Donut | 0.9110 | 0.8425 | 0.8836 | 基线 |
| Edge-Loc | 0.8023 | 0.8470 | 0.8193 | ConvNeXt |
| Edge-Ring | 0.6794 | 0.7789 | 0.8410 | EffNet |
| Loc | 0.6934 | 0.5966 | 0.6143 | 基线 |
| Near-full | 0.9684 | 0.9895 | 1.0000 | EffNet |
| Random | 0.7510 | 0.7899 | 0.7549 | ConvNeXt |
| Scratch | 0.7273 | 0.7879 | 0.7994 | EffNet |

### 各维度最优归属

| 指标 | D1 基线 | D-ConvNeXt | D-EffNetV2 | 备注 |
|------|---------|------------|------------|------|
| accuracy | **0.9541** | 0.9485 | **0.9549** | EffNet 微胜 |
| macro_f1 | **0.7485** | 0.7473 | 0.7415 | 基线微胜 |
| macro_recall | 0.7988 | 0.8120 | **0.8152** | EffNet 微胜 |
| minority_recall_mean | **0.7975** | 0.7861 | 0.7905 | 基线微胜 |
| score | **0.8017** | 0.8014 | **0.8017** | 持平 |

## 训练动态分析

### val_acc 振荡问题

三个模型均表现出明显的 val_acc 振荡现象，这是 CB sampler 的固有特性——训练时看到均衡分布，但验证集保持自然分布（none 类 ~93%），导致不同 epoch 在多数类和少数类之间的权衡点不同。

- **ConvNeXt-Tiny**：振荡最剧烈，val_acc 范围 0.472–0.949，best 出现在 epoch 21
- **EfficientNetV2-S**：振荡较温和，val_acc 范围 0.485–0.955，best 出现在 epoch 12，收敛更快
- **ResNet50+GeM**（参照）：振荡存在但相对较轻

EfficientNetV2-S 收敛速度最快（epoch 2 即达 val_acc=0.934），ConvNeXt-Tiny 最慢（epoch 4 才首次突破 0.9）。

### 训练效率

- ConvNeXt-Tiny：~546s/epoch（最慢）
- EfficientNetV2-S：~377s/epoch（最快，比 ConvNeXt 快 31%）
- ResNet50+GeM：~450s/epoch（中间）

## 分析与决策

### 核心发现：三个 backbone 在当前训练框架下的性能天花板高度一致

这是本线程最重要的结论。尽管 ConvNeXt-Tiny 和 EfficientNetV2-S 在 ImageNet 上分别比 ResNet50 高出 ~6% 和 ~8% 的 top-1 accuracy，但在 WM-811K 9 类多分类任务上，三者的最终 score 完全等价（0.8014–0.8017）。

### 原因分析

1. **WM-811K 不是 ImageNet**：晶圆图的视觉复杂度远低于自然图像，ResNet50 级别的表征能力可能已经足够捕捉晶圆缺陷模式的特征，更强的 backbone 带来的额外表征能力无法被利用。

2. **性能瓶颈不在 backbone**：当前任务的难点在于极端类不平衡（none 类占比 67.6%，Near-full 仅 54 个训练样本）和形态混淆类别对（如 Center vs Loc, Edge-Loc vs Edge-Ring），这些问题不会因为更强的 backbone 而消失。

3. **数据量限制了 backbone 的优势**：更大的模型需要更多数据来充分利用其表征能力。WM-811K 虽有 ~54K 训练样本，但去除 none 类后有效少数类数据仅约 17K，可能不足以支撑更强 backbone 的潜力发挥。

4. **训练策略是真正的差异因子**：从 Phase C 的实验中可以看到，损失函数选择（focal vs balanced_softmax）和采样策略（有/无 CB sampler）带来的性能差异远大于 backbone 之间的差异。

### 最终结论

**不建议替换当前 ResNet50 + GeM 基线。**

理由：
1. 新 backbone 没有带来任何有统计意义的性能提升
2. ConvNeXt-Tiny 训练时间比 ResNet50 多 ~20%，但不更好
3. EfficientNetV2-S 训练时间更短，但 macro_f1 反而稍低
4. ResNet50 + GeM 结构简单、生态成熟、debug 方便
5. 作为持续学习 backbone，ResNet50 的中间层接口更标准，兼容性更好

### 附加发现：EfficientNetV2-S 的潜在价值

虽然 score 持平，但 EfficientNetV2-S 具有一些独特优势：
- 训练速度最快（比 ConvNeXt 快 31%，比 ResNet50 快 ~16%）
- accuracy 最高（0.9549 vs 0.9541）
- macro_recall 最高（0.8152 vs 0.7988）
- Near-full recall = 1.0（完美），Scratch recall 最高（0.7994）

如果未来需要一个"速度更快且表现不差"的替代 backbone，EfficientNetV2-S 是值得保留的选项，但当前不构成替换理由。

## 被淘汰方案总结

| 方案 | 淘汰原因 |
|------|----------|
| ConvNeXt-Tiny（替代主线） | score=0.8014，与基线持平，训练更慢，不值得替换 |
| EfficientNetV2-S（替代主线） | score=0.8017，与基线持平，macro_f1 略低，不值得替换 |
| ConvNeXt-Small | Phase B 仅做冒烟验证，参数更多但预期收益有限，未进入筛选 |
| Balanced Softmax + CB | Phase C 筛选中 score=0.5953，双重过校正导致灾难性表现 |
| LDAM + CB | Phase C 筛选中 score=0.6790，精度崩塌 |
| Logit Adjustment + CB | Phase C 筛选中 score=0.6841，多数类偏置 |
| 无 CB sampler 方案 | Phase C 筛选中少数类 recall 崩塌 |

## 下一步

当前 ResNet50 + GeM 基线已经过多维度验证，不建议继续在 backbone 层面投入。后续提升方向应转向：
1. **数据层面**：少数类增强、多数类下采样策略精细化
2. **训练策略层面**：两阶段训练（表征学习 + 分类头重平衡）、logit 后校准
3. **任务层面**：在持续学习框架下利用当前 backbone 进行进一步探索
