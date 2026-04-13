# 更强 Backbone 搜索线程总览

## 研究目标

在 WM-811K 9 类多分类任务上，寻找比当前 ResNet50 + GeM 更强的静态分类 backbone。目标是同时提升 macro_f1、macro_recall 和少数类 recall，形成可用于后续持续学习的更强基线。当前 ResNet50 架构（2015 年设计）的表征能力已成为瓶颈，GeM 池化是唯一有效的结构增强，CBAM 等注意力机制无效。

## 最终结论

**不建议替换当前 ResNet50 + GeM 基线。**

经过完整的 4 阶段实验（调研 → 接入 → 筛选 → 完整训练），ConvNeXt-Tiny 和 EfficientNetV2-S 均未能超越 ResNet50 + GeM。三者在最终 score 上完全等价（0.8014–0.8017），说明当前任务的性能瓶颈不在 backbone 表征能力，而在数据不平衡和类间混淆。

## 当前最佳方案（未变更）

- 方案名称：D1_gem_tuned（ResNet50 + GeM）
- 关键结构/训练策略：ResNet50 + GeM + CB sampler + focal(γ=1.5) + lr=5e-4 + cosine + 25 epochs
- 对应配置：`configs/modal/baseline/experiments/phase_d/resnet50_gem_tuned.yaml`

## 当前最佳指标（未变更）

**D1_gem_tuned, 25 epochs, Test 集评估：**

- accuracy：0.9541
- macro_f1：0.7485
- macro_recall：0.7988
- minority_recall_mean：0.7975
- score：0.8017

## Phase D 完整对比

| 模型 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score |
|------|----------|----------|--------------|----------------------|-------|
| ResNet50+GeM (基线) | 0.9541 | **0.7485** | 0.7988 | **0.7975** | **0.8017** |
| ConvNeXt-Tiny | 0.9485 | 0.7473 | 0.8120 | 0.7861 | 0.8014 |
| EfficientNetV2-S | **0.9549** | 0.7415 | **0.8152** | 0.7905 | **0.8017** |

## 阶段列表

1. Phase A：外部调研与候选筛选 → `phase_a_survey.md`
2. Phase B：工程接入与冒烟验证 → `phase_b_smoke_and_integration.md`
3. Phase C：中小规模筛选实验（5 epochs, 7 组） → `phase_c_screening.md`
4. Phase D：Top-2 方案完整训练（25 epochs） → `phase_d_final.md`

## 关键发现

1. **CB sampler 是必须项**：无 CB sampler 时少数类 recall 崩塌（Scratch=0.0, Donut=0.034）
2. **focal loss 是最适合本任务的损失函数**：与 CB sampler 互补（CB 管采样均衡，focal 管难样本挖掘）
3. **损失层类频率校正 + CB sampler = 双重过校正**：Balanced Softmax + CB 导致 acc=0.41，LDAM + CB 导致精度崩塌
4. **更强 backbone 不等于更好性能**：ImageNet 上 +6~8% 的优势在 WM-811K 上完全消失
5. **训练策略比 backbone 选择更重要**：focal vs balanced_softmax 的差异远大于 ConvNeXt vs ResNet50

## 工程产出

- 新增模型文件：`src/waferlab/models/modern_backbones.py`（ConvNeXt-Tiny / EfficientNetV2-S / ConvNeXt-Small）
- 新增损失函数：`src/waferlab/engine/losses.py` 中的 LDAMLoss
- 新增模型配置：`configs/modal/baseline/models/convnext_tiny.yaml` 等 3 个
- 新增 recipe 配置：4 个（balanced_softmax / logit_adj / ldam / logit_adj_no_cb）
- 新增实验配置：10 个（Phase C 筛选） + 2 个（Phase D 完整训练）

## 下一步建议

当前 backbone 层面的探索空间已经穷尽。后续提升方向应转向：
1. 数据层面：少数类增强、多数类精细化下采样
2. 训练策略：两阶段训练（表征 + 分类头重平衡）、logit 后校准
3. 任务层面：在持续学习框架下利用 ResNet50+GeM backbone 进行进一步探索
