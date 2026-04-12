# ResNet50 多分类线程总览

## 研究目标

在确认 ResNet18 已不足以支撑 9 类多分类长期主线后，以 `ResNet50` 为新 backbone 重建更强的 `WM-811K` 多分类基线，并用分阶段方式系统完成基线复现、超参数搜索、结构增强和最终定版。

## 当前最佳方案

- 方案名称：`D1_gem_tuned`
- 关键结构/训练策略：`ResNet50 + GeM Pooling + class-balanced sampler + focal_gamma=1.5 + lr=5e-4`
- 对应配置：`configs/modal/baseline/experiments/phase_d/resnet50_gem_tuned.yaml`

## 当前最佳指标

- accuracy：`0.9541`
- macro_f1：`0.7485`
- macro_recall：`0.7988`
- minority_recall_mean：`0.7975`
- score：`0.8017`

## 阶段列表

1. Phase A：增强版 `R0` 基线复现
2. Phase B：9 组超参数搜索
3. Phase C：`GeM / CBAM / GeM+CBAM` 结构增强
4. Phase D：最佳超参与最佳结构组合定版

## 当前结论

这条线程已经形成明确结论：

- 最有效的结构改动是 `GeM Pooling`。
- 最有效的训练改动是降低 `learning rate` 和 `focal gamma`。
- `class-balanced sampler` 是必须保留的前提条件。
- `CBAM` 在当前任务上收益不够，不值得进最终方案。

目前 `D1_gem_tuned` 是项目内最值得继续沿用的多分类基线。

## 下一步计划

围绕更细粒度的 loss / sampling 微调继续优化当前最佳基线，而不是继续堆叠更重的注意力结构。
