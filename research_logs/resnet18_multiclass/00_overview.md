# ResNet18 多分类线程总览

## 研究目标

基于已经在二分类中验证有效的 `resnet18_recall_opt`，推进 `WM-811K 9 类多分类`，并重点评估长尾类别在不同采样与损失策略下的表现。

## 当前最佳方案

- 方案名称：`resnet18_recall_opt + class-balanced sampler + focal`
- 关键结构/训练策略：延续 recall-oriented ResNet18，并把长尾处理重点放在 sampler 与 loss 上
- 对应配置：`configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_cb_sampler_multiclass.yaml`

## 当前最佳指标

- accuracy：`0.9666`
- macro_f1：`0.7692`
- macro_recall：`0.7759`
- minority_recall_mean：`0.7084`
- score：`—`

补充：如果只看“少数类 recall 更激进”的筛选结果，`screen_cb_sampler_8ep` 的 `minority_recall_mean=0.7672` 更高，但稳定性和最终训练轮数不如完整版本。

## 阶段列表

1. Phase A：将 recall-opt ResNet18 直接切到 9 类多分类
2. Phase B：围绕 class imbalance 做 sampler / focal / combined 对比

## 当前结论

ResNet18 在线上切到多分类后，整体 accuracy 依旧很高，但很容易被 `none` 类主导。  
相比单纯沿用 focal，多分类场景中 `class-balanced sampler` 是更关键的改进；不过整体上看，ResNet18 仍然不足以成为长期的强基线。

## 下一步计划

基于当前结论转向更强 backbone，而不是继续在 ResNet18 上做过度精修。
