# Nested Learning 线程总览

## 研究目标

把 `Nested Learning / CMS / SelfModifier / teach-signal` 的思想迁移到晶圆图多分类与持续学习中，验证它是否能在长尾分类和增量学习场景中带来额外收益。

## 当前最佳方案

- 方案名称：`nested_cms_resnet`（去 token 化版本）
- 关键结构/训练策略：完整 CNN backbone + pooled feature 上的 `CMS + SelfModifier + replay + class-balanced continual setting`
- 对应配置：`configs/modal/research_nest/experiments/wm811k_cms_resnet_r1.yaml` / `wm811k_cms_resnet_r2.yaml`

## 当前最佳指标

- accuracy：`0.6037`
- macro_f1：`—`
- macro_recall：`—`
- minority_recall_mean：`—`
- score：`—`

说明：这里的 `accuracy` 指持续学习的 `avg_accuracy_final`。  
在 full-test 角度，`cms_resnet_refine_r2` 的 `overall_accuracy=0.9258`，但仍明显受到 `none` 类分布与任务顺序影响，不能等价理解为稳定优于强 CNN 基线。

## 阶段列表

1. Phase A：tokenized `nested_selfmod`
2. Phase B：de-tokenized `nested_cms_resnet`
3. Phase C：KD / bias correction / anti-forgetting 系统实验

## 当前结论

这条线程最重要的结论不是“已经成功替代 CNN 基线”，而是：

- token 化版本不适合当前任务；
- 去 token 化版本显著改善了持续学习内部指标；
- KD 能减轻 forgetting，但和 `none` 类在最后任务出现的设定存在冲突；
- 因此当前它仍是一条**值得保留的结构探索线**，但尚未形成比强基线更稳定的最终方案。

## 下一步计划

先在更强 backbone 上重建稳健多分类基线，再考虑把 Nested Learning 的机制重新引入。
