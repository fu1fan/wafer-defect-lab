# Phase A：tokenized Nested SelfMod

## 背景与问题

在 ResNet18 多分类阶段已经看到长尾问题和类别混淆依然明显，因此尝试引入 Nested Learning 的记忆与自调制机制，希望通过多频记忆和 teach-signal 帮助模型更好地区分难类并缓解后续持续学习中的遗忘。

## 采取措施

- 使用 `ResNet18` 前两层作为 CNN stem。
- 对 feature map 做 `patch embedding`，转成 token sequence。
- 堆叠 `NestedBlock`，引入 `CMS + SelfModifier + surprise gating`。
- 开启 `nested_teach` 机制。

## 实验设置

- 配置文件：
  - `configs/modal/research_nest/experiments/wm811k_nested_selfmod_multiclass.yaml`
  - `configs/modal/research_nest/experiments/wm811k_nested_selfmod_continual.yaml`
- 模型路径：`configs/modal/research_nest/models/nested_selfmod.yaml`
- 代码实现：`src/waferlab/models/nested_selfmod.py`
- 本地输出目录：
  - `outputs/nested_selfmod_multiclass_20260405_141739/`
  - `outputs/nested_selfmod_continual_20260405_142237/`
- checkpoint：`outputs/nested_selfmod_multiclass_20260405_141739/best.pt`
- commit / 日期范围：
  - `a6e3d36`（2026-04-06，nested_selfmod 模型与 continual 训练）
  - `5c183fe`（2026-04-06，nested_selfmod 相关实验配置）
- 训练轮数：
  - 多分类：`20`
  - continual：每 task `5`

## 结果汇总

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| A1_nested_selfmod_multiclass | tokenized Nested SelfMod，多分类 | 0.9382 | 0.2658 | 0.2340 | 0.1486 | — | 大部分类别几乎没学起来 |
| A2_nested_selfmod_continual | tokenized Nested SelfMod，3-task continual | 0.3815 | — | — | — | — | 持续学习内部表现也不理想 |

## 分析与决策

这一阶段基本确认：

- token 序列化后的 nested blocks 对当前 wafer map 任务并不匹配。
- 多分类结果大量塌陷到 `none` 类和少数几个大类，说明模型没有真正学到稳定的细粒度模式。
- 持续学习结果也没有体现出预期优势。

因此后续决策不是继续在 token 细节上打补丁，而是直接检讨“token 化这一步是否必要”。

## 下一步

移除 token 化流程，改为在完整 CNN backbone 的 pooled vector 上施加 CMS / SelfModifier，验证去 token 化是否更适合当前任务。
