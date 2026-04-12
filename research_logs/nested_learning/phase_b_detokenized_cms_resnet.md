# Phase B：de-tokenized CMS ResNet

## 背景与问题

Phase A 表明 tokenized `nested_selfmod` 对当前任务帮助有限，甚至会引入明显的学习塌陷。  
因此这一阶段的核心假设是：wafer defect 更依赖全局空间模式，CNN 已经能提取较强全局特征，Nested Learning 应该作用在 pooled vector 上，而不是强行构造 token sequence。

## 采取措施

- 设计 `nested_cms_resnet`，保留完整 ResNet backbone。
- 去掉 patch embedding 和 token sequence。
- 在 global pooled feature 上施加 `CMS + SelfModifier`。
- 在持续学习设定中加入 `balanced sampling + replay + class_weight_per_task`。
- 按 `search -> refine -> full train` 三步推进。

## 实验设置

- 配置文件：
  - `configs/modal/research_nest/experiments/wm811k_cms_resnet_s1.yaml`
  - `configs/modal/research_nest/experiments/wm811k_cms_resnet_r1.yaml`
  - `configs/modal/research_nest/experiments/wm811k_cms_resnet_r2.yaml`
  - `configs/modal/research_nest/experiments/wm811k_cms_resnet_full_r2.yaml`
- 模型路径：`configs/modal/research_nest/models/nested_cms_resnet.yaml`
- 代码实现：`src/waferlab/models/nested_cms_resnet.py`
- 本地输出目录：
  - `outputs/cms_resnet_search_s1/`
  - `outputs/cms_resnet_refine_r1/`
  - `outputs/cms_resnet_refine_r2/`
  - `outputs/cms_resnet_full_r2/`
- commit / 日期范围：
  - `0c81fdf`（2026-04-07，NestedCMSResNet 去 token 化模型）
  - `de634cd`（2026-04-07，search / refine / full-train configs）
- 训练轮数：
  - search：每 task `3`
  - refine：每 task `5`
  - full：每 task `15`

## 结果汇总

说明：本阶段 `accuracy` 统一记录为 continual 的 `avg_accuracy_final`。

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|
| B1_cms_search_s1 | 去 token 化初版搜索 | 0.5078 | — | — | — | — | 相比 tokenized 版本已有明显提升 |
| B2_cms_refine_r1 | balanced sampling + replay + focal，5 ep/task | 0.6020 | — | — | — | — | 首次把 avg_accuracy_final 提升到 0.60 级别 |
| B3_cms_refine_r2 | 进一步 refined 配置 | 0.6037 | — | — | — | — | 当前线程内部最好的 continual avg_accuracy_final |
| B4_cms_full_r2 | full train，15 ep/task | 0.5349 | — | — | — | — | 长训后并未稳定优于 refine 版本 |

### 持续学习补充指标

| exp_id | avg_accuracy_final | avg_forgetting | full_test_overall_accuracy | full_test_macro_accuracy |
|--------|--------------------|----------------|----------------------------|--------------------------|
| B1_cms_search_s1 | 0.5078 | 0.4300 | 0.9265 | 0.2645 |
| B2_cms_refine_r1 | 0.6020 | 0.2977 | 0.5183 | 0.4741 |
| B3_cms_refine_r2 | 0.6037 | 0.5347 | 0.9258 | 0.4957 |
| B4_cms_full_r2 | 0.5349 | 0.3664 | 0.4322 | 0.5068 |

## 分析与决策

这一步的重要意义在于：

- 去 token 化路线是有效的，至少在 continual 内部指标上显著优于 tokenized 版本。
- 但 full-test 指标依然强烈受 `none` 类分布和 task order 影响，因此不能简单把更高的 overall accuracy 视为“真正更好”。
- 长训练并没有自动带来更稳的收益，说明这条线的瓶颈已经不只是训练轮数。

因此本阶段的决策是：保留 `nested_cms_resnet` 作为更合理的结构版本，同时把后续重点转向 bias / forgetting 控制。

## 下一步

在去 token 化框架上系统测试 KD、weight alignment 和 cosine classifier，确认遗忘控制是否能带来真实收益。
