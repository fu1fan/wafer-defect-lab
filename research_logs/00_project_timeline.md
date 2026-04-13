# 项目总时间线

## 当前定位

当前项目已经从“搭建晶圆缺陷分类实验框架”推进到“围绕 9 类多分类建立更强基线并准备为后续结构探索提供更强 backbone”的阶段。  
主线演进可以概括为：

`框架搭建 -> ResNet18 二分类 -> ResNet18 多分类 -> Nested Learning / Continual Learning -> ResNet50 分阶段增强`

## 阶段迁移摘要

### 2026-03-29 ~ 2026-03-30：工程框架与数据闭环

- 完成项目基础结构、README、Makefile、requirements。
- 打通 `WM-811K` 数据处理链路，形成 `raw -> interim -> processed` 的统一入口。
- 建立训练、评估、可视化和远程运行基础脚本。
- 这一阶段奠定了后续所有实验共享的工程底座。

### 2026-04-04 ~ 2026-04-05：ResNet18 二分类基线与召回率优化

- 先跑通标准 `ResNet18 + CE` 二分类基线。
- 发现 abnormal 类在不平衡数据下召回偏低。
- 基于此实现 `resnet18_recall_opt`，引入 `GeM + SE + 两层分类头 + Focal Loss + class weights`。
- 二分类阶段的核心收获是：异常类召回率被明显抬升，说明“结构改造 + loss 调整”对不平衡场景是有效的。

### 2026-04-05 ~ 2026-04-06：ResNet18 多分类与长尾问题暴露

- 将前一阶段的经验迁移到 `WM-811K 9 类多分类`。
- 发现整体 accuracy 依然较高，但少数类表现明显受 `none` 类主导。
- 围绕 class imbalance 做了 `class-balanced sampler / focal / class-balanced focal / combined` 等一组对比。
- 这一阶段的结论是：仅靠 ResNet18 和损失/采样层面的修补，能改善一部分少数类，但不足以形成长期稳定的强基线。

### 2026-04-06 ~ 2026-04-07：Nested Learning / Continual Learning 结构探索

- 先引入 tokenized 的 `nested_selfmod` 路线，把 Nested Learning 的 `CMS + SelfModifier + teach-signal` 嵌入 ResNet18 stem 后的 token 序列。
- 结果显示 token 化版本在多分类上学习不足，效果远差于预期。
- 随后改为去 token 化的 `nested_cms_resnet`，在完整 CNN backbone 的 pooled feature 上施加 `CMS + SelfModifier`。
- 去 token 化后持续学习内部指标明显改善，但整体结果仍受到任务顺序和 `none` 类位置影响。
- 最后系统加入 `masked KD / replay-only KD / weight alignment / cosine classifier` 等偏差校正与抗遗忘方案，结论是：KD 能减轻 forgetting，但在当前任务顺序下并未形成最终可用方案。

### 2026-04-08 ~ 2026-04-09：ResNet50 多分类分阶段增强

- 在明确“嵌套学习更依赖强 backbone”后，转向以 `ResNet50` 重建多分类强基线。
- 按 `Phase A/B/C/D` 方式推进：
  - Phase A：建立增强版 `R0` 基线
  - Phase B：做 9 组超参数搜索
  - Phase C：做 `GeM / CBAM / GeM+CBAM` 结构增强对比
  - Phase D：整合最佳超参与最佳结构得到 `D1_gem_tuned`
- 当前最好结果已经达到：
  - `accuracy=0.9541`
  - `macro_f1=0.7485`
  - `macro_recall=0.7988`
  - `minority_recall_mean=0.7975`
  - `score=0.8017`

### 2026-04-13：更强 Backbone 搜索（ConvNeXt / EfficientNetV2）

- 线程：`research_logs/strong_backbone_search/`
- 目标：寻找比 ResNet50+GeM 更强的静态基线 backbone
- 经过完整 4 阶段实验（调研 → 接入 → 筛选 → 完整训练）
- 候选 backbone：ConvNeXt-Tiny (ImageNet 82.5%)、EfficientNetV2-S (ImageNet 84.2%)、ConvNeXt-Small
- 中小规模筛选（5 epochs, 7 组实验）确认：
  - CB sampler 是必须项；focal loss 是最佳损失函数
  - 损失层类频率校正 + CB sampler = 双重过校正
- 完整训练（25 epochs）结果：
  - ConvNeXt-Tiny：score=0.8014（与基线 0.8017 持平）
  - EfficientNetV2-S：score=0.8017（与基线持平）
- **结论：不建议替换 ResNet50+GeM 基线。** 更强的 ImageNet 预训练不能转化为 WM-811K 上的优势，性能瓶颈不在 backbone。
- 工程产出：新增 3 个 backbone 模型、LDAM 损失、12 个实验配置

## 当前状态

- 工程上已经具备较完整的配置化实验框架、注册表机制和可扩展训练入口。
- 历史主线已经验证了：
  - 二分类下，结构与 loss 联合优化是有效的。
  - 多分类下，长尾问题不能只靠 ResNet18 的轻量修补解决。
  - Nested Learning 提供了值得保留的结构思路，但当前还需要更强 backbone 才能支撑。
  - **更强的 backbone（ConvNeXt, EfficientNetV2）不能进一步提升 WM-811K 性能，瓶颈在数据不平衡而非表征能力。**
- 当前最优主线仍然是 `resnet50_multiclass` 的 `D1_gem_tuned`（score=0.8017）。
- 后续优先方向应放在数据层面增强、训练策略调整（两阶段训练、logit 校准），以及在持续学习框架下的探索。
