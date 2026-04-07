# 任务名称
去 token 化重构 Nested Learning 架构，并通过激进搜索获得最优持续增量分类模型

## 你的角色
你是可自主执行的高级机器学习工程师。请在不需要人工逐步干预的前提下，完成代码重构、实验迭代、调参与最终交付。

## 目标
1. 新建一个不依赖 patch embedding/token 序列的新模型（保留 nested learning 的持续学习机制）。
2. 跑通冒烟测试，确保 continual 全流程可执行。
3. 完成中小规模激进搜索（含不平衡处理策略），筛选 top 方案。
4. 对 top 方案做完整训练并交付最优模型。

## 项目背景（必须阅读并遵守）
- 数据任务：WM-811K 9 类持续增量分类，3 个 task。
- 关键不平衡：`none` 类占比极高（约 93%），Donut/Loc/Near-full 等为少数类。
- 当前主模型文件：`src/waferlab/models/nested_selfmod.py`。
- 持续学习训练入口：`scripts/train_continual.py`。
- 训练器与 KD/nested_teach 逻辑：`src/waferlab/engine/trainer.py`。
- nested learning 核心组件：
  - `src/waferlab/engine/nested_learning/nested_block.py`
  - `src/waferlab/engine/nested_learning/cms.py`
  - `src/waferlab/engine/nested_learning/self_modifier.py`
- 注册机制：`src/waferlab/registry.py` + `src/waferlab/models/__init__.py` 自动发现。
- 配置体系：`configs/modal/research_nest/models/*.yaml` 与 `configs/modal/research_nest/experiments/*.yaml` 采用 `_base_` 继承。

## 强约束
1. 不覆盖旧模型：禁止修改或替换 `src/waferlab/models/nested_selfmod.py`。
2. 必须新建文件实现新模型，例如：
   - `src/waferlab/models/nested_cms_resnet.py`
3. 必须走注册机制：
   - 在新文件底部用 `@MODEL_REGISTRY.register("nested_cms_resnet")` 注册。
4. 保持工程风格与可扩展性：
   - 与现有 `WaferClassifier` / `NestedSelfModClassifier` 风格一致。
   - 保留常用接口（如 `forward`、`forward_features`，必要属性如 `num_classes`、`in_channels`）。
5. 必须新建配置文件，不覆盖已有实验配置。

## 架构改造方向（可优化但需给出理由）
建议路线：
- 输入图像 -> CNN backbone（建议完整 ResNet18/34）
- 全局池化得到向量特征
- 在向量特征上应用 CMS 多频率记忆 + SelfModifier（替代 token 链路）
- 分类头输出 9 类

请说明：
- 为什么该结构比 token 链路更适配图像缺陷模式。
- 如何保留 nested learning 的持续学习优势。

## 实验计划（必须执行）
### Phase A：冒烟验证
- 新模型接入后，进行极短训练（如 1-3 epochs/task）。
- 验证训练、评估、结果落盘均正常。

### Phase B：中小规模激进搜索
- 固定随机种子，进行不少于 6 组组合实验。
- 至少覆盖以下维度：
  - 采样：普通 / class-balanced sampler
  - 损失：CE+class_weight / Focal / Balanced Softmax 或 Logit Adjustment
  - replay：uniform / minority-first
  - KD：开/关（若启用，需说明是否 replay-only）

- 评分函数（用于自动筛选 top-2）：
  - 0.5 * macro_accuracy
  - 0.3 * (1 - avg_forgetting)
  - 0.2 * minority_recall_mean（Center/Donut/Loc/Near-full）

### Phase C：完整训练
- 对 top-2 方案进行完整训练（15 epochs/task）。
- 在关键超参上至少做一轮细化搜索（lr、weight_decay、focal_gamma、kd_lambda 等）。

## 不平衡处理要求
必须明确比较并汇报以下类型策略：
1. 数据层：采样与 replay 配额策略。
2. 损失层：class weight / focal / balanced softmax / logit adjustment。
3. 预测层：必要时做 bias 校正（如 BiC 风格后处理）。

## 过程要求
1. 每次重大改动后都要先做可运行性验证再继续。
2. 遇错先自修复，不把半成品作为最终结果。
3. 每个阶段结束给出结论与下一步选择理由。
4. 在关键节点提交 git（至少：冒烟通过后一次、最终最优方案一次）。

## 最终交付物（必须完整）
1. 新增/修改文件清单。
2. 最优模型结构说明。
3. 全部实验对比表（含 v4 基线）。
4. 最终最优配置与训练日志路径。
5. 可复现命令（从环境激活到训练评估）。
6. 失败方案与淘汰原因总结。

## 终止条件
仅当以下全部满足时才停止：
1. 新模型已实现并可运行。
2. 完成中小规模搜索并选出 top 方案。
3. 完成完整训练并产出最优结果。
4. 交付全部结果与复现命令。