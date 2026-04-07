# 任务名称
围绕 WM-811K 的 ResNet50 多分类模型进行分阶段增强、搜索与最终定版

## 环境要求（强制执行）
- 所有 Python 命令、训练、评估、脚本执行前，必须先执行：
  `source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch`
- 全部工作必须在名为 `torch` 的 conda 环境中进行，不得使用其他 Python 环境。
- 确保工作目录位于项目根目录。

## 你的角色
你是可自主执行的高级机器学习工程师。请在红线约束内自主决策，完成实验执行、参数搜索、结构增强、结果分析与最终模型交付，无需人工逐步干预。

## 前置状态
- Phase A 的 R0 实验尚未完整运行，你需要从头执行 Phase A，产出完整评估报告。
- 当前已具备 R0 配置文件：
  - `configs/modal/baseline/experiments/wm811k_resnet50_cb_sampler_focal_heavy_aug_multiclass.yaml`

## 总目标
1. 完整运行 R0 增强版 ResNet50，建立可靠基线。
2. 在 R0 基础上进行中小规模调参与筛选，再对 Top 方案进行完整训练。
3. 分别引入 `GeM Pooling`、`CBAM` 或你检索后判断更适合的轻量注意力增强，评估其对 WM-811K 9 类识别的真实增益。
4. 综合实验结果形成最终版本模型，完成完整训练并交付可复现结果。

## 项目背景（必须阅读并遵守）
- 数据任务：WM-811K 晶圆图 9 类多分类，类别为：
  - `none`
  - `Center`
  - `Donut`
  - `Edge-Loc`
  - `Edge-Ring`
  - `Loc`
  - `Near-full`
  - `Random`
  - `Scratch`
- 核心代码路径：
  - 模型：`src/waferlab/models/resnet.py`
  - 参考实现：`src/waferlab/models/resnet_recall_opt.py`
  - 数据：`src/waferlab/data/dataloaders.py`、`src/waferlab/data/transforms.py`
  - 训练入口：`scripts/train_classifier.py`
- 当前 baseline 配置：
  - `configs/modal/baseline/experiments/wm811k_resnet50_baseline_multiclass.yaml`
- 当前增强版 ResNet50 配置：
  - `configs/modal/baseline/experiments/wm811k_resnet50_cb_sampler_focal_heavy_aug_multiclass.yaml`
- 配置体系采用 `_base_` 继承，保持工程风格一致。所有新配置优先放在：
  - `configs/modal/baseline/models/`
  - `configs/modal/baseline/recipes/`
  - `configs/modal/baseline/experiments/`

## 已知起点（Baseline 性能）
- 纯 ResNet50 多分类 baseline 已有结果大致为：
  - `accuracy ≈ 0.9696`
  - `macro_f1 ≈ 0.7487`
  - `macro_recall ≈ 0.7216`
- 当前较弱类别重点关注：
  - `Donut`
  - `Loc`
  - `Scratch`
  - `Center`
  - `Edge-Ring`

请不要只看 overall accuracy，必须重点关注 `macro_f1`、`macro_recall` 和少数类 recall。

## 强约束（必须遵守）
1. 必须完整执行 Phase A 的 R0 实验，记录全部指标后再进入 Phase B。
2. 不要覆盖已有 baseline 配置文件；所有新实验均新建配置文件。
3. 若新增模型结构，优先新建模型文件或在现有文件中以可维护方式扩展，不得破坏 `resnet18` / `resnet50` baseline 行为。
4. 保持工程风格一致：
   - 配置放在 `configs/modal/baseline/models|recipes|experiments/`
   - 模型通过 registry 注册
   - 训练统一走 `scripts/train_classifier.py`
5. 每次结构改动后必须先做可运行性验证，再进入批量实验。
6. 所有实验必须记录在对比表中，包括失败项和淘汰项。
7. 严禁无充分理由修改现有 class-balanced sampler 的核心逻辑；若必须改动，需在报告中说明原因、影响和回滚方案。
8. 最终必须给出完整训练的最优模型与复现命令。

## 自主决策权限
你可以在以下范围内自主检索资料并做出合理决策，无需逐一确认，但必须在报告中说明理由：

### 可自主决定的范围
1. 超参数搜索空间：
   - 可在预设网格基础上做小幅扩展，但需有明确依据。
2. 结构增强变体：
   - 若你判断 `Coordinate Attention`、`ECA-Net` 等比 `CBAM` 更适配 WM-811K，可替换或补充对比，但必须说明理由和来源。
3. 训练策略优化：
   - 可尝试 `EMA`、更稳定的 scheduler 或轻量正则化技巧，但要保持代码改动可控、易复现。
4. 增强策略补充：
   - 仅限 wafer-safe 前提下的空间类增强，不得破坏离散像素语义。

### 绝对不可逾越的红线
- 严禁使用 `ColorJitter`、`RandomErasing`、`Cutout`、`GaussianBlur`、`GridMask` 或任何会改变像素离散语义、引入虚假颜色或添加纯色块的操作。
- 严禁为了“追求高分”而隐藏任何失败实验、异常结果或淘汰项。
- 严禁在最终结论中只汇报 `accuracy` 而忽略 `macro_f1`、`macro_recall` 与少数类 recall。

## 优先优化目标
主目标按重要性排序：
1. `macro_f1`
2. `macro_recall`
3. 少数类平均 recall
4. `accuracy`

其中少数类平均 recall 至少基于以下 5 个类：
- `Center`
- `Donut`
- `Loc`
- `Near-full`
- `Scratch`

若某方法只提升 `accuracy` 而未改善 `macro_f1` 或少数类 recall，则判定为无效或低优先级。

## 实验总流程（必须执行）

### Phase A：R0 起点复现与基线确认
1. 运行配置：
   - `configs/modal/baseline/experiments/wm811k_resnet50_cb_sampler_focal_heavy_aug_multiclass.yaml`
2. 确认训练、验证、评估、结果落盘均正常。
3. 记录 R0 关键指标：
   - accuracy
   - macro_precision
   - macro_recall
   - macro_f1
   - per-class precision / recall / f1
   - confusion matrix
4. 产出一页简短结论：
   - 当前最主要错误模式
   - 哪些类别最难
   - 是否出现明显过采样副作用或 double-correction 风险

### Phase B：当前增强版 ResNet50 的参数搜索
先不改模型主结构，只调 R0 增强版 ResNet50 的参数。

至少覆盖以下维度：
- 学习率：`3e-4 / 5e-4 / 1e-3`
- weight decay：`1e-4 / 5e-4 / 1e-3`
- dropout：`0.1 / 0.2 / 0.3`
- focal gamma：`1.0 / 1.5 / 2.0 / 2.5`
- heavy augmentation 强度：
  - `random_translate_frac`
  - `random_scale_min`
  - `random_scale_max`
- 是否保留 class-balanced sampler
- 若你判断有必要，可对 batch size / scheduler / epochs 做少量合理调整

执行要求：
- 中小规模搜索阶段，将 `epochs` 设为 15 或完整训练 epoch 的约 30%，用于快速筛选趋势。
- 固定随机种子，避免把偶然波动当成结论。
- 至少完成 6 组以上有效对比实验。
- 先筛出 Top-3 组合，再用更完整训练验证。

### Phase C：结构增强实验
结构实验应尽量控制变量，避免把结构变化和大量超参数变化混在一起。

第一轮结构实验要求：
- 固定使用 R0 配置作为结构基线，分别评估：
  - `R0`
  - `R0 + GeM`
  - `R0 + CBAM`
  - 如有价值，再做 `R0 + GeM + CBAM`

要求：
1. `GeM Pooling`
   - 可参考 `src/waferlab/models/resnet_recall_opt.py` 的实现思路
   - 评估 GeM 单独加入的效果
2. `CBAM` 或替代轻量注意力
   - 需说明插入位置与理由
   - 优先选择不会过度增加复杂度、且适合 wafer 缺陷空间模式的插法
3. 若 GeM 或 CBAM 在 R0 上显示明确收益，再与 Phase B 的最优调参方案做组合验证一次，避免错过“调参最优 + 结构最优”的最终组合。

### Phase D：综合判断与完整训练
1. 综合比较以下几类方案：
   - 纯调参最优 ResNet50
   - `+ GeM`
   - `+ CBAM` 或替代轻量注意力
   - 如有效，再比较组合方案
2. 选出最终方案。
3. 对最终方案进行完整训练。
4. 跑完整评估，给出最终结果汇总。

## 评分与筛选规则
在中小规模搜索阶段，使用统一评分函数：

`score = 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall_mean + 0.15 * accuracy`

其中 `minority_recall_mean` 仅基于以下 5 个类的 recall 计算算术平均：
- `Center`
- `Donut`
- `Loc`
- `Near-full`
- `Scratch`

若两个模型 score 接近，优先选择：
1. 少数类 recall 更稳的
2. 结构更简单的
3. 更容易复现的

## 工程实现要求
1. 新增模型变体命名应清晰，例如：
   - `resnet50_gem`
   - `resnet50_cbam`
   - `resnet50_gem_cbam`
2. 新增配置命名应体现实验意图，例如：
   - `wm811k_resnet50_focal_balanced_tuned_multiclass.yaml`
   - `wm811k_resnet50_gem_multiclass.yaml`
   - `wm811k_resnet50_cbam_multiclass.yaml`
3. 不要把多个完全不同的实验硬塞进同一个配置文件。
4. 调节 Heavy Augmentation 强度时，必须保持 wafer-safe 原则。
5. 每个实验都应有清晰输出目录，并能从输出目录追溯到配置文件。

## 过程要求
1. 每个阶段结束后都要给出结论与下一步理由。
2. 遇到报错或结果异常，优先自行修复，不要把半成品当最终交付。
3. 重要结果必须形成表格，便于横向对比。
4. 若发现某种策略明显无效，应尽早淘汰，避免浪费完整训练预算。
5. 可参考近年相关论文或资料，但以本仓库中可落地、可复现、可验证的实验结果为最终依据。

## 最终交付物（必须完整）
1. 新增/修改文件清单。
2. 全部实验对比表，至少包含：
   - 配置名
   - 关键改动
   - accuracy
   - macro_recall
   - macro_f1
   - minority_recall_mean
3. 最终最优模型结构说明。
4. 最终最优配置路径。
5. 最终训练输出目录、checkpoint 路径、评估结果路径。
6. 完整复现命令：
   - 环境激活
   - 训练
   - 评估
7. 被淘汰方法及原因总结：
   - 为什么某些调参方向无效
   - GeM 是否值得保留
   - CBAM 或替代注意力是否值得保留

## 终止条件
仅当以下全部满足时才停止：
1. R0 完整跑通并记录结果。
2. 完成中小规模参数搜索并选出 Top 方案。
3. 完成 GeM / CBAM 或替代轻量注意力的对比实验。
4. 完成最终组合验证并确定最终模型。
5. 形成最终版本模型并完成完整训练。
6. 交付完整结果、结论与复现命令。
