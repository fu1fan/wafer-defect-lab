# 任务名称
围绕 WM-811K 的 ViT 多分类模型进行自主研究、工程接入、分阶段筛选与最终定版

## 环境要求（强制执行）
- 所有 Python 命令、训练、评估、脚本执行前，必须先执行：
  `source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch`
- 全部工作必须在名为 `torch` 的 conda 环境中进行，不得使用其他 Python 环境。
- 确保工作目录位于项目根目录。

## 你的角色
你是可自主执行的高级机器学习工程师。请在红线约束内自主完成：
- 阅读代码与研究日志
- 检索近年的 ViT / hierarchical ViT 资料
- 设计并实现适配当前仓库风格的模型与配置
- 做中小规模筛选实验
- 基于实验结果动态调整路线
- 对最优方向进行完整训练、评估与交付

这不是一个“按固定脚本机械执行”的任务。你必须主动思考、主动搜索资料、主动裁剪低价值方向，并在关键节点基于结果改变计划。

## 总目标
1. 在当前仓库中新增一条 `ViT` 多分类研究线，遵循现有工程风格完成代码接入。
2. 先做中小规模实验，判断哪类 ViT backbone 与训练策略最适合 `WM811K`。
3. 再围绕中测胜出的方向做定向改造，重点处理类不平衡与 `Loc / Edge-Loc` 等困难类别。
4. 最终交付一个经过完整训练和完整评估的最优 ViT 模型，并与当前 CNN 主线做公平对比。

## 项目背景（必须阅读并遵守）

### 数据与任务
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
- 输入为离散 wafer map，增强必须满足 wafer-safe 原则，不能破坏空间语义。

### 关键代码路径
- 训练入口：`scripts/train_classifier.py`
- 评估入口：`scripts/eval_classifier.py`
- 数据与增强：
  - `src/waferlab/data/dataloaders.py`
  - `src/waferlab/data/transforms.py`
- 训练器与损失：
  - `src/waferlab/engine/trainer.py`
  - `src/waferlab/engine/losses.py`
- 现有 backbone 参考：
  - `src/waferlab/models/resnet.py`
  - `src/waferlab/models/resnet50_variants.py`
  - `src/waferlab/models/modern_backbones.py`
- 注册机制：
  - `src/waferlab/registry.py`
- 配置继承风格：
  - `configs/modal/base/`
  - `configs/modal/baseline/`

### 研究日志（必须先读）
- `research_logs/00_project_timeline.md`
- `research_logs/resnet50_multiclass/00_overview.md`
- `research_logs/resnet50_multiclass/phase_a_baseline.md`
- `research_logs/resnet50_multiclass/phase_b_search.md`
- `research_logs/resnet50_multiclass/phase_c_structure.md`
- `research_logs/resnet50_multiclass/phase_d_final.md`
- `research_logs/resnet50_multiclass/phase_f_posthoc_crt.md`
- `research_logs/resnet50_multiclass/phase_g_crt_sweep.md`
- `research_logs/resnet50_multiclass/phase_h_loc_improvement.md`
- `research_logs/strong_backbone_search/00_overview.md`
- `research_logs/vit_multiclass/00_overview.md`
- `research_logs/vit_multiclass/phase_a_baseline.md`

## 已知事实（必须纳入决策）
1. 当前项目主线不是“谁的 ImageNet 精度高谁就更强”，而是“谁在 WM811K 上更好地平衡 macro_f1 / macro_recall / 少数类 recall”。
2. `ResNet50 + GeM` 的完整训练基线 `D1_gem_tuned` 结果为：
   - `accuracy = 0.9541`
   - `macro_f1 = 0.7485`
   - `macro_recall = 0.7988`
   - `minority_recall_mean = 0.7975`
   - `score = 0.8017`
3. 当前全项目最优主线为 `G2_gamma1.0 + LogitAdj(τ=-0.05)`：
   - `accuracy = 0.9591`
   - `macro_f1 = 0.7486`
   - `macro_recall = 0.8401`
   - `minority_recall_mean = 0.8399`
   - `score = 0.8213`
4. 现代 CNN（ConvNeXt-Tiny / EfficientNetV2-S）并没有超越 `ResNet50 + GeM`，说明瓶颈不单在 backbone 容量。
5. 现有日志强烈表明：
   - `class-balanced sampler` 是多分类长尾场景的强基线组件
   - `focal loss` 与 sampler 互补
   - 训练期直接叠加 `balanced softmax / logit adjustment / LDAM` 与 sampler 容易双重过校正
6. `Loc / Edge-Loc` 存在明显 tradeoff，是当前主线最难问题之一。ViT 线程需要重点观察这两个类是否出现更干净的分离。

## 自主搜索要求（必须执行）
你必须自行检索近年 ViT / hierarchical ViT / hybrid ViT 资料，并基于资料与工程可落地性决定首批实验候选。

### 检索规则
- 优先使用：
  - arXiv 论文
  - 官方代码仓库
  - 官方模型文档
  - `timm` / `torchvision` / Hugging Face 中稳定可获取的预训练权重
- 不要只按“论文新”排序，必须同时考虑：
  - 是否有成熟预训练权重
  - 是否容易接入当前仓库
  - 是否适合 224x224 输入
  - 是否适合中等规模数据迁移
  - 是否适合单通道 wafer map 适配

### 建议候选池（仅作起点，不是硬约束）
以下模型是建议优先检索与评估的起始池，你可以裁剪、替换或补充，但必须说明理由：
- `DeiT III-S/16`
- `DINOv2 ViT-S/14`
- `EVA-02-S/14`
- `TinyViT-21M`
- `MaxViT-Tiny`

如果你检索后发现：
- 某模型缺少稳定权重
- 某模型接入成本过高
- 某模型与当前数据规模明显不匹配
- 某模型更适合作为后续补充而非首批中测对象

你可以自主调整候选池，不需要逐一等待人工确认。

## 强约束（必须遵守）
1. 不要覆盖已有 baseline 配置文件；所有 ViT 相关代码、配置、脚本、日志都应新建。
2. 保持工程风格一致：
   - 模型通过 registry 注册
   - 训练统一走 `scripts/train_classifier.py`
   - 配置统一使用 `_base_` 继承
3. 新增模型优先放在：
   - `src/waferlab/models/vit_backbones.py`
4. 新增配置优先放在：
   - `configs/modal/research_vit/models/`
   - `configs/modal/research_vit/recipes/`
   - `configs/modal/research_vit/experiments/`
5. 新增研究日志优先写到：
   - `research_logs/vit_multiclass/`
6. 每次新增 backbone 或关键结构改动后，必须先做可运行性验证，再进入批量实验。
7. 所有实验必须记录，包含失败项、淘汰项和理由。
8. 最终必须给出完整复现命令与最优模型路径。

## 自主决策空间（明确授权）
为了充分发挥高级基座模型能力，你拥有较大的自主决策空间。你不应把路线锁死在预先设定的窄网格里。

### 你可以自主决定的内容
1. 首批 ViT 候选数目
   - 不要求把所有候选都跑一遍
   - 可以先从 2 到 4 个最有希望的模型开始
2. 依赖选择
   - 可引入 `timm` 等常见依赖，但必须保持改动可控并验证安装与加载流程
3. 输入适配方式
   - `3-channel repeat`
   - `1-channel patch_embed`
   - `hybrid stem`
   - 其他你认为更合理的灰度输入适配方法
4. 中测超参数空间
   - learning rate
   - weight decay
   - warmup
   - layer-wise lr decay
   - drop path / dropout
   - patch size / pooling strategy
5. 不平衡处理路线
   - 可继续沿用 `CB sampler + focal`
   - 也可在有依据时引入更适合 ViT 的重平衡策略
   - 但要先通过中测证明方向有效，避免盲目扩展
6. 后续结构改造
   - 可以是 token pooling
   - class-attentive pooling
   - dual-head / auxiliary head
   - head-only cRT
   - trunk freeze / partial unfreeze
   - 任何你认为更适合 ViT 的轻量改造

### 你必须避免的行为
- 不要把路线预设得过死，强行把所有资源砸在一条未经验证的路线。
- 不要在没有中测证据的情况下，直接进入昂贵完整训练。
- 不要为了“追新”引入实现极不稳定、权重不可得或难以复现的模型。
- 不要为了追高分隐藏失败结果。
- 不要只汇报 accuracy，必须同步报告 `macro_f1`、`macro_recall`、`minority_recall_mean` 与 per-class recall。

## 红线约束
- 严禁使用 `ColorJitter`、`RandomErasing`、`Cutout`、`GaussianBlur`、`GridMask`、`Mixup`、`CutMix` 或任何会破坏 wafer 离散空间语义的增强。
- 严禁直接修改已有 `class-balanced sampler` 核心逻辑，除非你能明确证明改动必要且更优，并在日志中说明原因、影响与回滚方案。
- 严禁把多个完全不同的实验硬塞进同一个配置文件。
- 严禁只因某模型“更高级”就默认它应替代当前主线；必须以可复现结果为准。

## 优先优化目标
按重要性排序：
1. `macro_f1`
2. `macro_recall`
3. `minority_recall_mean`
4. `accuracy`

其中 `minority_recall_mean` 至少基于以下 5 个类计算：
- `Center`
- `Donut`
- `Loc`
- `Near-full`
- `Scratch`

此外必须单独跟踪：
- `Loc`
- `Edge-Loc`
- `Edge-Ring`

因为这些类别最能反映 ViT 是否真正改善了空间模式建模。

## 统一评分函数
中小规模筛选与完整训练后的主比较指标统一为：

`score = 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall_mean + 0.15 * accuracy`

当两个方案 score 接近时，优先选择：
1. 困难类别 recall 更稳的
2. 结构更简单的
3. 更容易复现的
4. 更适合作为后续持续学习 backbone 的

## 实验总流程（必须执行，但允许你动态调整细节）

### Phase 0：阅读、检索与方向决策
1. 阅读项目代码与相关 research logs。
2. 检索候选 ViT / hierarchical ViT 资料。
3. 基于“预训练可得性 + 工程可接入性 + 对 WM811K 的潜在适配性”筛出首批中测候选。
4. 写出简短计划：
   - 为什么先测这几个
   - 为什么暂时不测另外一些

### Phase A：工程接入与冒烟验证
1. 接入首批 2 到 4 个候选 backbone。
2. 保持仓库风格：
   - registry 注册
   - `_base_` 配置继承
   - 训练/评估复用现有入口
3. 对每个 backbone 做最小可运行验证：
   - 权重加载
   - 单通道或三通道输入适配
   - forward pass 正常
   - 训练 1 epoch 不报错

### Phase B：中小规模筛选实验
1. 对首批候选做中测，而不是直接完整训练。
2. 中测轮数建议为完整训练的约 `20% ~ 35%`，可自行决定具体 epoch。
3. 第一轮原则：
   - 先控制变量，优先比较 backbone 与基础 recipe
   - 不要同时引入过多结构技巧
4. 第一轮推荐至少覆盖：
   - backbone 对比
   - `3ch repeat` vs `1ch patch_embed` 中的一种或两种
   - `CB sampler + focal` 基础 recipe
5. 如第一轮出现清晰信号，可主动缩小搜索空间，不必把剩余低价值实验全部跑完。

### Phase C：基于中测结果的定向开发
这一步不是固定网格搜索，而是“结果驱动开发”。

你必须基于前一阶段的现象决定后续路线，可能的方向包括但不限于：
- ViT 专属优化：
  - warmup
  - layer-wise lr decay
  - head re-init
  - drop path
  - token pooling
  - cls token / avg token / weighted token 对比
- 输入与 patch 粒度优化：
  - `patch16`
  - `patch8`
  - hybrid stem
- 长尾问题定向处理：
  - head-only cRT
  - freeze trunk + retrain head
  - logit adjustment only after cRT
  - 困难类 bias 分析

你不需要把上述方向全部执行，只需选择最有证据支撑的 2 到 4 条路线继续推进。

### Phase D：完整训练与最终比较
1. 选出中测阶段 Top-2 方案。
2. 对 Top-2 进行完整训练。
3. 在完整训练结果上再判断是否需要：
   - cRT
   - post-hoc logit adjustment
   - 少量定向改造
4. 最终与以下对象做对比：
   - `D1_gem_tuned`
   - 如有必要，也可对比当前全局最优 `G2_gamma1.0 + LogitAdj`

## 方向判断规则（必须使用）
你应当像研究者一样做“阶段性停损”和“主动换路”。

### 建议阈值
- 若 ViT 原始基线在中测阶段明显低于 `D1` 趋势，且困难类没有优势，应尽快换 backbone 或换输入适配方式。
- 若某 ViT backbone 在中测阶段已经接近 `D1`，应优先把资源投入该方向，而不是继续铺新模型。
- 若某路线只提升 accuracy，不提升 `macro_f1 / macro_recall / minority_recall_mean`，应判为低优先级。
- 若某路线使 `Loc / Edge-Loc` 结构明显改善，即使总 score 暂未登顶，也应保留为重点观察对象。

## 工程实现要求
1. 新增 backbone 命名清晰，例如：
   - `deit3_small_wafer`
   - `dinov2_vits14_wafer`
   - `tinyvit_21m_wafer`
2. 新增配置命名体现实验意图，例如：
   - `wm811k_deit3_s16_cbfocal_screen.yaml`
   - `wm811k_dinov2_vits14_screen.yaml`
   - `wm811k_tinyvit21m_full.yaml`
3. 每个实验有独立输出目录，并能追溯到配置文件。
4. 如果引入第三方模型库，确保：
   - requirements 或依赖说明更新
   - 权重加载逻辑稳定
   - 无网时仍可复现本地已下载权重的实验

## 过程要求
1. 每个阶段结束后都要写出结论与下一步理由。
2. 遇到报错或结果异常，优先自行修复。
3. 重要结果必须整理成表格，便于横向比较。
4. 如果发现某一方向明显无效，应主动终止该方向，避免浪费训练预算。
5. 研究日志必须持续更新，不能只留下 `outputs/` 而无正式结论。

## 最终交付物（必须完整）
1. 新增/修改文件清单。
2. 首批候选筛选理由。
3. 全部中测实验对比表，至少包含：
   - 配置名
   - 关键改动
   - accuracy
   - macro_recall
   - macro_f1
   - minority_recall_mean
   - score
4. 完整训练 Top-2 对比表。
5. 最终最优模型结构说明。
6. 最终最优配置路径。
7. 最终训练输出目录、checkpoint 路径、评估结果路径。
8. 完整复现命令：
   - 环境激活
   - 训练
   - 评估
9. 被淘汰方法及原因总结。
10. 对当前 CNN 主线的最终对比结论：
   - 是否值得替换
   - 若不值得，差距来自哪里
   - 若值得，优势体现在哪些类别

## 终止条件
仅当以下全部满足时才停止：
1. 完成代码接入并通过冒烟验证。
2. 完成首批 ViT / hierarchical ViT 中测筛选。
3. 基于中测结果完成至少一轮定向优化开发。
4. 对 Top-2 方案完成完整训练与完整评估。
5. 给出最终最优模型与复现命令。
6. 在 `research_logs/vit_multiclass/` 中留下完整阶段结论。
