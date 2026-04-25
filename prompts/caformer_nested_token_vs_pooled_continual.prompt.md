# 任务名称
在 CAFormer / ViT 路线上引入完整 Nested Learning 组件，并对比 token 化与不 token 化两版持续学习防遗忘效果

## 环境要求（强制执行）
- 所有 Python 命令、训练、评估、脚本执行前，必须先执行：
  `source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch`
- 全部工作必须在名为 `torch` 的 conda 环境中进行，不得使用其他 Python 环境。
- 工作目录必须是项目根目录：
  `/home/fu1fan/Develop/torch/wafer-defect-lab`
- 如果需要访问外部 GitHub 仓库，优先克隆到 `/tmp` 或其他临时目录，不要把整个外部仓库直接复制进本项目。
- 当前仓库可能已有未提交研究文件。开始前必须执行 `git status --short`，不要覆盖或回滚已有改动。

---

## 你的角色

你是高级机器学习工程师，任务不是“随手加一个 head”，而是把外部 Nested Learning 复现仓库中的关键机制正确迁移到当前 WM-811K wafer defect 任务中，并完成 **两条可比实验线**：

1. **Token 化版**：CAFormer spatial tokens + HOPE / SelfMod / CMS
2. **不 token 化版**：CAFormer pooled vector + 同一套 HOPE / SelfMod / CMS

必须保证两版除了 token 入口不同，其余训练、持续学习协议、损失、replay、采样和评估尽量一致。目标是判断：**在 ViT/CAFormer 路线上，Nested Learning 的防遗忘机制是否需要 token 序列才能发挥作用。**

---

## 外部参考仓库（必须阅读）

必须参考这个 GitHub 仓库：

- https://github.com/kmccleary3301/nested_learning

这是 Nested Learning / HOPE 的机制级复现仓库。不要只看 README；必须读源码和文档。该仓库 README 明确强调它实现了：

- HOPE / CMS / Self-Modifying Titans update rules + wiring
- teach-signal / delta-loss / CMS chunking / causality 的张量级测试
- boundary-target online chunking
- surprise gating
- stop-grad online writes
- continual-learning forgetting evaluation harness

### 必须定位并参考的模块 / 符号

克隆后必须用 `rg` 定位以下符号和文件。不同版本路径可能有小变动，所以以符号搜索为准：

```bash
rg -n "class HOPESelfModBlock|HOPESelfModBlock" /tmp/nested_learning_ref/src /tmp/nested_learning_ref/tests /tmp/nested_learning_ref/configs
rg -n "class HOPEBlockConfig|HOPEBlockConfig" /tmp/nested_learning_ref/src /tmp/nested_learning_ref/tests /tmp/nested_learning_ref/configs
rg -n "class HOPEBlock|HOPEBlock" /tmp/nested_learning_ref/src /tmp/nested_learning_ref/tests /tmp/nested_learning_ref/configs
rg -n "class CMS|Continuum|chunk|surprise|teach_signal|delta" /tmp/nested_learning_ref/src /tmp/nested_learning_ref/tests
rg -n "hope_hybrid|pilot_selfmod_paper_faithful|memorize|continual" /tmp/nested_learning_ref
```

必须重点阅读并提炼以下内容：

1. `HOPESelfModBlock`
   - 这是优先迁移的主线组件。
   - 重点看 Self-Modifying Titans 与 CMS 的连接方式。
   - 不要用当前仓库里简化版 `SelfModifier + CMSBlock` 冒充完整实现。

2. `cms` / Continuum Memory System
   - 它是防遗忘核心。
   - 必须理解 fast / slow memory level、chunk accumulation、update cadence、surprise gating、stop-grad write semantics。
   - 当前仓库已有 `src/waferlab/engine/nested_learning/cms.py`，但它是简化版；需要对照外部实现补足关键机制，而不是重复造一个普通 residual MLP。

3. `hope_hybrid -> HOPEBlockConfig + HOPEBlock`
   - 这是组件最全面的一条线，包含更完整的 HOPE wiring。
   - 第一阶段不一定直接上 full `hope_hybrid`，但必须读懂它，并在日志中说明哪些组件被迁移、哪些暂缓、为什么。
   - 如果 `HOPESelfModBlock` 已经足够支持两版对比，先用它完成主实验；`HOPEBlockConfig + HOPEBlock` 作为 full-components 后续扩展。

4. 机制文档
   - `docs/PAPER_COMPLIANCE.md`
   - `docs/STREAMING_CONTRACT.md`
   - `docs/stability_journal.md`
   - `docs/future_directions.md`

5. 测试文件
   - 搜索并阅读涉及 `teach_signal`、`delta`、`CMS chunking`、`causality`、`surprise`、`memorize` 的 tests。
   - 迁移后必须给当前仓库加最小张量级 smoke / unit test，验证输入输出 shape、teach path、CMS update path 不崩。

### 外部仓库使用边界

- 可以参考或小范围迁移代码，但不要整仓复制。
- 不要引入它的 Hydra / uv / tokenizer / LLM training pipeline。
- 当前项目仍使用本仓库的 `scripts/train_classifier.py` 和 `scripts/train_continual.py`。
- 如果外部实现依赖当前环境没有的包，优先改写成纯 PyTorch 小模块，而不是扩大依赖。

---

## 当前项目背景（必须先读）

### 研究日志

必须先阅读：

- `research_logs/00_project_timeline.md`
- `research_logs/vit_multiclass/00_overview.md`
- `research_logs/vit_multiclass/phase_i_data_improvement.md`
- `research_logs/vit_multiclass/phase_j_clean_calibration.md`（如果存在）
- `research_logs/nested_learning/00_overview.md`
- `research_logs/nested_learning/phase_a_tokenized_nested_selfmod.md`
- `research_logs/nested_learning/phase_b_detokenized_cms_resnet.md`
- `research_logs/nested_learning/phase_c_kd_bias_correction.md`

必须吸收的历史结论：

1. 旧 tokenized `NestedSelfMod` 失败，不代表 CAFormer token 路线必然失败。
   - 旧路线是弱 CNN stem + patch embedding，token 表征不足。
   - 新路线应使用 CAFormer 中后层 spatial feature / token，语义更强。

2. 去 token 化 `nested_cms_resnet` 比旧 token 化路线更稳定。
   - 因此必须保留 pooled-vector 非 token 版作为强对照。

3. KD 曾经降低 forgetting，但会与最后任务中的 `none` 类冲突。
   - 第一轮 CAFormer Nested continual 实验禁止默认开启 KD。
   - KD 只能作为第二轮扩展，不可在首轮混入主变量。

4. 当前最高静态分类路线是 CNN + ViT ensemble。
   - 但本任务的目标不是再做 ensemble，而是验证 CAFormer / ViT 路线上 Nested Learning 是否能改善 continual 防遗忘。

### 当前关键代码

必须阅读：

- `src/waferlab/models/vit_backbones.py`
- `src/waferlab/models/nested_selfmod.py`
- `src/waferlab/models/nested_cms_resnet.py`
- `src/waferlab/engine/nested_learning/`
- `src/waferlab/engine/trainer.py`
- `scripts/train_classifier.py`
- `scripts/train_continual.py`
- `configs/modal/research_vit/`
- `configs/modal/research_nest/`

重点确认：

- `TimmViTWrapper.forward_features()` 当前返回的是全局 pooled feature，不一定保留 spatial tokens。
- 如需 token 化版，不能只调用现有 `forward_features()`；必须从 CAFormer backbone 中提取 spatial feature map / token grid。
- `Trainer._nested_teach_step()` 当前对 vector-based nested 模型有硬编码假设，可能需要泛化。

---

## 总目标

实现并对比两个新模型：

### A. Token 化版：`caformer_hope_token`

结构建议：

```text
input wafer map [B, 1, 224, 224]
  -> CAFormer-S18 backbone / stages
  -> spatial feature map or token grid
  -> flatten tokens [B, T, C]
  -> projection [B, T, D]
  -> HOPESelfModBlock / CMS / SelfMod stack
  -> token pooling
  -> classifier [B, 9]
```

要求：

- `T` 优先控制在 49 或类似量级；不要使用过大的 raw patch token 数导致训练爆炸。
- token 必须来自 CAFormer 的中后层语义 feature，不要回退到 raw patch embedding。
- 必须支持 `forward_with_teach(x, teach_signal, surprise_value)`。
- 必须支持 continual training 中的 nested teach path。

### B. 不 token 化版：`caformer_hope_pooled`

结构建议：

```text
input wafer map [B, 1, 224, 224]
  -> CAFormer-S18 backbone.forward_features()
  -> pooled feature [B, C]
  -> projection [B, D]
  -> reshape [B, 1, D]
  -> 同一套 HOPESelfModBlock / CMS / SelfMod stack
  -> classifier [B, 9]
```

要求：

- 除 token 入口外，HOPE/CMS 模块和训练协议应尽量与 token 化版一致。
- 这是消融对照，不是低配实现；不要偷换成普通 MLP head。

---

## 实现要求

### 文件建议

优先新增：

- `src/waferlab/engine/nested_learning/hope_blocks.py`
- `src/waferlab/models/caformer_hope.py`
- `configs/modal/research_nest/models/caformer_hope_token.yaml`
- `configs/modal/research_nest/models/caformer_hope_pooled.yaml`
- `configs/modal/research_nest/experiments/caformer_hope_token_smoke.yaml`
- `configs/modal/research_nest/experiments/caformer_hope_pooled_smoke.yaml`
- `configs/modal/research_nest/experiments/caformer_hope_token_continual_s1.yaml`
- `configs/modal/research_nest/experiments/caformer_hope_pooled_continual_s1.yaml`
- `research_logs/nested_learning/phase_d_caformer_hope_token_vs_pooled.md`

如果你判断文件命名需要调整，可以调整，但必须保持含义清晰。

### 模型注册

必须通过 `MODEL_REGISTRY` 注册，例如：

```python
@MODEL_REGISTRY.register("caformer_hope_token")
@MODEL_REGISTRY.register("caformer_hope_pooled")
```

两个注册项可以共用同一个类，通过 config 区分：

```yaml
model:
  arch: caformer_hope_token
  base_arch: caformer_s18_wafer
  token_mode: spatial
```

```yaml
model:
  arch: caformer_hope_pooled
  base_arch: caformer_s18_wafer
  token_mode: pooled
```

### 必备接口

模型必须提供：

- `forward(x) -> logits`
- `forward_features(x) -> feature`
- `forward_tokens(x) -> tokens [B, T, D]`
- `forward_with_teach(x, teach_signal=None, surprise_value=None) -> logits`
- `get_cam_target_layer()`
- `num_classes`
- `in_channels`
- `get_token_dim()`
- `get_num_tokens()`

`Trainer._nested_teach_step()` 如有必要要泛化为：

- 优先调用 `model.forward_tokens(x)` 获取 pre-HOPE tokens。
- 再调用 `model.forward_from_tokens(tokens)` 或等价路径计算 logits。
- 不要继续硬编码 `model.backbone/global_pool/proj/nested_blocks/norm/fc`。

---

## 实验设计

### Phase 0：外部仓库审计与迁移计划

必须先完成并记录：

1. 外部仓库 clone / inspect 路径。
2. `HOPESelfModBlock`、`CMS`、`HOPEBlockConfig`、`HOPEBlock` 的实际源码路径。
3. 你选择迁移哪些机制：
   - 必须迁移：CMS fast/slow levels、teach signal update、surprise gating、stop-grad online write 语义。
   - 优先迁移：chunk update / update cadence。
   - 可暂缓：attention-cache carry、LLM-specific streaming boundary target。
4. 哪些机制不迁移，以及原因。

必须写入：

- `research_logs/nested_learning/phase_d_caformer_hope_token_vs_pooled.md`

### Phase 1：实现 + 张量级 smoke

先做最小张量测试，不训练：

- 构建 token 模型，输入 `[2, 1, 224, 224]`，验证输出 `[2, 9]`。
- 构建 pooled 模型，输入 `[2, 1, 224, 224]`，验证输出 `[2, 9]`。
- 验证 token 模型 `forward_tokens()` 结果是 `[B, T, D]` 且 `T > 1`。
- 验证 pooled 模型 `forward_tokens()` 结果是 `[B, 1, D]`。
- 验证 `forward_with_teach()` 不崩。
- 验证反向传播能跑通。

### Phase 2：训练入口 smoke

必须分别跑两版 continual smoke：

```bash
python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_token_smoke.yaml \
  --output-dir outputs/caformer_hope/token_smoke \
  --smoke-test
```

```bash
python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_pooled_smoke.yaml \
  --output-dir outputs/caformer_hope/pooled_smoke \
  --smoke-test
```

如果 smoke fail，必须修复；不能跳过。

### Phase 3：中测对比实验

第一轮只跑无 KD 版本，避免重复旧问题。

建议基础协议：

```yaml
continual:
  num_tasks: 3
  classes_per_task: 3
  freeze_slow_cms: false
  class_weight_per_task: true
  class_weight_mode: sqrt_inv_freq
  balanced_sampling: true
  max_samples_per_class: 5000
  epochs_per_task: [3, 3, 3]
  load_best_final: false
  task_order:
    - [1, 2, 3]
    - [4, 5, 6]
    - [7, 8, 0]
  replay:
    enabled: true
    exemplars_per_class: 1000
  knowledge_distillation:
    enabled: false
```

必须分别运行：

```bash
python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_token_continual_s1.yaml \
  --output-dir outputs/caformer_hope/token_s1
```

```bash
python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_pooled_continual_s1.yaml \
  --output-dir outputs/caformer_hope/pooled_s1
```

### Phase 4：如果第一轮有效，再做 task-order 改进

旧 task order 把 `none` 放在最后，会制造强分布冲击。第一轮为了和旧 Nested 结果可比可以保留旧顺序；第二轮必须测试更合理的 `none` 处理方式。

至少设计一个改进协议，例如：

```yaml
task_order:
  - [0, 1, 3]     # none + Center + Edge-Loc
  - [4, 5, 8]     # Edge-Ring + Loc + Scratch
  - [2, 6, 7]     # Donut + Near-full + Random
```

或实现每个 task 保留少量 `none` background replay。必须说明选择理由。

---

## 评价指标

不能只看 `avg_accuracy_final`。

必须报告：

1. Continual 指标：
   - `avg_accuracy_final`
   - `avg_forgetting`
   - task accuracy matrix
   - old-task retention
2. Full-test 指标：
   - overall accuracy
   - macro accuracy
   - per-class accuracy / recall
   - `none` accuracy
   - `Loc`
   - `Edge-Loc`
   - `Scratch`
3. 与旧 Nested 基线比较：
   - `nested_cms_resnet_r2`: `avg_accuracy_final=0.6037`
   - `cms_resnet_refine_r2`: full-test `overall_accuracy=0.9258`, macro around `0.4957`
   - `v4_ref`: no-KD reference around `avg_accuracy_final=0.443`, macro around `0.389`
4. 与静态 ViT/CNN 参考只做背景比较，不作为 continual 主指标：
   - ViT I5 raw `score≈0.8058`
   - CNN G2 `score≈0.8213`
   - CNN+ViT ensemble clean split expected `score≈0.830`

---

## 红线约束

1. 不要把当前简化版 `NestedBlock` 小修一下就宣称完成 HOPE 迁移。
2. 不要只做 token 化版；必须同时做 token 和 pooled 两版。
3. 不要只做 pooled 版；token 化版必须 `T > 1`。
4. 不要默认开启 KD；首轮必须无 KD。
5. 不要为了高 overall accuracy 牺牲 `none` 或少数类，然后只报 overall。
6. 不要覆盖已有配置和日志。
7. 不要改变 WM-811K 数据 split 或评分定义，除非明确记录为新实验协议。
8. 不要引入 Mixup / CutMix / ColorJitter / GaussianBlur / GridMask 等破坏 wafer 离散空间语义的增强。
9. 不要把外部 LLM 训练 pipeline、Hydra 全套配置、tokenizer 逻辑搬进来。
10. 不要把 `hope_hybrid` 里的全部组件无差别堆上去。先做可解释、可消融的最小完整机制，再扩展。

---

## 交付要求

最终必须交付：

1. 新增/修改文件清单。
2. 外部仓库参考摘要：
   - GitHub URL
   - 参考 commit hash
   - 参考源码路径
   - 迁移组件列表
3. 两版模型结构说明：
   - token 化版如何取 tokens
   - 不 token 化版如何构造 `[B,1,D]`
   - 两版共享哪些 HOPE/CMS 参数
4. smoke test 命令和结果。
5. 中测实验命令和结果表。
6. token vs pooled 的明确判断：
   - 哪个更稳
   - 哪个更防遗忘
   - 哪个对 `none` / `Loc` / `Edge-Loc` 更友好
7. 失败项和淘汰原因。
8. 下一步建议：
   - 是否继续 full `hope_hybrid / HOPEBlock`
   - 是否引入 KD
   - 是否改 task protocol
   - 是否做完整训练

---

## 推荐执行顺序摘要

1. `git status --short`
2. 读本项目日志和现有模型/训练代码
3. clone / inspect `https://github.com/kmccleary3301/nested_learning`
4. 写迁移计划到 `research_logs/nested_learning/phase_d_caformer_hope_token_vs_pooled.md`
5. 实现 HOPE/CMS 机制模块
6. 实现 `caformer_hope_token` 和 `caformer_hope_pooled`
7. 张量级 smoke
8. `train_continual.py --smoke-test` 两版 smoke
9. 3 epoch/task 无 KD 中测对比
10. 汇总结果，判断 token 化是否值得继续

不要在第 5 步之前开始长训练。
