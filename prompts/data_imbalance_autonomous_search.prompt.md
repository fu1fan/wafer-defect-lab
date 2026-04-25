# 任务名称
围绕 WM-811K 多分类任务自主探索数据层面不平衡处理方法，验证平滑重采样、类感知增强及相关扩展方向能否超越当前主线

## 环境要求（强制执行）
- 所有 Python 命令、训练、评估、脚本执行前，必须先执行：
  `source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch`
- 全部工作必须在名为 `torch` 的 conda 环境中进行，不得使用其他 Python 环境。
- 确保工作目录始终位于项目根目录 `/home/fu1fan/Develop/torch/wafer-defect-lab`。
- GPU 训练前确认 `nvidia-smi` 可用；若 GPU 不可用，使用 CPU 做 smoke test。

## 你的角色
你是可自主执行的高级机器学习工程师。任务核心是：

1. 阅读项目代码与研究日志，理解当前状态和瓶颈。
2. 从 Phase I 消融实验出发，系统验证已实现的数据层改进。
3. 基于中测结果自主判断，向更多未探索方向扩展。
4. 把胜出方向与当前最优 CNN 主线做完整对比。

这不是"按固定脚本机械执行"。在关键节点你必须基于实验结果主动裁剪低价值方向，并独立决策后续路线。

---

## 项目背景（必须先读）

### 关键代码路径
- 训练入口：`scripts/train_classifier.py`
- 评估入口：`scripts/eval_classifier.py`
- cRT 脚本：`scripts/crt_retrain.py`
- cRT sweep：`scripts/crt_sweep.py`
- 数据与增强：
  - `src/waferlab/data/dataloaders.py`
  - `src/waferlab/data/transforms.py`
- 训练器与损失：
  - `src/waferlab/engine/trainer.py`
  - `src/waferlab/engine/losses.py`
- 损失注册：`src/waferlab/engine/losses.py`
- 配置继承风格：`configs/modal/base/`、`configs/modal/baseline/`

### 必须阅读的研究日志
- `research_logs/00_project_timeline.md`（重点看 2026-04-08 之后的所有阶段）
- `research_logs/resnet50_multiclass/00_overview.md`
- `research_logs/resnet50_multiclass/phase_d_final.md`
- `research_logs/resnet50_multiclass/phase_f_posthoc_crt.md`
- `research_logs/resnet50_multiclass/phase_g_crt_sweep.md`
- `research_logs/resnet50_multiclass/phase_h_loc_improvement.md`

---

## 当前状态（必须纳入决策的已知事实）

### 当前最优主线
- 模型：ResNet50 + GeM (`WaferClassifierGeM`)
- 训练方案：`G2_gamma1.0`（cRT focal γ=1.0, lr=1e-3, 1ep）
- 后处理：LogitAdj(τ=-0.05)
- **score=0.8213**（macro_f1 / macro_recall / minority_recall_mean / accuracy 参见日志）

### Phase D 单模型基线
- ResNet50 + GeM，25 ep，CB sampler + focal γ=1.5，lr=5e-4
- **score=0.8017**（checkpoint: `outputs/phase_d/` 或参见日志中路径）

### 类别分布（训练集）
```
none:       36730  (67.6%)   — 强多数类
Edge-Ring:   8554  (15.7%)
Center:      3462   (6.4%)
Edge-Loc:    2417   (4.5%)
Loc:         1620   (3.0%)   — 当前最困难的少数类之一
Random:       609   (1.1%)
Scratch:      500   (0.9%)
Donut:        409   (0.8%)
Near-full:     54   (0.1%)   — 当前采样倍数最大，过拟合风险最高
```

### 当前已实现的机制
- `WeightedRandomSampler` (pure 1/N, alpha=1.0)：过采样 Near-full ~680x（**过采样强度过大**）
- `WaferAugmentation`：翻转 + 90° 旋转 + translate(0.08) + scale(0.95~1.05)
- **新增（上次提交）：**
  - `_build_class_balanced_sampler(alpha)` — 平滑重采样，`N^{-α}` 权重
  - `WaferRandomErasing` — 晶圆安全随机擦除（填充 0/背景）
  - `ClassAwareAugmentation` — 少数类专用更强空间增强 + random erasing

### Phase I 已准备的消融实验配置
| 实验 | 文件 | 改动 | 目的 |
|---|---|---|---|
| I1 | `configs/modal/baseline/experiments/phase_i/I1_smooth_resample_075.yaml` | `sampler_alpha=0.75` | 平滑采样单独效果 |
| I2 | `configs/modal/baseline/experiments/phase_i/I2_classaware_aug.yaml` | `class_aware=true`，α=1.0 | 类感知增强单独效果 |
| I3 | `configs/modal/baseline/experiments/phase_i/I3_smooth_classaware_combined.yaml` | α=0.75 + class_aware | 组合效果 |
| I4 | `configs/modal/baseline/experiments/phase_i/I4_global_erasing.yaml` | `random_erasing_p=0.3`（全局） | 全局 erasing 正则化 |
| I5 | `configs/modal/baseline/experiments/phase_i/I5_smooth_resample_050.yaml` | `sampler_alpha=0.5` | 更激进平滑采样 |

---

## 统一评分函数
全程使用：

```
score = 0.40 * macro_f1 + 0.25 * macro_recall + 0.20 * minority_recall_mean + 0.15 * accuracy
```

其中 `minority_recall_mean` 基于：Center、Donut、Loc、Near-full、Scratch 五类。

此外必须单独跟踪 Loc 和 Edge-Loc 的 recall，因为两者是最难改善且最有分析价值的类别。

当两个方案 score 差距 ≤ 0.003 时，优先选 Loc recall 更高的方案。

---

## 红线约束（必须遵守，任何情况下不能违反）

1. **不能使用破坏 wafer map 离散空间语义的增强**：
   - 禁止：`Mixup`、`CutMix`、`ColorJitter`、`GaussianBlur`、`GridMask`、`SMOTE`、任何 GAN 生成增强
   - 允许的安全增强：翻转、90°旋转、平移（NEAREST 插值）、缩放（NEAREST 插值）、`WaferRandomErasing`（填0）

2. **不能修改已有核心逻辑**（除非有实验证据且在日志中说明原因）：
   - `_build_class_balanced_sampler` 的核心数学逻辑已被修改为支持 alpha，如果要再改必须说明
   - 不能修改 `WaferAugmentation._apply` 的插值方式（必须保持 NEAREST）

3. **不能覆盖已有实验配置文件**；所有新实验新建文件。

4. **不能只报 accuracy**；每次实验必须报 `score`、`macro_f1`、`macro_recall`、`minority_recall_mean` 与 per-class recall。

5. **中测阶段不得直接跳到完整训练**；必须先通过中测结果筛选，再进行 25 ep 完整训练。

6. **不得在训练中同时叠加多个重平衡机制**（例如 CB sampler + loss-level class weights + logit adj 同时作用于训练），避免双重过校正。LogitAdj 只在 cRT 之后后处理使用。

---

## 执行阶段（必须按顺序推进，但细节可自主调整）

### Phase 0：读代码（不执行训练）
1. 读 `src/waferlab/data/transforms.py`，确认 `WaferRandomErasing` 和 `ClassAwareAugmentation` 已存在，功能正确。
2. 读 `src/waferlab/data/dataloaders.py`，确认 `sampler_alpha`、`class_aware`、`minority_erasing_p` 等参数被正确读取并传递。
3. 读 `configs/modal/baseline/experiments/phase_i/*.yaml`，确认 5 组实验配置格式正确，`_base_` 继承路径存在。
4. 快速 smoke test：对 I1 做 `--smoke-test`，确认可以正常跑通：
   ```
   python scripts/train_classifier.py \
     --config configs/modal/baseline/experiments/phase_i/I1_smooth_resample_075.yaml \
     --output-dir outputs/phase_i/smoke_I1 \
     --smoke-test
   ```
5. 如果 smoke test 有报错，**先修复错误再继续**，不要跳过。修复内容要记录。

---

### Phase I：中规模消融筛选（~8 epochs）
**目标**：用中规模快速估计 5 个改动方向的相对收益，不要跑完整 25 ep。

中测 epoch 数：`8`（约为完整训练 32%，足以看出趋势，控制时间成本）。

**逐一执行以下实验**（顺序执行，不要并行）：

```bash
# I1: smooth resampling alpha=0.75
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_i/I1_smooth_resample_075.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/I1_smooth_075

# I2: class-aware augmentation only
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_i/I2_classaware_aug.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/I2_classaware_aug

# I3: combined smooth + class-aware
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_i/I3_smooth_classaware_combined.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/I3_combined

# I4: global random erasing p=0.3
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_i/I4_global_erasing.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/I4_global_erasing

# I5: aggressive smooth alpha=0.5
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_i/I5_smooth_resample_050.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/I5_smooth_050
```

作为对照，也运行标准基线 8 ep：
```bash
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_d/resnet50_gem_tuned.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/baseline_8ep
```

**每跑完一个，立即记录 score / macro_f1 / macro_recall / minority_recall_mean / Loc recall / Edge-Loc recall。**

**中测结果分析（决策节点 A）**：
在 6 组都跑完后，按如下规则进入 Phase II：
- 选出 score 最高的 1~2 个配置作为"主线候选"
- 如果 I1 ~ I5 中没有一个超过基线 8 ep，进入**扩展探索路径**（见附录 A）
- 如果 I3（组合）不优于 I1 和 I2 中的更优者，说明两者有冲突，后续只保留更好的单一机制

---

### Phase II：中测扩展（条件触发）

只要 Phase I 中有至少一个候选超过基线，就进行以下扩展。否则跳到附录 A。

#### 扩展方向 II-A：超参搜索最优配置的 alpha 精细化
如果 I1 或 I3 显示平滑采样有效，搜索更精细的 alpha 区间：
- 候选值：0.6、0.7、0.8（在 8 ep 筛选的基础上各跑一个）
- 以胜出的 augmentation 配置为基础（I2 或无 class_aware）

```bash
# 例，alpha=0.6
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/phase_i/I1_smooth_resample_075.yaml \
  --epochs 8 \
  --output-dir outputs/phase_i/II_alpha_060
# （需动态生成或修改 sampler_alpha=0.6 的配置，自行新建 yaml）
```

#### 扩展方向 II-B：少数类增强强度微调
如果 I2 或 I3 显示 class_aware 有效，测试 minority_translate_frac 和 minority_erasing_p：
- 候选：`translate=0.10 / erasing_p=0.2`（弱），`translate=0.15 / erasing_p=0.4`（强）

#### 扩展方向 II-C：新增数据层机制（仅在上述方向收益有限时考虑）
可以选择以下任一个新机制实现并测试（**不得同时引入多个**）：

1. **Per-class oversampling cap**（避免 Near-full 被极端复制）：
   - 给每类设置最大采样倍数上限，超出后 clip 到 max_ratio × majority_count
   - 适合 Near-full(54) 这种极端少数类
   - 实现方式：在 `_build_class_balanced_sampler` 中加 `max_oversample_ratio` 参数

2. **Minority-only hard augmentation schedule**（训练后半段强化少数类）：
   - 前 N/2 epoch 用标准 CB sampler
   - 后 N/2 epoch 切换为 ClassAwareAugmentation + smooth sampler
   - 需要在 `Trainer` 中支持 epoch-based dataloader 切换，或分两次调用训练

3. **Edge-Loc / Loc 专项 translate augmentation**（针对当前最困难类）：
   - 扩大 Edge-Loc 和 Loc 的 translate range 到 0.15~0.20（晶圆图中它们位置边缘，大平移可以模拟位置变化）
   - 在 `ClassAwareAugmentation` 中为特定类设置独立参数（当前 minority_classes 共享同一参数，可以细化）

从上述 II-C 中只选**最有理论依据且实现成本最低的一个**，8 ep 验证有效后再考虑是否加入主线。

---

### Phase III：最优配置完整训练
基于 Phase I + II 筛选出的 **1~2 个最优配置**，进行完整 25 ep 训练。

```bash
python scripts/train_classifier.py \
  --config <best_config.yaml> \
  --epochs 25 \
  --output-dir outputs/phase_i/<best_name>_full25ep
```

**完整训练后，必须对最优 checkpoint 做 cRT sweep**：

```bash
python scripts/crt_sweep.py
# （需要修改脚本中的 checkpoint 路径指向新训练的 best.pt，
#   或者新建一个 crt_sweep_phaseI.py 参照原脚本风格实现）
```

cRT sweep 应至少覆盖：
- `lr ∈ {5e-4, 1e-3, 2e-3}`
- `focal_gamma ∈ {1.0, 1.5}`
- 对每个 cRT checkpoint，扫 `tau ∈ [-0.20, 0.30]` 的 LogitAdj

记录全部结果，与当前最优 G2 主线（score=0.8213）做对比。

---

### Phase IV：结果分析与交付

完成全部实验后，输出以下内容：

#### 1. 消融汇总表
| 实验 | 改动 | 8ep score | 完整 score | cRT score | cRT+LogitAdj | vs G2 基线 |
|---|---|---|---|---|---|---|
| baseline | 无 | ? | 0.8017 | — | 0.8213 | — |
| I1 | α=0.75 | ? | ? | ? | ? | ? |
| I2 | class_aware | ? | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |

#### 2. 最优方案明确说明
- 配置文件路径
- checkpoint 路径
- 复现命令（完整可执行）

#### 3. 关键发现
- 平滑采样 alpha 的最佳值与机制分析
- class_aware augmentation 对哪些类别帮助最大
- Near-full 过采样 cap 是否有效（如果做了 II-C）
- Loc/Edge-Loc 是否得到了改善（核心判断指标）

#### 4. 结论
明确说明：
- 数据层改进是否超过当前主线（G2, score=0.8213）
- 如果没超过，还差多少、最可能的原因是什么
- 后续最值得尝试的一个方向是什么

---

## 附录 A：Phase I 全部负结果时的扩展探索路径

如果 5 组 Phase I 实验均未超过 8 ep baseline，说明：
- 平滑采样和类感知增强对当前模型没有正向作用，或者正向作用极小
- 此时不应该继续在数据层投入

**你应该做的是**：

1. 分析为什么没有改善：
   - 是 near-full 过采样减少后整体信号变弱？
   - 还是 augmentation 的随机性对少数类有害？
   - 检查各类 recall 的变化方向（对比每一类，不只看汇总指标）

2. 转向以下优先级更高的方向之一：
   - **方向 A1：MLP head + cRT（对当前 ViT 线程）**：把 CAFormer 的 fc 改为 2 层 MLP，重跑 cRT sweep，预期 ViT score 从 0.8097 提升到 ~0.815
   - **方向 A2：集成学习**：ResNet50+GeM 与 CAFormer-S18 的 logit 平均或加权集成，预期 score 超过 0.83
   - **方向 A3：Loc/Edge-Loc 数据审计**：手动抽查这两类样本，检查是否存在标注噪声（如果确实有混淆，是数据问题不是模型问题）

3. 不论选哪条路，执行前先写 5 行以内的方案简述，说明为什么选它。

---

## 自主决策权限（明确授权）

你拥有以下决策自由：

- 可以新建任意 yaml 配置文件（在 `configs/modal/baseline/experiments/phase_i/` 下）
- 可以修改 `src/waferlab/data/transforms.py` 和 `src/waferlab/data/dataloaders.py`（但修改前必须先阅读现有代码，且修改内容必须记录）
- 可以新建评估脚本或 cRT sweep 脚本（参照已有风格）
- 可以在 `research_logs/resnet50_multiclass/` 下新建 `phase_i_data_improvement.md` 记录结果（**必须记录**）
- 可以裁剪实验：如某方向中测连续两组均为负结果，可以主动终止该方向，不必强行跑完

你**不需要**：
- 每一步都等待人工确认
- 跑完所有排列组合
- 把每个失败实验都完整训练 25 ep

---

## 最终交付物
1. `research_logs/resnet50_multiclass/phase_i_data_improvement.md`：完整实验记录
2. 最优配置文件路径（可复现）
3. 最优 checkpoint 路径
4. 最终 score 与 G2 主线的对比表
5. 一段 200 字以内的结论：数据层改进到底有没有用、用在哪儿、值得后续继续投入吗
