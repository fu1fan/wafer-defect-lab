# `configs/modal`

这里放的是分类模型相关配置。

当前采用“共享 base + 分线路组织”的方式：
- `base/`: 全部训练线共享的数据集与通用训练默认项
- `baseline/`: 当前主线，可稳定复现和对比
- `research_hybrid/`: 自制 HOPE Hybrid 研究线
- `research_nest/`: Nested Learning 方向研究线

## 目录说明

- `baseline/models|recipes|experiments/`
- `research_hybrid/models|recipes|experiments/`
- `research_nest/models|recipes|experiments/`

## 用法

训练脚本支持 `_base_` 递归继承，路径相对当前 YAML 文件解析。

```yaml
_base_:
  - ../../base/wm811k_classifier.yaml
  - ../models/resnet18_recall_opt.yaml
  - ../recipes/focal_recall.yaml
```

运行示例：

```bash
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/wm811k_resnet18_recall_opt_focal.yaml
```

```bash
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/wm811k_resnet50_baseline_multiclass.yaml
```

```bash
python scripts/train_classifier.py \
  --config configs/modal/baseline/experiments/wm811k_resnet50_cb_sampler_focal_heavy_aug_multiclass.yaml
```

## 约定

- 共享配置放到 `base/`
- 每条线路内部的结构保持一致：`models/`、`recipes/`、`experiments/`
- 模型差异放到各线路的 `models/`，不要混进实验文件
- 损失函数或优化策略差异放到各线路的 `recipes/`
- `experiments/` 尽量只做组合，必要时再覆盖少量字段

异常检测 dataloader 配置已经移到 `configs/anomaly/`，不再和分类训练配置混放。

## 线路说明

- `baseline/`
  - `resnet18` baseline
  - `resnet50` multiclass baseline
  - `resnet50` multiclass + balanced sampler + focal + heavy aug
  - `resnet18_recall_opt` 及其 CE / Focal 变体
- `research_hybrid/`
  - 自制 HOPE Hybrid 研究线
  - 当前保留完整配置，但不再作为默认推荐入口
- `research_nest/`
  - Nested Learning 方向研究线
  - 当前首先落地的是低风险的 prototype-memory 版本
  - 后续如果继续引入 CMS / self-mod / HOPE block，也继续收敛在这条线

## Prototype Memory（Nested-Learning 启发）

`research_nest/recipes/prototype_ce.yaml` 启用基于 surprise-gated class prototype 的训练增强。
核心思想来自 Google Nested Learning 论文中的 CMS 多频记忆更新和 surprise 门控机制，
翻译为图像分类场景下的 per-class 特征原型 + 辅助对齐损失。

- **Prototype EMA**：训练期维护每类特征质心，以 EMA 缓慢更新
- **Surprise 门控**：仅在 per-sample loss 超过阈值时更新原型（跳过"太容易"的样本）
- **辅助损失**：将特征拉向所属类别原型，提供额外监督信号
- **仅训练阶段生效**：验证和测试不会使用或更新原型

```yaml
train:
  prototype:
    enabled: true
    momentum: 0.99            # EMA 动量（越高漂移越慢）
    surprise_threshold: 0.5   # per-sample loss 门控阈值
    aux_weight: 0.1           # 辅助损失权重
    warmup_epochs: 3          # 预热 epoch 数（先稳定原型再施加辅助损失）
```

使用方式：

```bash
python scripts/train_classifier.py \
  --config configs/modal/research_nest/experiments/wm811k_resnet18_prototype.yaml \
  --smoke-test
```
