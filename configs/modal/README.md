# `configs/modal`

这里放的是分类模型相关配置，采用“基线 + 模型 + recipe + 实验”四层组合，避免为一个小实验复制整份 YAML。

## 目录说明

- `base/`: 数据集与通用训练默认项
- `models/`: 模型结构及模型专属参数
- `recipes/`: 损失函数、类别权重、优化策略等训练方案
- `experiments/`: 最终可直接传给脚本的实验入口配置

## 用法

训练脚本支持 `_base_` 递归继承，路径相对当前 YAML 文件解析。

```yaml
_base_:
  - ../base/wm811k_classifier.yaml
  - ../models/resnet18_recall_opt.yaml
  - ../recipes/focal_recall.yaml
```

运行示例：

```bash
python scripts/train_classifier.py \
  --config configs/modal/experiments/wm811k_resnet18_recall_opt_focal.yaml
```

## 约定

- 模型差异放到 `models/`，不要混进实验文件
- 损失函数或优化策略差异放到 `recipes/`
- `experiments/` 尽量只做组合，必要时再覆盖少量字段

异常检测 dataloader 配置已经移到 `configs/anomaly/`，不再和分类训练配置混放。

## Prototype Memory（Nested-Learning 启发）

`recipes/prototype_ce.yaml` 启用基于 surprise-gated class prototype 的训练增强。
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
  --config configs/modal/experiments/wm811k_resnet18_prototype.yaml --smoke-test
```
