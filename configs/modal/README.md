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
