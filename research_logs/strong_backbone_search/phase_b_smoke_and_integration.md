# Phase B：工程接入与冒烟验证

## 背景与问题

Phase A 确定了 3 个候选 backbone（ConvNeXt-Tiny, EfficientNetV2-S, ConvNeXt-Small）和 4 种损失函数（focal, balanced_softmax, logit_adjustment, LDAM）。本阶段需要在不破坏现有代码的前提下完成工程接入，并验证所有组合可正常运行。

## 采取措施

### 1. 新增模型文件

创建 `src/waferlab/models/modern_backbones.py`，包含 3 个模型类：

- `WaferConvNeXtTiny`：基于 `torchvision.models.convnext_tiny(pretrained=True)`
  - 关键细节：ConvNeXt 的 `classifier` 结构为 `[LayerNorm2d, Flatten, Linear]`，不能直接替换整个 classifier 为 Identity（会丢失 norm 和 flatten），需要只替换 `classifier[2]`（最后的 Linear 层）
  - 特征维度：768
  - 注册名：`convnext_tiny_wafer`

- `WaferEfficientNetV2S`：基于 `torchvision.models.efficientnet_v2_s(pretrained=True)`
  - EfficientNetV2 的 `classifier` 为 `[Dropout, Linear]`，avgpool 已经 flatten，可以安全地替换整个 classifier 为 Identity
  - 特征维度：1280
  - 注册名：`efficientnetv2_s_wafer`

- `WaferConvNeXtNano`：基于 `torchvision.models.convnext_small(pretrained=True)`
  - 与 ConvNeXt-Tiny 相同的 classifier 处理方式
  - 特征维度：768
  - 注册名：`convnext_small_wafer`

所有模型均遵循项目规范：
- 通过 `MODEL_REGISTRY.register()` 注册
- 实现 `forward()`, `forward_features()`, `get_cam_target_layer()` 接口
- 兼容现有 `WaferClassifier` 的 API 模式
- 保留 `forward_features()` 以便后续持续学习复用

### 2. 新增 LDAM Loss

在 `src/waferlab/engine/losses.py` 中新增 `LDAMLoss` 类：
- 基于论文 "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
- 根据类别频率计算每类 margin：$\Delta_j = C / n_j^{1/4}$
- 在训练时对 logit 减去对应 margin，再计算交叉熵
- 在 `trainer.py` 中添加 `loss_type: ldam` 分支

### 3. 配置文件

模型配置（`configs/modal/baseline/models/`）：
- `convnext_tiny.yaml`
- `efficientnetv2_s.yaml`
- `convnext_small.yaml`

Recipe 配置（`configs/modal/baseline/recipes/`）：
- `cb_sampler_balanced_softmax_multiclass.yaml`
- `cb_sampler_logit_adj_multiclass.yaml`
- `cb_sampler_ldam_multiclass.yaml`

实验配置（`configs/modal/baseline/experiments/phase_e_backbone/`）：
- `E1_convnext_tiny_focal.yaml` — ConvNeXt-Tiny + CB sampler + focal(γ=1.5)
- `E2_convnext_tiny_balanced_softmax.yaml` — ConvNeXt-Tiny + CB sampler + balanced_softmax
- `E3_efficientnetv2_s_focal.yaml` — EfficientNetV2-S + CB sampler + focal(γ=1.5)
- `E4_efficientnetv2_s_balanced_softmax.yaml` — EfficientNetV2-S + CB sampler + balanced_softmax
- `E5_convnext_tiny_logit_adj.yaml` — ConvNeXt-Tiny + CB sampler + logit_adjustment
- `E6_efficientnetv2_s_logit_adj.yaml` — EfficientNetV2-S + CB sampler + logit_adjustment
- `E7_convnext_small_focal.yaml` — ConvNeXt-Small + CB sampler + focal(γ=1.5)
- `E8_convnext_tiny_ldam.yaml` — ConvNeXt-Tiny + CB sampler + LDAM

所有实验统一 5 epochs, lr=5e-4, cosine schedule, 安全增强。

### 4. 冒烟验证

对所有 3 个 backbone × 4 种损失函数组合进行了冒烟验证：
- 能正常加载预训练权重
- 能完成 forward pass
- 损失函数能正确计算
- 输出维度正确（9 类）

全部 12 个组合通过验证。

## 实验设置

- 配置文件：见上方列表
- 模型路径：`src/waferlab/models/modern_backbones.py`
- 损失修改：`src/waferlab/engine/losses.py`（新增 LDAM）、`src/waferlab/engine/trainer.py`（新增 ldam 分支）

## 结果汇总

| 验证项 | 状态 |
|--------|------|
| ConvNeXt-Tiny 加载 + forward | ✅ 通过 |
| EfficientNetV2-S 加载 + forward | ✅ 通过 |
| ConvNeXt-Small 加载 + forward | ✅ 通过 |
| Focal loss 兼容性 | ✅ 通过 |
| Balanced Softmax 兼容性 | ✅ 通过 |
| Logit Adjustment 兼容性 | ✅ 通过 |
| LDAM 兼容性 | ✅ 通过 |

### 遇到的问题

- **ConvNeXt classifier 结构不同于 ResNet**：最初将整个 `classifier` 替换为 `nn.Identity()` 导致维度错误（输出为 4D tensor 而非 2D）。原因是 ConvNeXt 的 classifier 包含 LayerNorm2d 和 Flatten 步骤。修复方案：只替换 `classifier[2]`（Linear 层），保留前两层。

## 分析与决策

所有候选 backbone 和损失函数均已成功接入，可以进入 Phase C 筛选实验。代码实现遵循了项目现有风格，未破坏任何已有功能。

## 下一步

Phase C：5 epoch 中小规模筛选实验，比较不同 backbone × 损失函数组合。
