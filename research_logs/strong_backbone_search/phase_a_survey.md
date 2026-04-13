# Phase A：外部调研与候选筛选

## 背景与问题

ResNet50 + GeM 已是当前项目内静态分类最优主线（score=0.8017），但进一步提升受限：

- ResNet50 是 2015 年架构，ImageNet top-1 约 76%；现代 CNN 如 ConvNeXt-Tiny 达 82.5%，EfficientNetV2-S 达 84.2%
- 在 ResNet50 上，GeM 池化是唯一有效的结构增强（macro_f1 从 0.6738 提升到 0.7452）；CBAM 注意力反而有害（0.6972）
- 类不平衡仍是核心瓶颈：none 类占比 67.6%，Near-full 仅 54 样本，minority 类 recall 仍有明显提升空间
- 继续微调 ResNet50 超参收益已趋于边际（Phase B→D 各阶段 score 从 0.7295 到 0.8017，增量递减）

本阶段目标：通过外部调研，选出最有可能超越 ResNet50 的现代 backbone 候选，并整理适用于 WM-811K 的类不平衡处理策略。

## 采取措施

### 1. WM-811K / wafer map 分类经验检索

主要发现：
- WM-811K 上常见 backbone：ResNet、VGG、自定义浅层 CNN；近年论文开始使用 DenseNet、EfficientNet
- 晶圆图分类的核心挑战是极端类不平衡和形态相近类别混淆（如 Edge-Loc vs Edge-Ring, Center vs Loc）
- 数据增强必须保持 wafer map 的离散空间语义：翻转、90° 旋转、小幅平移/缩放是安全的；mixup/cutmix 会破坏缺陷图案的空间结构，不推荐
- 多数论文采用重采样 + 加权损失的组合处理不平衡

### 2. 长尾/类不平衡分类方法检索

关键方法汇总：
- **Focal Loss**：已在项目中使用，γ=1.5 效果最佳（Phase B 已验证 γ=2.0/2.5 反而有害）
- **Balanced Softmax**：在 softmax 中直接引入类别先验，理论上 Bayes 最优，实现简单
- **Logit Adjustment**：与 Balanced Softmax 类似的后验校正思路，训练时调整 logit
- **LDAM (Label-Distribution-Aware Margin Loss)**：为少数类强制更大分类间隔，理论上更适合极端不平衡
- **Class-Balanced Sampler**：已在项目中使用且效果显著，是数据层的基础配置
- **Decoupled Training**：先学表征再重平衡分类头，适合表征学习阶段和分类器调整分离
- **Mixup/CutMix**：在自然图像长尾分类中有效，但对 wafer map 离散语义不安全，**不采用**

### 3. 候选 Backbone 检索与筛选

考察了以下现代 CNN：

| 模型 | 参数量 | GFLOPS | ImageNet Top-1 | 评估 |
|------|--------|--------|----------------|------|
| ConvNeXt-Tiny | 28.6M | 4.5 | 82.5% | ✅ 首选候选 |
| EfficientNetV2-S | 21.5M | 8.4 | 84.2% | ✅ 高效候选 |
| ConvNeXt-Small | 50.2M | 8.7 | 83.1% | ✅ 大容量对比 |
| RegNetY-4GF | 20.6M | 4.0 | 80.0% | ❌ 提升幅度不够 |
| RepLKNet | 31M+ | 高 | 83.5% | ❌ 大核卷积实现复杂 |
| MobileNetV4 | 轻量 | 低 | ~75% | ❌ 精度不够 |
| DeiT/Swin | ~29M | 4.6 | 81-83% | ❌ ViT 对小数据集和离散图像不友好 |

### 筛选理由

1. **ConvNeXt-Tiny**（首选）：
   - 优点：现代化 CNN 设计（大核卷积、LayerNorm、GELU），ImageNet 82.5%，与 Swin-T 精度相当但为纯 CNN，迁移学习表现稳定
   - 风险：参数量较 ResNet50 大（28.6M vs 25.6M），需验证是否过拟合
   - 适合 WM-811K 的理由：纯 CNN 天然适合空间规则的 wafer map，预训练表征更强

2. **EfficientNetV2-S**（高效候选）：
   - 优点：ImageNet 84.2% 最高精度，参数效率极佳（21.5M），训练速度通过 progressive learning 优化
   - 风险：复合缩放可能在小图像上不如大图像增益明显
   - 适合 WM-811K 的理由：高参数效率意味着更少过拟合风险，强预训练有助于少样本类别

3. **ConvNeXt-Small**（大容量对比）：
   - 优点：50.2M 参数，ImageNet 83.1%，用于验证更大模型是否带来额外收益
   - 风险：参数量翻倍，过拟合风险更高，训练更慢
   - 适合 WM-811K 的理由：作为上界参考，如果大模型也无法显著提升则说明瓶颈不在模型容量

### 不采用的方法及理由

- **Mixup / CutMix**：wafer map 是离散空间图案（缺陷区域有特定空间位置语义），混合两张图会产生语义无效的合成样本。明确放弃。
- **生成式增强（SMOTE-image, GAN 等）**：对于仅 54 样本的 Near-full 类，生成质量无法保证，且验证成本高。不在本线程尝试。
- **ViT 系列**：WM-811K 图像为 96×96 的离散灰度图，patch embedding 对小图像效率低，且数据量不足以支撑 ViT 从头训练。预训练 ViT 迁移效果不确定，优先级低于纯 CNN。

## 结论

进入实验的候选方案：

1. **ConvNeXt-Tiny** + 多种损失函数（focal / balanced_softmax / logit_adj / LDAM）
2. **EfficientNetV2-S** + 多种损失函数
3. **ConvNeXt-Small** + focal（仅作大容量参考）

统一保持 CB sampler + 现有安全增强策略，通过损失函数变体控制变量。

## 下一步

Phase B：将 3 个候选 backbone 接入项目工程，完成注册、配置和冒烟验证。
