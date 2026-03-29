# Interim Datasets

`data/interim/` 存放从原始数据整理出来的中间版本数据集，采用 `HDF5 + index.parquet + index.csv` 的形式组织，方便后续处理。

## 数据集格式

### WM-811K

- `wm811k.h5`
  - `maps`：变长 `uint8` wafer map，按一维展开存储
  - `map_shape`：每个样本对应的原始 `(height, width)`
  - `sample_id`、`orig_index`
  - `die_size`、`wafer_index`、`lot_name`
  - `split_label`、`failure_type`
  - `is_labeled`、`label_count`、`is_normal`
- `wm811k_index.parquet`
  - 推荐优先读取的主索引
  - 读取更快，类型保留更完整
- `wm811k_index.csv`
  - 便于快速查看、调试和临时导出

### MixedWM38

- `mixedwm38.h5`
  - `maps`：定长 `uint8` 张量，形状为 `[N, 52, 52]`
  - `labels`：多标签矩阵，形状为 `[N, 8]`
  - `label_names`
  - `sample_id`、`orig_index`
  - `label_count`、`is_labeled`、`is_normal`、`has_mixed_defect`
- `mixedwm38_index.parquet`
  - 推荐优先读取的主索引
  - 适合按列筛选和批量统计
- `mixedwm38_index.csv`
  - 便于快速查看多标签组合和样本统计

## 说明

- `interim` 层不是最终训练输入。
- `WM-811K` 保留原始可变尺寸结构，适合后续再做 pad / resize / mask。
- `MixedWM38` 原始就是固定 `52x52`，因此直接保存为稠密张量。

## 快速读取

```python
import h5py
import pandas as pd

with h5py.File("data/interim/mixedwm38.h5", "r") as f:
    maps = f["maps"]
    labels = f["labels"]

wm811k_index = pd.read_parquet("data/interim/wm811k_index.parquet")
```
