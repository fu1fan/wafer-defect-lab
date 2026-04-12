# 研究日志规范

`research_logs/` 是本仓库的正式研究日志目录，用来保存**受 Git 跟踪**的阶段总结、关键决策和实验结论。

它与 `outputs/` 的分工不同：

- `outputs/` 保存本地训练产物、checkpoint、原始评估结果，不纳入版本管理。
- `research_logs/` 保存研究背景、采取措施、核心结果和阶段结论，纳入版本管理，方便师兄/合作者直接阅读。

## 目录约定

```text
research_logs/
├── README.md
├── 00_project_timeline.md
├── _templates/
│   ├── thread_overview.md
│   └── phase_log.md
├── resnet18_binary/
├── resnet18_multiclass/
├── nested_learning/
└── resnet50_multiclass/
```

- `00_project_timeline.md`：只负责全项目时间线与主线切换。
- 每个线程目录至少包含一个 `00_overview.md` 和若干 `phase_*.md`。
- 文件名统一使用英文蛇形命名，正文使用中文 Markdown。

## 线程概览文件要求

每个 `00_overview.md` 固定包含以下内容：

1. 研究目标
2. 当前最佳方案
3. 当前最佳指标
4. 线程内阶段列表
5. 当前结论
6. 下一步计划

目标是让读者只看这一页，就能知道该线程做到了哪里、最好结果是什么、接下来准备做什么。

## 阶段日志文件要求

每个 `phase_*.md` 固定采用相同结构：

1. 背景与问题
2. 采取措施
3. 实验设置
4. 结果汇总
5. 分析与决策
6. 下一步

其中“结果汇总”统一使用下列表头：

| exp_id | 关键改动 | accuracy | macro_f1 | macro_recall | minority_recall_mean | score | 结论 |
|--------|----------|----------|----------|--------------|----------------------|-------|------|

说明：

- `accuracy / macro_f1 / macro_recall / minority_recall_mean / score` 优先写最终读者关心的结论指标。
- 如果某类实验没有对应字段，例如持续学习实验没有 `macro_f1`，则填 `—`，并在下方补一张“持续学习补充指标”表。
- `output_dir`、`checkpoint`、`run_summary.json` 等路径可以保留为佐证，但不能替代正文结论。

## 写作原则

1. 先写为什么做，再写做了什么，最后写结果和决策。
2. 不允许只堆路径或文件名，必须让不看 `outputs/` 的读者也能看懂。
3. 只写与上一阶段相比“变了什么”，不要把没变的公共设置重复展开。
4. 指标优先写最终决策用到的指标，不把正文写成原始日志转储。
5. 若某些元信息缺失，例如 commit、命令或日期范围，明确写“未追踪”，不要猜测。
6. “下一步”只保留单一主方向，不展开为愿望清单。

## 更新原则

- 新实验先补 `phase_*.md`，确认形成新主线后再更新对应线程的 `00_overview.md`。
- 主线切换时更新 `00_project_timeline.md`。
- 若一个阶段内部实验很多，优先保留对最终决策真正有影响的对比，不强求把所有失败尝试逐条写满。
