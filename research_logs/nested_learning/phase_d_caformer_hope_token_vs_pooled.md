# Phase D: CAFormer HOPE Token vs Pooled Continual

## Phase 0 audit

- External reference: https://github.com/kmccleary3301/nested_learning
- Local clone: `/tmp/nested_learning_ref`
- Reference commit: `56463f3853552f17791bc10a40c534864724d614`
- Required source locations:
  - `src/nested_learning/hope/block.py:719` `HOPESelfModBlockConfig`
  - `src/nested_learning/hope/block.py:745` `HOPESelfModBlock`
  - `src/nested_learning/hope/block.py:115` `HOPEBlockConfig`
  - `src/nested_learning/hope/block.py:1298` `HOPEBlock`
  - `src/nested_learning/cms.py` `CMS` / `CMSBlock`
  - `src/nested_learning/titan/self_modifying.py` `SelfModifyingTitans`
- Required docs read:
  - `docs/PAPER_COMPLIANCE.md`
  - `docs/STREAMING_CONTRACT.md`
  - `docs/future_directions.md`
  - `docs/stability_journal.md` was requested but is not present in this reference commit.
- Mechanism tests inspected:
  - `tests/test_cms.py`
  - `tests/test_selfmod_online.py`
  - `tests/test_cms_delta_rule.py`
  - `tests/test_surprise_metric.py`
  - `tests/test_teach_signal.py`
  - `tests/test_cms_cross_call.py` via symbol search

## Migration plan

Migrated now:

- HOPE-SelfMod order: self-modifying read path followed by CMS.
- CMS fast/slow memory levels with independent `update_period`.
- Chunk-accumulated CMS online writes using `update_period` as chunk size.
- Delta-rule target-shift loss with stop-gradient targets.
- Surprise gating on online CMS writes.
- Explicit stop-gradient online writes after the outer optimizer step.
- Update telemetry via `pop_update_stats()`.

Deferred:

- Full `HOPEBlock` attention + TitanMemory path. It is heavier and mixes an attention memory path with CMS; first comparison should isolate tokenization as the main variable.
- Fast-state / per-context delta state. The wafer continual trainer uses persistent model parameters and batch-level replay, not LLM streaming fast-state contexts.
- Attention-cache carry, boundary-target streaming, tokenizer and LM pipeline. These are sequence-model specific and not needed for fixed wafer images.
- Paper optimizer variants such as M3. Current comparison keeps the project trainer/AdamW path unchanged.

## Implemented structures

Token model `caformer_hope_token`:

```text
[B,1,224,224] -> CAFormer-S18 stem/stages through stage 3
              -> [B,512,7,7] -> 49 tokens -> Linear(512, D)
              -> shared HOPESelfModBlock stack -> mean pool -> classifier
```

Pooled model `caformer_hope_pooled`:

```text
[B,1,224,224] -> CAFormer-S18 pooled feature [B,512]
              -> [B,1,512] -> Linear(512, D)
              -> same HOPESelfModBlock stack -> classifier
```

Both variants use the same HOPE/CMS configuration unless explicitly changed by config.

## Commands

Tensor smoke:

```bash
source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch
python - <<'PY'
...
PY
```

Continual smoke:

```bash
python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_token_smoke.yaml \
  --output-dir outputs/caformer_hope/token_smoke \
  --smoke-test

python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_pooled_smoke.yaml \
  --output-dir outputs/caformer_hope/pooled_smoke \
  --smoke-test
```

S1 no-KD comparison:

```bash
python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_token_continual_s1.yaml \
  --output-dir outputs/caformer_hope/token_s1

python scripts/train_continual.py \
  --config configs/modal/research_nest/experiments/caformer_hope_pooled_continual_s1.yaml \
  --output-dir outputs/caformer_hope/pooled_s1
```

## Results

### Tensor smoke

Passed.

| Model | `forward_tokens()` | logits | teach logits | backward |
|---|---:|---:|---:|---|
| `caformer_hope_token` | `[2, 49, 32]` | `[2, 9]` | `[2, 9]` | pass |
| `caformer_hope_pooled` | `[2, 1, 32]` | `[2, 9]` | `[2, 9]` | pass |

Command:

```bash
source /home/fu1fan/miniconda3/etc/profile.d/conda.sh && conda activate torch
PYTHONPATH=src pytest -q tests/test_caformer_hope.py
```

Result: `2 passed in 2.56s`.

### Continual smoke

Both `train_continual.py --smoke-test` runs completed and wrote `continual_results.json`.

| Model | Output dir | avg_accuracy_final | avg_forgetting | full macro acc | Note |
|---|---|---:|---:|---:|---|
| token | `outputs/caformer_hope/token_smoke` | 0.3333 | 0.0000 | 0.3333 | smoke subset collapses to final `none` task |
| pooled | `outputs/caformer_hope/pooled_smoke` | 0.3333 | 0.0000 | 0.3333 | smoke subset collapses to final `none` task |

The smoke subset is intentionally tiny and class-sparse; these numbers are executable-path checks only, not evidence about model quality.

### S1 no-KD middle run

Completed.

| Model | avg_accuracy_final | avg_forgetting | full overall | full macro | none | Loc | Edge-Loc | Scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `caformer_hope_token` | 0.8057 | **0.0793** | 0.9192 | 0.8001 | 0.9312 | **0.6731** | **0.8330** | **0.8153** |
| `caformer_hope_pooled` | **0.8075** | 0.0939 | **0.9356** | **0.8212** | **0.9489** | 0.5864 | 0.7828 | 0.7807 |

Task accuracy matrix:

```text
token:
[
  [0.9608, 0.0000, 0.0000],
  [0.9448, 0.6534, 0.0000],
  [0.8021, 0.6850, 0.9298],
]

pooled:
[
  [0.9741, 0.0000, 0.0000],
  [0.9048, 0.6885, 0.0000],
  [0.7907, 0.6841, 0.9476],
]
```

Full per-class accuracy:

| Class | token | pooled | Higher |
|---|---:|---:|---|
| none | 0.9312 | 0.9489 | pooled |
| Center | 0.6707 | 0.8125 | pooled |
| Donut | 0.9658 | 0.8082 | token |
| Edge-Loc | 0.8330 | 0.7828 | token |
| Edge-Ring | 0.6821 | 0.8313 | pooled |
| Loc | 0.6731 | 0.5864 | token |
| Near-full | 0.9684 | 0.9684 | tie |
| Random | 0.6615 | 0.8716 | pooled |
| Scratch | 0.8153 | 0.7807 | token |

## Interpretation

- Both CAFormer HOPE variants substantially exceed the old Nested continual references on `avg_accuracy_final`:
  - `nested_cms_resnet_r2`: 0.6037
  - `v4_ref` no-KD: about 0.443
  - new token: 0.8057
  - new pooled: 0.8075
- Pooled is slightly better on final average accuracy (+0.0018), full-test overall (+0.0164), full macro (+0.0211), `none`, Center, Edge-Ring, and Random.
- Token has lower forgetting (-0.0146) and is better for Donut, Edge-Loc, Loc, and Scratch.
- The result does not support the old conclusion that tokenization is intrinsically bad. With CAFormer final-stage semantic tokens, tokenized HOPE is competitive and preserves old tasks slightly better.
- For this first no-KD S1 run, pooled is the more stable general-purpose choice, while tokenization appears useful for several spatial defect classes and forgetting control.

## Next protocol candidate

The old task order still places `none` in the final task and creates a large distribution shock. The next run should test a mixed-background order:

```yaml
task_order:
  - [0, 1, 3]
  - [4, 5, 8]
  - [2, 6, 7]
```

Rationale: include `none` from the start as background, keep Edge-Loc/Loc/Scratch distributed across tasks, and avoid forcing the final task to learn almost all background mass at once.
