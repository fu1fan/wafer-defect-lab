#!/usr/bin/env bash
# Phase E screening: run all backbone experiments sequentially
# Usage: bash scripts/run_phase_e_screening.sh
set -euo pipefail

cd "$(dirname "$0")/.."
CONF_DIR="configs/modal/baseline/experiments/phase_e_backbone"
OUT_DIR="outputs/phase_e"

declare -A EXPERIMENTS=(
  [E1_convnext_tiny_focal]="$CONF_DIR/E1_convnext_tiny_focal.yaml"
  [E2_convnext_tiny_balanced_softmax]="$CONF_DIR/E2_convnext_tiny_balanced_softmax.yaml"
  [E3_efficientnetv2_s_focal]="$CONF_DIR/E3_efficientnetv2_s_focal.yaml"
  [E4_efficientnetv2_s_balanced_softmax]="$CONF_DIR/E4_efficientnetv2_s_balanced_softmax.yaml"
  [E5_convnext_tiny_logit_adj]="$CONF_DIR/E5_convnext_tiny_logit_adj.yaml"
  [E6_efficientnetv2_s_logit_adj]="$CONF_DIR/E6_efficientnetv2_s_logit_adj.yaml"
  [E7_convnext_small_focal]="$CONF_DIR/E7_convnext_small_focal.yaml"
  [E8_convnext_tiny_ldam]="$CONF_DIR/E8_convnext_tiny_ldam.yaml"
)

ORDER=(E1_convnext_tiny_focal E3_efficientnetv2_s_focal E2_convnext_tiny_balanced_softmax E4_efficientnetv2_s_balanced_softmax E5_convnext_tiny_logit_adj E6_efficientnetv2_s_logit_adj E7_convnext_small_focal E8_convnext_tiny_ldam)

for exp in "${ORDER[@]}"; do
  config="${EXPERIMENTS[$exp]}"
  out="$OUT_DIR/$exp"
  echo "=========================================="
  echo " Running: $exp"
  echo "=========================================="
  python scripts/train_classifier.py \
    --config "$config" \
    --task-mode multiclass \
    --output-dir "$out"
  echo ""
  echo " Evaluating: $exp"
  python scripts/eval_classifier.py \
    --run-summary "$out/run_summary.json" \
    --split Test \
    --output-dir "$out"
  echo ""
done

echo "All Phase E experiments complete!"
