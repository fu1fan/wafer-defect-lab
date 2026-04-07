#!/usr/bin/env bash
# Run all 8 search experiments for Nested CMS ResNet
# Each experiment: 3 epochs/task, fixed seed via CUBLAS/PYTHONHASHSEED
set -euo pipefail

PYTHON="/home/fu1fan/miniconda3/envs/torch/bin/python"
BASE="configs/modal/research_nest/experiments"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

for i in 1 2 3 4 5 6 7 8; do
    CFG="${BASE}/wm811k_cms_resnet_s${i}.yaml"
    OUT="outputs/cms_resnet_search_s${i}"
    echo ""
    echo "========================================================"
    echo " Running Search S${i}: $(head -1 "$CFG")"
    echo "========================================================"
    $PYTHON scripts/train_continual.py \
        --config "$CFG" \
        --output-dir "$OUT" 2>&1 | tee "outputs/cms_resnet_search_s${i}.log"
    echo "S${i} complete. Results: ${OUT}/continual_results.json"
done

echo ""
echo "All search experiments complete!"
