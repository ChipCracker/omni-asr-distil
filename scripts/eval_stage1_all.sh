#!/bin/bash
# Submit all Stage 1 evaluation jobs (3 architectures × 2 datasets).
#
# Usage: bash scripts/eval_stage1_all.sh

DATA_BASE="/nfs1/scratch/students/witzlch88229/data/omni-asr-ft"
ORT_DATASET="${DATA_BASE}/rvg1_de/version=0"
TR2_DATASET="${DATA_BASE}/rvg1_de_tr2/version=0"

ARCHS=("distill_s_small" "distill_s_medium" "distill_s_large")

for arch in "${ARCHS[@]}"; do
    echo "Submitting ${arch} × ORT..."
    sbatch slurm/eval_rvg1.sh "${arch}" test "${ORT_DATASET}"

    echo "Submitting ${arch} × TR2..."
    sbatch slurm/eval_rvg1.sh "${arch}" test "${TR2_DATASET}"
done

echo "Submitted ${#ARCHS[@]} × 2 = $(( ${#ARCHS[@]} * 2 )) eval jobs."
