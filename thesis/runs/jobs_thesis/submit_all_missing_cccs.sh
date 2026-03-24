#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
cd "$ROOT"

jobs=(
  "thesis/runs/jobs_thesis/train_eval_cccs_bars_and_stripes.sbatch"
  "thesis/runs/jobs_thesis/train_eval_cccs_hidden_manifold.sbatch"
  "thesis/runs/jobs_thesis/train_eval_cccs_hyperplanes_diff.sbatch"
  "thesis/runs/jobs_thesis/train_eval_cccs_mnist_pca.sbatch"
  "thesis/runs/jobs_thesis/train_eval_cccs_mnist_pca_small.sbatch"
  "thesis/runs/jobs_thesis/train_eval_cccs_two_curves_diff.sbatch"
)

for job in "${jobs[@]}"; do
  echo "Submitting $job"
  sbatch "$job"
done
