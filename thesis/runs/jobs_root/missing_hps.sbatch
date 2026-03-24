#!/bin/bash
#SBATCH --job-name=missing_hps
#SBATCH --partition=c23ms
#SBATCH --array=0-10%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

ulimit -n 4096 || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd ~/qml_benchmarks/qml-benchmarks-main
source ~/venvs/qmlb/bin/activate
export PYTHONNOUSERSITE=1

TASKS=(
  "DataReuploadingClassifier 19"
  "DataReuploadingClassifier 20"
  "DressedQuantumCircuitClassifier 12"
  "DressedQuantumCircuitClassifier 13"
  "DressedQuantumCircuitClassifier 14"
  "DressedQuantumCircuitClassifier 15"
  "DressedQuantumCircuitClassifier 16"
  "DressedQuantumCircuitClassifier 17"
  "DressedQuantumCircuitClassifier 18"
  "DressedQuantumCircuitClassifier 19"
  "DressedQuantumCircuitClassifier 20"
)

read -r MODEL D <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

DATA="thesis/datasets_tests/linearly_separable/linearly_separable_${D}d_train.csv"
OUT="thesis/my_results/${MODEL}/d${D}"

echo "MODEL=$MODEL  D=$D"
echo "DATA=$DATA"
echo "OUT=$OUT"
python -V

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: missing dataset: $DATA"
  exit 2
fi

if [[ -d "$OUT/results" ]] && [[ -n "$(ls -A "$OUT/results" 2>/dev/null)" ]]; then
  echo "SKIP: $OUT/results already exists and is non-empty"
  exit 0
fi

mkdir -p "$OUT"

python -u scripts/run_hyperparameter_search.py \
  --model "$MODEL" \
  --dataset-path "$DATA" \
  --results-path "$OUT" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --clean False \
  --hyperparameter-scoring accuracy roc_auc \
  --hyperparameter-refit accuracy
