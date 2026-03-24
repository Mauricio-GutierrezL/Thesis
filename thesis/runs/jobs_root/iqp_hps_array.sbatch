#!/bin/bash
#SBATCH --job-name=iqp_hps
#SBATCH --partition=c23ms
#SBATCH --array=2-20%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

D="${SLURM_ARRAY_TASK_ID}"
DATA="thesis/datasets_tests/linearly_separable/linearly_separable_${D}d_train.csv"
OUT="thesis/my_results/iqp/linearly_separable_${D}d"

echo "===== IQP HPS: d=${D} ====="
echo "DATA=${DATA}"
echo "OUT=${OUT}"
python -V
python -c "import autoray, pennylane as qml, pandas as pd; print('autoray', autoray.__version__, 'pennylane', qml.__version__, 'pandas', pd.__version__)"

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: missing dataset: $DATA"
  exit 2
fi

python -u scripts/run_hyperparameter_search.py \
  --model IQPKernelClassifier \
  --dataset-path "$DATA" \
  --results-path "$OUT" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --clean True \
  --hyperparameter-scoring accuracy \
  --hyperparameter-refit accuracy
