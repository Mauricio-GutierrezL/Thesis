#!/usr/bin/env bash
#SBATCH --job-name=ccc_hmd
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%A_%a.err
#SBATCH --array=0-56%8

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
DATA_DIR="$ROOT/thesis/datasets_tests/hidden_manifold_diff"
OUT_BASE="$ROOT/thesis/my_results/hidden_manifold_diff"

mkdir -p "$ROOT/thesis/logs"

export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

source /home/fwm91820/venvs/qmlb/bin/activate

models=(
  "CircuitCentricClassifier"
  "CircuitCentricClassifierHalfSeparable"
  "CircuitCentricClassifierSeparable"
)

mapfile -t trains < <(ls -1 "$DATA_DIR"/hidden_manifold-10d-*manifold_train.csv | sort -V)

n_trains=${#trains[@]}
n_models=${#models[@]}
total=$(( n_trains * n_models ))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$total" ]; then
  echo "ERROR: task id out of range"
  exit 1
fi

m_idx=$(( SLURM_ARRAY_TASK_ID / n_trains ))
d_idx=$(( SLURM_ARRAY_TASK_ID % n_trains ))

model="${models[$m_idx]}"
train="${trains[$d_idx]}"

outdir="$OUT_BASE/$model"
mkdir -p "$outdir"

cd "$ROOT"

srun python scripts/run_hyperparameter_search.py \
  --model "$model" \
  --dataset-path "$train" \
  --results-path "$outdir" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --clean True \
  --hyperparameter-scoring accuracy roc_auc \
  --hyperparameter-refit accuracy
