#!/usr/bin/env bash
#SBATCH --job-name=ccc_eval
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
source /home/fwm91820/venvs/qmlb/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

models=(
  "CircuitCentricClassifier"
  "CircuitCentricClassifierHalfSeparable"
  "CircuitCentricClassifierSeparable"
)

mapfile -t trains < <(ls -1 "$DATA_DIR"/hidden_manifold-10d-*manifold_train.csv | sort -V)
n_trains=${#trains[@]}
total=$(( ${#models[@]} * n_trains ))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$total" ]; then
  echo "ERROR: task id out of range"
  exit 1
fi

m_idx=$(( SLURM_ARRAY_TASK_ID / n_trains ))
d_idx=$(( SLURM_ARRAY_TASK_ID % n_trains ))

model="${models[$m_idx]}"
train="${trains[$d_idx]}"
test="${train/_train.csv/_test.csv}"

resbase="$OUT_BASE/$model"
mkdir -p "$resbase"

dataset_name="$(basename "$train" .csv)"
dataset_name="${dataset_name%_train}"
stem="${model}_${dataset_name}_GridSearchCV"
best="$resbase/results/${stem}-best-hyperparams.csv"
test -f "$best" || { echo "Missing best hyperparams file: $best"; exit 2; }

cd "$ROOT"
srun python scripts/score_with_best_hyperparameters.py \
  --classifier-name "$model" \
  --trainset-path "$train" \
  --testset-path "$test" \
  --hyperparams-path "$best" \
  --results-path "$resbase" \
  --clean True \
  --n-jobs "$SLURM_CPUS_PER_TASK"
