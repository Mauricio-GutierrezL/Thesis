#!/usr/bin/env bash
#SBATCH --job-name=iqp_ls_train_eval
#SBATCH --account=thes2204
#SBATCH --partition=c23ms
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.err
#SBATCH --array=0-56

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
DATA_DIR="$ROOT/thesis/datasets_tests/linearly_separable"
OUT_BASE="$ROOT/thesis/my_results/linearly_separable"
LOG_DIR="$ROOT/thesis/runs/logs_thesis"
TRAIN_GLOB='linearly_separable_*d_train.csv'

mkdir -p "$LOG_DIR" "$OUT_BASE"
ulimit -n 4096 || true

export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

python - <<'INNERPY'
from importlib import import_module
m = import_module('qml_benchmarks.models')
for name in ['IQPKernelClassifier', 'IQPKernelClassifierHalfSeparable', 'IQPKernelClassifierSeparable']:
    getattr(m, name)
print('BATCH_ENV_OK')
INNERPY

MODELS=(
  "IQPKernelClassifier"
  "IQPKernelClassifierHalfSeparable"
  "IQPKernelClassifierSeparable"
)

mapfile -t TRAINS < <(find "$DATA_DIR" -maxdepth 1 -name "$TRAIN_GLOB" -type f | sort -V)
N_TRAINS=${#TRAINS[@]}
TOTAL=$(( ${#MODELS[@]} * N_TRAINS ))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "ERROR: task id out of range"
  exit 1
fi

MODEL_INDEX=$(( SLURM_ARRAY_TASK_ID / N_TRAINS ))
DATA_INDEX=$(( SLURM_ARRAY_TASK_ID % N_TRAINS ))

MODEL="${MODELS[$MODEL_INDEX]}"
TRAIN="${TRAINS[$DATA_INDEX]}"
TEST="${TRAIN/_train.csv/_test.csv}"
DATASET_NAME="$(basename "$TRAIN" .csv)"
DATASET_NAME="${DATASET_NAME%_train}"

OUT_DIR="$OUT_BASE/$MODEL"
RESULTS_DIR="$OUT_DIR/results"
mkdir -p "$RESULTS_DIR"

FINAL_STEM="${MODEL}_${DATASET_NAME}_GridSearchCV"
FINAL_HPS="$RESULTS_DIR/${FINAL_STEM}-best-hyperparams.csv"
FINAL_SCORE="$RESULTS_DIR/${FINAL_STEM}-best-hyperparams-results.csv"

echo "MODEL=$MODEL"
echo "TRAIN=$TRAIN"
echo "TEST=$TEST"
echo "OUT_DIR=$OUT_DIR"
python -V

[[ -f "$TRAIN" ]] || { echo "Missing train dataset: $TRAIN"; exit 2; }
[[ -f "$TEST" ]] || { echo "Missing test dataset: $TEST"; exit 2; }

cd "$ROOT"

if [[ ! -f "$FINAL_HPS" ]]; then
  srun python -u scripts/run_hyperparameter_search.py     --model "$MODEL"     --dataset-path "$TRAIN"     --results-path "$OUT_DIR"     --n-jobs "$SLURM_CPUS_PER_TASK"     --clean False     --hyperparameter-scoring accuracy roc_auc     --hyperparameter-refit accuracy
fi

[[ -f "$FINAL_HPS" ]] || { echo "Missing best hyperparams after HPS: $FINAL_HPS"; exit 3; }

if [[ ! -f "$FINAL_SCORE" ]]; then
  srun python scripts/score_with_best_hyperparameters.py     --classifier-name "$MODEL"     --trainset-path "$TRAIN"     --testset-path "$TEST"     --hyperparams-path "$FINAL_HPS"     --results-path "$OUT_DIR"     --clean False     --n-jobs "$SLURM_CPUS_PER_TASK"
fi

[[ -f "$FINAL_SCORE" ]] || { echo "Missing final scored results: $FINAL_SCORE"; exit 4; }

echo "DONE $MODEL $DATASET_NAME"
