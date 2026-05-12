#!/usr/bin/env bash
#SBATCH --job-name=iqp_ls_score_missing
#SBATCH --account=thes2204
#SBATCH --partition=c23ms
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs/%x-%A_%a.err
#SBATCH --array=0

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
DATA_DIR="$ROOT/thesis/datasets_tests/linearly_separable"
OUT_BASE="$ROOT/thesis/my_results/linearly_separable"
LOG_DIR="$ROOT/thesis/runs/logs"
TRAIN_GLOB='linearly_separable_*d_train.csv'

mkdir -p "$LOG_DIR" "$OUT_BASE"
export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

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

[[ -f "$FINAL_HPS" ]] || { echo "Missing best hyperparams: $FINAL_HPS"; exit 2; }

if [[ ! -f "$FINAL_SCORE" ]]; then
  srun python scripts/score_with_best_hyperparameters.py \
    --classifier-name "$MODEL" \
    --trainset-path "$TRAIN" \
    --testset-path "$TEST" \
    --hyperparams-path "$FINAL_HPS" \
    --results-path "$OUT_DIR" \
    --clean False \
    --n-jobs 1
fi

[[ -f "$FINAL_SCORE" ]] || { echo "Missing final scored results: $FINAL_SCORE"; exit 3; }

echo "DONE $MODEL $DATASET_NAME"
