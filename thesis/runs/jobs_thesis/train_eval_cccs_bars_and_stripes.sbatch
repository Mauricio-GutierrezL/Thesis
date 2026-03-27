#!/usr/bin/env bash
#SBATCH --job-name=ccc_bas_train_eval
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.err
#SBATCH --array=0,1,4,5,6,8,9%2

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
DATA_DIR="$ROOT/thesis/datasets_tests/bars_and_stripes"
OUT_BASE="$ROOT/thesis/my_results/bars_and_stripes"
LOG_DIR="$ROOT/thesis/runs/logs_thesis"
TRAIN_GLOB='bars_and_stripes_*noise_train.csv'

# The 32_x_32 bars-and-stripes tasks have repeatedly OOMed under the standard
# 48G allocation. They are submitted via the dedicated high-memory script.
HIGH_RISK_ARRAY_IDS=(3 7 11)

mkdir -p "$LOG_DIR" "$OUT_BASE"
ulimit -n 4096 || true

export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

python - <<'PY'
from importlib import import_module
m = import_module('qml_benchmarks.models')
for name in ['CircuitCentricClassifier', 'CircuitCentricClassifierHalfSeparable', 'CircuitCentricClassifierSeparable']:
    getattr(m, name)
print('BATCH_ENV_OK')
PY

MODEL_LABELS=(
  "CircuitCentricClassifier"
  "CircuitCentricClassifierHalfSeparableRandom50"
  "CircuitCentricClassifierSeparable"
)

MODEL_CLASSES=(
  "CircuitCentricClassifier"
  "CircuitCentricClassifierHalfSeparable"
  "CircuitCentricClassifierSeparable"
)

mapfile -t TRAINS < <(find "$DATA_DIR" -maxdepth 1 -name "$TRAIN_GLOB" -type f | sort -V)
N_TRAINS=${#TRAINS[@]}
TOTAL=$(( ${#MODEL_CLASSES[@]} * N_TRAINS ))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "ERROR: task id out of range"
  exit 1
fi

for high_risk_id in "${HIGH_RISK_ARRAY_IDS[@]}"; do
  if (( SLURM_ARRAY_TASK_ID == high_risk_id )); then
    echo "ERROR: task id $SLURM_ARRAY_TASK_ID is reserved for high-memory resubmission"
    echo "Use thesis/runs/jobs_thesis/train_eval_cccs_bars_and_stripes_32x32.sbatch instead."
    exit 5
  fi
done

MODEL_INDEX=$(( SLURM_ARRAY_TASK_ID / N_TRAINS ))
DATA_INDEX=$(( SLURM_ARRAY_TASK_ID % N_TRAINS ))

MODEL_LABEL="${MODEL_LABELS[$MODEL_INDEX]}"
MODEL_CLASS="${MODEL_CLASSES[$MODEL_INDEX]}"
TRAIN="${TRAINS[$DATA_INDEX]}"
TEST="${TRAIN/_train.csv/_test.csv}"
DATASET_NAME="$(basename "$TRAIN" .csv)"
DATASET_NAME="${DATASET_NAME%_train}"

OUT_DIR="$OUT_BASE/$MODEL_LABEL"
RESULTS_DIR="$OUT_DIR/results"
mkdir -p "$RESULTS_DIR"

FINAL_STEM="${MODEL_LABEL}_${DATASET_NAME}_GridSearchCV"
RAW_STEM="${MODEL_CLASS}_${DATASET_NAME}_GridSearchCV"

FINAL_HPS="$RESULTS_DIR/${FINAL_STEM}-best-hyperparams.csv"
FINAL_SCORE="$RESULTS_DIR/${FINAL_STEM}-best-hyperparams-results.csv"
FINAL_CV="$RESULTS_DIR/${FINAL_STEM}.csv"

RAW_HPS="$RESULTS_DIR/${RAW_STEM}-best-hyperparams.csv"
RAW_SCORE="$RESULTS_DIR/${RAW_STEM}-best-hyperparams-results.csv"
RAW_CV="$RESULTS_DIR/${RAW_STEM}.csv"

rename_halfsep_outputs() {
  if [[ "$MODEL_LABEL" != "CircuitCentricClassifierHalfSeparableRandom50" ]]; then
    return 0
  fi

  [[ -f "$RAW_CV" ]] && mv -f "$RAW_CV" "$FINAL_CV"
  [[ -f "$RAW_HPS" ]] && mv -f "$RAW_HPS" "$FINAL_HPS"
  [[ -f "$RAW_SCORE" ]] && mv -f "$RAW_SCORE" "$FINAL_SCORE"
}

echo "MODEL_LABEL=$MODEL_LABEL"
echo "MODEL_CLASS=$MODEL_CLASS"
echo "TRAIN=$TRAIN"
echo "TEST=$TEST"
echo "OUT_DIR=$OUT_DIR"
python -V

HPS_N_JOBS=2
SCORE_N_JOBS=2

if [[ "$MODEL_LABEL" != "CircuitCentricClassifierHalfSeparableRandom50" ]]; then
  HPS_N_JOBS=4
  SCORE_N_JOBS=4
fi

if [[ "$DATASET_NAME" == *"32_x_32"* ]]; then
  HPS_N_JOBS=1
  SCORE_N_JOBS=1
elif [[ "$DATASET_NAME" == *"16_x_16"* ]]; then
  if (( HPS_N_JOBS > 2 )); then
    HPS_N_JOBS=2
  fi
  if (( SCORE_N_JOBS > 2 )); then
    SCORE_N_JOBS=2
  fi
fi

echo "HPS_N_JOBS=$HPS_N_JOBS"
echo "SCORE_N_JOBS=$SCORE_N_JOBS"

[[ -f "$TRAIN" ]] || { echo "Missing train dataset: $TRAIN"; exit 2; }
[[ -f "$TEST" ]] || { echo "Missing test dataset: $TEST"; exit 2; }

cd "$ROOT"

if [[ ! -f "$FINAL_HPS" ]]; then
  srun python -u scripts/run_hyperparameter_search.py \
    --model "$MODEL_CLASS" \
    --dataset-path "$TRAIN" \
    --results-path "$OUT_DIR" \
    --n-jobs "$HPS_N_JOBS" \
    --clean False \
    --hyperparameter-scoring accuracy roc_auc \
    --hyperparameter-refit accuracy

  rename_halfsep_outputs
fi

[[ -f "$FINAL_HPS" ]] || { echo "Missing best hyperparams after HPS: $FINAL_HPS"; exit 3; }

if [[ ! -f "$FINAL_SCORE" ]]; then
  srun python scripts/score_with_best_hyperparameters.py \
    --classifier-name "$MODEL_CLASS" \
    --trainset-path "$TRAIN" \
    --testset-path "$TEST" \
    --hyperparams-path "$FINAL_HPS" \
    --results-path "$OUT_DIR" \
    --clean False \
    --n-jobs "$SCORE_N_JOBS"

  rename_halfsep_outputs
fi

[[ -f "$FINAL_SCORE" ]] || { echo "Missing final scored results: $FINAL_SCORE"; exit 4; }

echo "DONE $MODEL_LABEL $DATASET_NAME"
