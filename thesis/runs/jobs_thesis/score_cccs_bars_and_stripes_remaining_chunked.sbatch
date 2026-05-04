#!/usr/bin/env bash
#SBATCH --job-name=ccc_bas_score_chunk
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.err
#SBATCH --array=0

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
DATA_DIR="$ROOT/thesis/datasets_tests/bars_and_stripes"
OUT_BASE="$ROOT/thesis/my_results/bars_and_stripes"
LOG_DIR="$ROOT/thesis/runs/logs_thesis"
TRAIN_GLOB='bars_and_stripes_*noise_train.csv'
TARGET_TASK_IDS=(1 5 9 3 7 11)
NUM_SEEDS=5

mkdir -p "$LOG_DIR" "$OUT_BASE"
ulimit -n 4096 || true

export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

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

active_task_ids=()
for base_task_id in "${TARGET_TASK_IDS[@]}"; do
  model_index=$(( base_task_id / N_TRAINS ))
  data_index=$(( base_task_id % N_TRAINS ))

  model_label="${MODEL_LABELS[$model_index]}"
  train="${TRAINS[$data_index]}"
  dataset_name="$(basename "$train" .csv)"
  dataset_name="${dataset_name%_train}"
  results_dir="$OUT_BASE/$model_label/results"
  final_hps="$results_dir/${model_label}_${dataset_name}_GridSearchCV-best-hyperparams.csv"
  final_score="$results_dir/${model_label}_${dataset_name}_GridSearchCV-best-hyperparams-results.csv"

  if [[ -f "$final_hps" && ! -f "$final_score" ]]; then
    active_task_ids+=("$base_task_id")
  fi
done

logical_total=$(( ${#active_task_ids[@]} * NUM_SEEDS ))
if (( logical_total == 0 )); then
  echo "No remaining CCC chunked score tasks."
  exit 0
fi

if (( SLURM_ARRAY_TASK_ID >= logical_total )); then
  echo "Array index $SLURM_ARRAY_TASK_ID exceeds current logical total $logical_total. Exiting."
  exit 0
fi

task_pos=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
base_task_id="${active_task_ids[$task_pos]}"

model_index=$(( base_task_id / N_TRAINS ))
data_index=$(( base_task_id % N_TRAINS ))

MODEL_LABEL="${MODEL_LABELS[$model_index]}"
MODEL_CLASS="${MODEL_CLASSES[$model_index]}"
TRAIN="${TRAINS[$data_index]}"
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

RAW_HPS="$RESULTS_DIR/${RAW_STEM}-best-hyperparams.csv"
RAW_SCORE="$RESULTS_DIR/${RAW_STEM}-best-hyperparams-results.csv"

rename_halfsep_outputs() {
  if [[ "$MODEL_LABEL" != "CircuitCentricClassifierHalfSeparableRandom50" ]]; then
    return 0
  fi

  [[ -f "$RAW_HPS" ]] && mv -f "$RAW_HPS" "$FINAL_HPS"
  [[ -f "$RAW_SCORE" ]] && mv -f "$RAW_SCORE" "$FINAL_SCORE"
}

echo "MODEL_LABEL=$MODEL_LABEL"
echo "MODEL_CLASS=$MODEL_CLASS"
echo "TRAIN=$TRAIN"
echo "SEED=$seed/$NUM_SEEDS"
python -V

cd "$ROOT"
srun python scripts/score_with_best_hyperparameters_chunked.py \
  --classifier-name "$MODEL_CLASS" \
  --trainset-path "$TRAIN" \
  --testset-path "$TEST" \
  --hyperparams-path "$FINAL_HPS" \
  --results-path "$OUT_DIR" \
  --seed "$seed" \
  --num-seeds "$NUM_SEEDS"

rename_halfsep_outputs
echo "DONE $MODEL_LABEL $DATASET_NAME seed=$seed"
