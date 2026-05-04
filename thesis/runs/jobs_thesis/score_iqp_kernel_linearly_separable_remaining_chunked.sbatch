#!/usr/bin/env bash
#SBATCH --job-name=iqp_ls_score_chunk
#SBATCH --account=thes2204
#SBATCH --partition=c23ms
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.err
#SBATCH --array=0

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
DATA_DIR="$ROOT/thesis/datasets_tests/linearly_separable"
OUT_BASE="$ROOT/thesis/my_results/linearly_separable"
LOG_DIR="$ROOT/thesis/runs/logs_thesis"
TRAIN_GLOB='linearly_separable_*d_train.csv'
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

MODELS=(
  "IQPKernelClassifier"
  "IQPKernelClassifierHalfSeparable"
  "IQPKernelClassifierSeparable"
)

mapfile -t TRAINS < <(find "$DATA_DIR" -maxdepth 1 -name "$TRAIN_GLOB" -type f | sort -V)
N_TRAINS=${#TRAINS[@]}

active_task_ids=()
for model_index in "${!MODELS[@]}"; do
  model="${MODELS[$model_index]}"
  for data_index in "${!TRAINS[@]}"; do
    train="${TRAINS[$data_index]}"
    dataset_name="$(basename "$train" .csv)"
    dataset_name="${dataset_name%_train}"
    task_id=$(( model_index * N_TRAINS + data_index ))
    results_dir="$OUT_BASE/$model/results"
    stem="${model}_${dataset_name}_GridSearchCV"
    final_hps="$results_dir/${stem}-best-hyperparams.csv"
    final_score="$results_dir/${stem}-best-hyperparams-results.csv"

    if [[ -f "$final_hps" && ! -f "$final_score" ]]; then
      active_task_ids+=("$task_id")
    fi
  done
done

logical_total=$(( ${#active_task_ids[@]} * NUM_SEEDS ))
if (( logical_total == 0 )); then
  echo "No remaining IQP chunked score tasks."
  exit 0
fi

if (( SLURM_ARRAY_TASK_ID >= logical_total )); then
  echo "Array index $SLURM_ARRAY_TASK_ID exceeds current logical total $logical_total. Exiting."
  exit 0
fi

task_pos=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
task_id="${active_task_ids[$task_pos]}"

model_index=$(( task_id / N_TRAINS ))
data_index=$(( task_id % N_TRAINS ))

MODEL="${MODELS[$model_index]}"
TRAIN="${TRAINS[$data_index]}"
TEST="${TRAIN/_train.csv/_test.csv}"
DATASET_NAME="$(basename "$TRAIN" .csv)"
DATASET_NAME="${DATASET_NAME%_train}"
OUT_DIR="$OUT_BASE/$MODEL"
RESULTS_DIR="$OUT_DIR/results"
FINAL_HPS="$RESULTS_DIR/${MODEL}_${DATASET_NAME}_GridSearchCV-best-hyperparams.csv"

echo "MODEL=$MODEL"
echo "TRAIN=$TRAIN"
echo "SEED=$seed/$NUM_SEEDS"
python -V

cd "$ROOT"
srun python scripts/score_with_best_hyperparameters_chunked.py \
  --classifier-name "$MODEL" \
  --trainset-path "$TRAIN" \
  --testset-path "$TEST" \
  --hyperparams-path "$FINAL_HPS" \
  --results-path "$OUT_DIR" \
  --seed "$seed" \
  --num-seeds "$NUM_SEEDS"

echo "DONE $MODEL $DATASET_NAME seed=$seed"
