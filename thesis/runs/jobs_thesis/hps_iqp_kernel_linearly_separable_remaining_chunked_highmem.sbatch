#!/usr/bin/env bash
#SBATCH --job-name=iqp_ls_hps_chunk_hm
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/runs/logs_thesis/%x-%A_%a.err
#SBATCH --array=0

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

MODELS=(
  "IQPKernelClassifier"
  "IQPKernelClassifierHalfSeparable"
  "IQPKernelClassifierSeparable"
)

mapfile -t TRAINS < <(find "$DATA_DIR" -maxdepth 1 -name "$TRAIN_GLOB" -type f | sort -V)
N_TRAINS=${#TRAINS[@]}

combo_count="$(python - <<'PY'
from sklearn.model_selection import ParameterGrid
from qml_benchmarks.hyperparam_search_utils import construct_hyperparameter_grid
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

grid = construct_hyperparameter_grid(hyper_parameter_settings, "IQPKernelClassifier")
print(len(list(ParameterGrid(grid))))
PY
)"

active_task_ids=()
for model_index in "${!MODELS[@]}"; do
  model="${MODELS[$model_index]}"
  for data_index in "${!TRAINS[@]}"; do
    train="${TRAINS[$data_index]}"
    dataset_name="$(basename "$train" .csv)"
    dataset_name="${dataset_name%_train}"
    dim="${dataset_name#linearly_separable_}"
    dim="${dim%d}"
    if (( dim < 14 )); then
      continue
    fi

    task_id=$(( model_index * N_TRAINS + data_index ))
    results_dir="$OUT_BASE/$model/results"
    stem="${model}_${dataset_name}_GridSearchCV"
    final_hps="$results_dir/${stem}-best-hyperparams.csv"
    final_score="$results_dir/${stem}-best-hyperparams-results.csv"

    if [[ ! -f "$final_score" && ! -f "$final_hps" ]]; then
      active_task_ids+=("$task_id")
    fi
  done
done

logical_total=$(( ${#active_task_ids[@]} * combo_count ))
if (( logical_total == 0 )); then
  echo "No remaining high-memory IQP chunked HPS tasks."
  exit 0
fi

if (( SLURM_ARRAY_TASK_ID >= logical_total )); then
  echo "Array index $SLURM_ARRAY_TASK_ID exceeds current logical total $logical_total. Exiting."
  exit 0
fi

task_pos=$(( SLURM_ARRAY_TASK_ID / combo_count ))
combo_index=$(( SLURM_ARRAY_TASK_ID % combo_count ))
task_id="${active_task_ids[$task_pos]}"

model_index=$(( task_id / N_TRAINS ))
data_index=$(( task_id % N_TRAINS ))

MODEL="${MODELS[$model_index]}"
TRAIN="${TRAINS[$data_index]}"
DATASET_NAME="$(basename "$TRAIN" .csv)"
DATASET_NAME="${DATASET_NAME%_train}"
OUT_DIR="$OUT_BASE/$MODEL"

echo "MODEL=$MODEL"
echo "TRAIN=$TRAIN"
echo "COMBO_INDEX=$combo_index/$combo_count"
python -V

cd "$ROOT"
srun python scripts/run_hyperparameter_search_chunked.py \
  --model "$MODEL" \
  --dataset-path "$TRAIN" \
  --results-path "$OUT_DIR" \
  --combo-index "$combo_index" \
  --n-jobs 1 \
  --pre-dispatch 1 \
  --hyperparameter-scoring accuracy roc_auc \
  --hyperparameter-refit accuracy

echo "DONE $MODEL $DATASET_NAME combo=$combo_index"
