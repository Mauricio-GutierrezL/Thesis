#!/usr/bin/env bash
#SBATCH --job-name=halfsep_train
#SBATCH --account=thes2204
#SBATCH --partition=c23ms
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2500M
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%A_%a.err
#SBATCH --array=0-37%12

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
mkdir -p "$ROOT/thesis/logs"

export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

# avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

MODEL="CircuitCentricClassifierHalfSeparable"

LS_DIR="$ROOT/thesis/datasets_tests/linearly_separable"
HMD_DIR="$ROOT/thesis/datasets_tests/hidden_manifold_diff"

mapfile -t ls_trains  < <(ls -1 "$LS_DIR"/linearly_separable_*d_train.csv | sort -V)
mapfile -t hmd_trains < <(ls -1 "$HMD_DIR"/hidden_manifold-10d-*manifold_train.csv | sort -V)

n_ls=${#ls_trains[@]}
n_hmd=${#hmd_trains[@]}
total=$((n_ls + n_hmd))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$total" ]; then
  echo "ERROR: task id out of range"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt "$n_ls" ]; then
  TRAIN="${ls_trains[$SLURM_ARRAY_TASK_ID]}"
  TEST="${TRAIN/_train.csv/_test.csv}"
  BASEOUT="$ROOT/thesis/my_results/linearly_separable/$MODEL"
else
  idx=$((SLURM_ARRAY_TASK_ID - n_ls))
  TRAIN="${hmd_trains[$idx]}"
  TEST="${TRAIN/_train.csv/_test.csv}"
  BASEOUT="$ROOT/thesis/my_results/hidden_manifold_diff/$MODEL"
fi

DATASTEM="$(basename "$TRAIN" .csv)"   # e.g., linearly_separable_20d_train
RUNOUT="$BASEOUT/$DATASTEM"            # unique per task to avoid race
mkdir -p "$RUNOUT"

cd "$ROOT"

# 1) training / gridsearch
srun python scripts/run_hyperparameter_search.py \
  --model "$MODEL" \
  --dataset-path "$TRAIN" \
  --results-path "$RUNOUT" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --clean True \
  --hyperparameter-scoring accuracy roc_auc \
  --hyperparameter-refit accuracy

# 2) evaluation
STEM="${MODEL}_${DATASTEM}_GridSearchCV"
BEST="$RUNOUT/results/${STEM}-best-hyperparameters.csv"

srun python scripts/score_with_best_hyperparameters.py \
  --classifier-name "$MODEL" \
  --trainset-path "$TRAIN" \
  --testset-path "$TEST" \
  --hyperparams-path "$BEST" \
  --results-path "$RUNOUT" \
  --clean True \
  --n-jobs "$SLURM_CPUS_PER_TASK"

# 3) move CSVs into BASEOUT (so model folder holds the outputs directly)
#    add DATASTEM prefix so files from different datasets never overwrite each other
shopt -s nullglob
for f in "$RUNOUT/results/"*.csv; do
  base="$(basename "$f")"
  base="${base/_train_GridSearchCV/_GridSearchCV}"  # drop "_train" in middle
  mv -f "$f" "$BASEOUT/${DATASTEM}_${base}"
done

# cleanup per-run folder
rm -rf "$RUNOUT"

echo "DONE: $TRAIN"
