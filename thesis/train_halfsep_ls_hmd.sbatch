#!/usr/bin/env bash
#SBATCH --job-name=halfsep_train
#SBATCH --account=thes2204
#SBATCH --partition=c23ms
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2500M
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%A_%a.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%A_%a.err
#SBATCH --array=0-37%8

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
LOGDIR="$ROOT/thesis/logs"
mkdir -p "$LOGDIR"

export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

MODEL="CircuitCentricClassifierHalfSeparable"

LS_DIR="$ROOT/thesis/datasets_tests/linearly_separable"
HMD_DIR="$ROOT/thesis/datasets_tests/hidden_manifold_diff"

# collect train files
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
  OUTDIR="$ROOT/thesis/my_results/linearly_separable/$MODEL"
else
  idx=$((SLURM_ARRAY_TASK_ID - n_ls))
  TRAIN="${hmd_trains[$idx]}"
  TEST="${TRAIN/_train.csv/_test.csv}"
  OUTDIR="$ROOT/thesis/my_results/hidden_manifold_diff/$MODEL"
fi

mkdir -p "$OUTDIR"
cd "$ROOT"

# 1) training / gridsearch
srun python scripts/run_hyperparameter_search.py \
  --model "$MODEL" \
  --dataset-path "$TRAIN" \
  --results-path "$OUTDIR" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --clean True \
  --hyperparameter-scoring accuracy roc_auc \
  --hyperparameter-refit accuracy

# 2) evaluation
STEM="${MODEL}_$(basename "$TRAIN" .csv)_GridSearchCV"
BEST="$OUTDIR/results/${STEM}-best-hyperparameters.csv"

srun python scripts/score_with_best_hyperparameters.py \
  --classifier-name "$MODEL" \
  --trainset-path "$TRAIN" \
  --testset-path "$TEST" \
  --hyperparams-path "$BEST" \
  --results-path "$OUTDIR" \
  --clean True \
  --n-jobs "$SLURM_CPUS_PER_TASK"

# 3) flatten results/ and drop "_train" in filenames
shopt -s nullglob
for f in "$OUTDIR/results/"*.csv; do
  base="$(basename "$f")"
  new="${base/_train_GridSearchCV/_GridSearchCV}"
  mv -f "$f" "$OUTDIR/$new"
done
rmdir "$OUTDIR/results" 2>/dev/null || true

echo "DONE: $TRAIN -> $OUTDIR"
