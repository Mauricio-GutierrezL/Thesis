#!/usr/bin/env bash
#SBATCH --job-name=halfsep_one
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%j.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%j.err

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
LOGDIR="$ROOT/thesis/logs"
mkdir -p "$LOGDIR"

export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

# avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

MODEL="CircuitCentricClassifierHalfSeparable"

TRAIN="$ROOT/thesis/datasets_tests/linearly_separable/linearly_separable_2d_train.csv"
TEST="$ROOT/thesis/datasets_tests/linearly_separable/linearly_separable_2d_test.csv"

OUTDIR="$ROOT/thesis/my_results/linearly_separable/$MODEL"
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

# 2) evaluation with best hyperparams (produces *-best-hyperparams-results.csv)
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

# 3) move CSVs out of results/ and drop "_train" in filenames to match your naming style
shopt -s nullglob
for f in "$OUTDIR/results/"*.csv; do
  base="$(basename "$f")"
  new="${base/_train_GridSearchCV/_GridSearchCV}"
  mv -f "$f" "$OUTDIR/$new"
done
rmdir "$OUTDIR/results" 2>/dev/null || true

echo "DONE. Files in: $OUTDIR"
ls -1 "$OUTDIR" | head
