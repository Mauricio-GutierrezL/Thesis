#!/usr/bin/env bash
#SBATCH --job-name=ccc_eval1
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%j.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%j.err

set -euo pipefail
ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"

mkdir -p "$ROOT/thesis/logs"

# Use venv and avoid ~/.local mixing
export PYTHONNOUSERSITE=1
source /home/fwm91820/venvs/qmlb/bin/activate

# Keep BLAS quiet; let XLA use the allocated cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

TRAIN="$ROOT/thesis/datasets_tests/hidden_manifold_diff/hidden_manifold-10d-6manifold_train.csv"
TEST="$ROOT/thesis/datasets_tests/hidden_manifold_diff/hidden_manifold-10d-6manifold_test.csv"

RESBASE="$ROOT/thesis/my_results/hidden_manifold_diff/CircuitCentricClassifier"
STEM="CircuitCentricClassifier_hidden_manifold-10d-6manifold_GridSearchCV"
BEST="$RESBASE/results/${STEM}-best-hyperparams.csv"

python -c "import sys,pandas,pytz; print('ENV_OK', sys.executable, pandas.__version__, pytz.__version__)"
test -f "$BEST" || { echo "Missing best hyperparams file: $BEST"; exit 2; }

cd "$ROOT"
srun python scripts/score_with_best_hyperparameters.py \
  --classifier-name CircuitCentricClassifier \
  --trainset-path "$TRAIN" \
  --testset-path "$TEST" \
  --hyperparams-path "$BEST" \
  --results-path "$RESBASE" \
  --clean True \
  --n-jobs "$SLURM_CPUS_PER_TASK"
