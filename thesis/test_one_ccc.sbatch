#!/usr/bin/env bash
#SBATCH --job-name=ccc_test
#SBATCH --account=thes2204
#SBATCH --partition=c23mm
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%j.out
#SBATCH --error=/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/logs/%x-%j.err

set -euo pipefail

ROOT="/home/fwm91820/qml_benchmarks/qml-benchmarks-main"
mkdir -p "$ROOT/thesis/logs"

# IMPORTANT: don't use ~/.local packages
export PYTHONNOUSERSITE=1

# Prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Use your venv
source /home/fwm91820/venvs/qmlb/bin/activate

python -c "import pandas, pytz; print('ENV_OK', pandas.__version__)"

cd "$ROOT"

srun python scripts/run_hyperparameter_search.py \
  --model CircuitCentricClassifier \
  --dataset-path "/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/datasets_tests/hidden_manifold_diff/hidden_manifold-10d-6manifold_train.csv" \
  --results-path "/home/fwm91820/qml_benchmarks/qml-benchmarks-main/thesis/my_results/hidden_manifold_diff/CircuitCentricClassifier" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --clean True \
  --hyperparameter-scoring accuracy roc_auc \
  --hyperparameter-refit accuracy
