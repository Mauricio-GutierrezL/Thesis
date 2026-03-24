#!/bin/bash
#SBATCH --job-name=eval_ls
#SBATCH --partition=c23ms
#SBATCH --account=thes2204
#SBATCH --array=0-151%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs
ulimit -n 4096 || true

# avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd ~/qml_benchmarks/qml-benchmarks-main
source ~/venvs/qmlb/bin/activate
export PYTHONNOUSERSITE=1

DATASET="linearly_separable"

# EDIT this list if any folder names differ (must match thesis/my_results/<MODEL>/...)
MODELS=(
  "CircuitCentricClassifierSeparable"
  "DressedQuantumCircuitClassifier"
  "DataReuploadingClassifier"
  "IQPVariationalClassifier"
  "QuantumBoltzmannMachine"
  "QuantumMetricLearner"
  "TreeTensorClassifier"
  "MLPClassifier"
)

DIMS=($(seq 2 20))

NM=${#MODELS[@]}
ND=${#DIMS[@]}

MODEL_INDEX=$(( SLURM_ARRAY_TASK_ID / ND ))
DIM_INDEX=$(( SLURM_ARRAY_TASK_ID % ND ))

if (( MODEL_INDEX >= NM )); then
  echo "TASK_ID out of range: $SLURM_ARRAY_TASK_ID"
  exit 0
fi

MODEL="${MODELS[$MODEL_INDEX]}"
D="${DIMS[$DIM_INDEX]}"

TRAIN="thesis/datasets_tests/${DATASET}/${DATASET}_${D}d_train.csv"
TEST="thesis/datasets_tests/${DATASET}/${DATASET}_${D}d_test.csv"

# results-path should be the model/d folder (script writes into OUT/results/)
OUT="thesis/my_results/${MODEL}/d${D}"
RES_DIR="${OUT}/results"
mkdir -p "$RES_DIR"

echo "MODEL=$MODEL  D=$D"
echo "TRAIN=$TRAIN"
echo "TEST=$TEST"
echo "OUT=$OUT"
python -V

# skip if train/test missing
if [[ ! -f "$TRAIN" ]] || [[ ! -f "$TEST" ]]; then
  echo "SKIP: missing train/test for d${D}"
  exit 0
fi

# best HPO settings file (your exact naming)
HPS_FILE="$(find "$RES_DIR" -maxdepth 1 -type f -name "${MODEL}_${DATASET}_${D}d_GridSearchCV-best-hyperparams.csv" -print -quit 2>/dev/null || true)"
if [[ -z "$HPS_FILE" ]]; then
  # fallback (in case naming differs slightly)
  HPS_FILE="$(find "$RES_DIR" -maxdepth 1 -type f -name "*_GridSearchCV-best-hyperparams.csv" -print -quit 2>/dev/null || true)"
fi
if [[ -z "$HPS_FILE" ]]; then
  echo "SKIP: missing best-hyperparams CSV in $RES_DIR"
  exit 0
fi

# skip if scoring already exists
if [[ -n "$(find "$RES_DIR" -maxdepth 1 -type f -name "*_GridSearchCV-best-hyperparams-results.csv" -print -quit 2>/dev/null || true)" ]]; then
  echo "SKIP: scoring results already exist in $RES_DIR"
  exit 0
fi

python -u scripts/score_with_best_hyperparameters.py \
  --classifier-name "$MODEL" \
  --trainset-path "$TRAIN" \
  --testset-path "$TEST" \
  --hyperparams-path "$HPS_FILE" \
  --results-path "$OUT" \
  --clean False \
  --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "DONE: $MODEL d$D"
