#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

count_iqp_standard_hps_tasks() {
  python - <<'PY'
from pathlib import Path

root = Path("/home/fwm91820/qml_benchmarks/qml-benchmarks-main")
data_dir = root / "thesis/datasets_tests/linearly_separable"
out_base = root / "thesis/my_results/linearly_separable"
models = [
    "IQPKernelClassifier",
    "IQPKernelClassifierHalfSeparable",
    "IQPKernelClassifierSeparable",
]
trains = sorted(data_dir.glob("linearly_separable_*d_train.csv"))

count = 0
for model in models:
    for train in trains:
        dataset_name = train.stem.removesuffix("_train")
        dim = int(dataset_name.removeprefix("linearly_separable_").removesuffix("d"))
        if dim >= 14:
            continue
        results_dir = out_base / model / "results"
        stem = f"{model}_{dataset_name}_GridSearchCV"
        hps = results_dir / f"{stem}-best-hyperparams.csv"
        score = results_dir / f"{stem}-best-hyperparams-results.csv"
        if not score.exists() and not hps.exists():
            count += 1
print(count)
PY
}

count_iqp_highmem_hps_tasks() {
  python - <<'PY'
from pathlib import Path

root = Path("/home/fwm91820/qml_benchmarks/qml-benchmarks-main")
data_dir = root / "thesis/datasets_tests/linearly_separable"
out_base = root / "thesis/my_results/linearly_separable"
models = [
    "IQPKernelClassifier",
    "IQPKernelClassifierHalfSeparable",
    "IQPKernelClassifierSeparable",
]
trains = sorted(data_dir.glob("linearly_separable_*d_train.csv"))

count = 0
for model in models:
    for train in trains:
        dataset_name = train.stem.removesuffix("_train")
        dim = int(dataset_name.removeprefix("linearly_separable_").removesuffix("d"))
        if dim < 14:
            continue
        results_dir = out_base / model / "results"
        stem = f"{model}_{dataset_name}_GridSearchCV"
        hps = results_dir / f"{stem}-best-hyperparams.csv"
        score = results_dir / f"{stem}-best-hyperparams-results.csv"
        if not score.exists() and not hps.exists():
            count += 1
print(count)
PY
}

count_iqp_score_tasks() {
  python - <<'PY'
from pathlib import Path

root = Path("/home/fwm91820/qml_benchmarks/qml-benchmarks-main")
data_dir = root / "thesis/datasets_tests/linearly_separable"
out_base = root / "thesis/my_results/linearly_separable"
models = [
    "IQPKernelClassifier",
    "IQPKernelClassifierHalfSeparable",
    "IQPKernelClassifierSeparable",
]
trains = sorted(data_dir.glob("linearly_separable_*d_train.csv"))

count = 0
for model in models:
    for train in trains:
        dataset_name = train.stem.removesuffix("_train")
        results_dir = out_base / model / "results"
        stem = f"{model}_{dataset_name}_GridSearchCV"
        hps = results_dir / f"{stem}-best-hyperparams.csv"
        score = results_dir / f"{stem}-best-hyperparams-results.csv"
        if hps.exists() and not score.exists():
            count += 1
print(count)
PY
}

iqp_combo_count=12
num_seeds=5

iqp_standard_hps_tasks="$(count_iqp_standard_hps_tasks)"
iqp_highmem_hps_tasks="$(count_iqp_highmem_hps_tasks)"
iqp_score_tasks="$(count_iqp_score_tasks)"

iqp_standard_hps_jobid=""
iqp_highmem_hps_jobid=""

if (( iqp_standard_hps_tasks > 0 )); then
  echo "Submitting chunked standard IQP HPS tasks: $iqp_standard_hps_tasks x $iqp_combo_count combos"
  iqp_standard_hps_jobid="$(
    sbatch --parsable --array="0-$(( iqp_standard_hps_tasks * iqp_combo_count - 1 ))%4" \
      "$ROOT/thesis/runs/jobs_thesis/hps_iqp_kernel_linearly_separable_remaining_chunked_standard.sbatch"
  )"
  echo "Standard IQP HPS job id: $iqp_standard_hps_jobid"
else
  echo "No chunked standard IQP HPS tasks to submit."
fi

if (( iqp_highmem_hps_tasks > 0 )); then
  echo "Submitting chunked high-memory IQP HPS tasks: $iqp_highmem_hps_tasks x $iqp_combo_count combos"
  iqp_highmem_hps_jobid="$(
    sbatch --parsable --array="0-$(( iqp_highmem_hps_tasks * iqp_combo_count - 1 ))%2" \
      "$ROOT/thesis/runs/jobs_thesis/hps_iqp_kernel_linearly_separable_remaining_chunked_highmem.sbatch"
  )"
  echo "High-memory IQP HPS job id: $iqp_highmem_hps_jobid"
else
  echo "No chunked high-memory IQP HPS tasks to submit."
fi

iqp_future_score_tasks=$(( (iqp_score_tasks + iqp_standard_hps_tasks + iqp_highmem_hps_tasks) * num_seeds ))
if (( iqp_future_score_tasks > 0 )); then
  echo "Submitting chunked IQP score tasks: $iqp_future_score_tasks"
  dependency_ids=()
  [[ -n "$iqp_standard_hps_jobid" ]] && dependency_ids+=("$iqp_standard_hps_jobid")
  [[ -n "$iqp_highmem_hps_jobid" ]] && dependency_ids+=("$iqp_highmem_hps_jobid")

  if (( ${#dependency_ids[@]} > 0 )); then
    dependency="$(IFS=:; echo "${dependency_ids[*]}")"
    sbatch --dependency="afterok:$dependency" \
      --array="0-$(( iqp_future_score_tasks - 1 ))%4" \
      "$ROOT/thesis/runs/jobs_thesis/score_iqp_kernel_linearly_separable_remaining_chunked.sbatch"
  else
    sbatch --array="0-$(( iqp_future_score_tasks - 1 ))%4" \
      "$ROOT/thesis/runs/jobs_thesis/score_iqp_kernel_linearly_separable_remaining_chunked.sbatch"
  fi
else
  echo "No chunked IQP score tasks to submit."
fi
