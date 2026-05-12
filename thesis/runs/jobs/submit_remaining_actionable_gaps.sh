#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

count_ccc_hps_tasks() {
  python - <<'PY'
from pathlib import Path

root = Path("/home/fwm91820/qml_benchmarks/qml-benchmarks-main")
data_dir = root / "thesis/datasets_tests/bars_and_stripes"
trains = sorted(data_dir.glob("bars_and_stripes_*noise_train.csv"))
labels = [
    "CircuitCentricClassifier",
    "CircuitCentricClassifierHalfSeparableRandom50",
    "CircuitCentricClassifierSeparable",
]
target_task_ids = [1, 5, 9, 3, 7, 11]

count = 0
for task_id in target_task_ids:
    model_index, data_index = divmod(task_id, len(trains))
    dataset_name = trains[data_index].stem.removesuffix("_train")
    results_dir = root / "thesis/my_results/bars_and_stripes" / labels[model_index] / "results"
    if not (results_dir / f"{labels[model_index]}_{dataset_name}_GridSearchCV-best-hyperparams.csv").exists():
        count += 1
print(count)
PY
}

count_ccc_score_tasks() {
  python - <<'PY'
from pathlib import Path

root = Path("/home/fwm91820/qml_benchmarks/qml-benchmarks-main")
data_dir = root / "thesis/datasets_tests/bars_and_stripes"
trains = sorted(data_dir.glob("bars_and_stripes_*noise_train.csv"))
labels = [
    "CircuitCentricClassifier",
    "CircuitCentricClassifierHalfSeparableRandom50",
    "CircuitCentricClassifierSeparable",
]
target_task_ids = [1, 5, 9, 3, 7, 11]

count = 0
for task_id in target_task_ids:
    model_index, data_index = divmod(task_id, len(trains))
    dataset_name = trains[data_index].stem.removesuffix("_train")
    results_dir = root / "thesis/my_results/bars_and_stripes" / labels[model_index] / "results"
    hps = results_dir / f"{labels[model_index]}_{dataset_name}_GridSearchCV-best-hyperparams.csv"
    score = results_dir / f"{labels[model_index]}_{dataset_name}_GridSearchCV-best-hyperparams-results.csv"
    if hps.exists() and not score.exists():
        count += 1
print(count)
PY
}

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

ccc_combo_count=27
iqp_combo_count=12
num_seeds=5

ccc_hps_tasks="$(count_ccc_hps_tasks)"
ccc_score_tasks="$(count_ccc_score_tasks)"
iqp_standard_hps_tasks="$(count_iqp_standard_hps_tasks)"
iqp_highmem_hps_tasks="$(count_iqp_highmem_hps_tasks)"
iqp_score_tasks="$(count_iqp_score_tasks)"

ccc_hps_jobid=""
iqp_standard_hps_jobid=""
iqp_highmem_hps_jobid=""

if (( ccc_hps_tasks > 0 )); then
  echo "Submitting chunked CCC HPS tasks: $ccc_hps_tasks x $ccc_combo_count combos"
  ccc_hps_jobid="$(
    sbatch --parsable --array="0-$(( ccc_hps_tasks * ccc_combo_count - 1 ))%2" \
      "$ROOT/thesis/runs/jobs/hps_cccs_bars_and_stripes_remaining_chunked.sbatch"
  )"
  echo "CCC HPS job id: $ccc_hps_jobid"
else
  echo "No chunked CCC HPS tasks to submit."
fi

ccc_future_score_tasks=$(( (ccc_score_tasks + ccc_hps_tasks) * num_seeds ))
if (( ccc_future_score_tasks > 0 )); then
  echo "Submitting chunked CCC score tasks: $ccc_future_score_tasks"
  if [[ -n "$ccc_hps_jobid" ]]; then
    sbatch --dependency="afterok:$ccc_hps_jobid" \
      --array="0-$(( ccc_future_score_tasks - 1 ))%2" \
      "$ROOT/thesis/runs/jobs/score_cccs_bars_and_stripes_remaining_chunked.sbatch"
  else
    sbatch --array="0-$(( ccc_future_score_tasks - 1 ))%2" \
      "$ROOT/thesis/runs/jobs/score_cccs_bars_and_stripes_remaining_chunked.sbatch"
  fi
else
  echo "No chunked CCC score tasks to submit."
fi

if (( iqp_standard_hps_tasks > 0 )); then
  echo "Submitting chunked standard IQP HPS tasks: $iqp_standard_hps_tasks x $iqp_combo_count combos"
  iqp_standard_hps_jobid="$(
    sbatch --parsable --array="0-$(( iqp_standard_hps_tasks * iqp_combo_count - 1 ))%4" \
      "$ROOT/thesis/runs/jobs/hps_iqp_kernel_linearly_separable_remaining_chunked_standard.sbatch"
  )"
  echo "Standard IQP HPS job id: $iqp_standard_hps_jobid"
else
  echo "No chunked standard IQP HPS tasks to submit."
fi

if (( iqp_highmem_hps_tasks > 0 )); then
  echo "Submitting chunked high-memory IQP HPS tasks: $iqp_highmem_hps_tasks x $iqp_combo_count combos"
  iqp_highmem_hps_jobid="$(
    sbatch --parsable --array="0-$(( iqp_highmem_hps_tasks * iqp_combo_count - 1 ))%2" \
      "$ROOT/thesis/runs/jobs/hps_iqp_kernel_linearly_separable_remaining_chunked_highmem.sbatch"
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
      "$ROOT/thesis/runs/jobs/score_iqp_kernel_linearly_separable_remaining_chunked.sbatch"
  else
    sbatch --array="0-$(( iqp_future_score_tasks - 1 ))%4" \
      "$ROOT/thesis/runs/jobs/score_iqp_kernel_linearly_separable_remaining_chunked.sbatch"
  fi
else
  echo "No chunked IQP score tasks to submit."
fi
