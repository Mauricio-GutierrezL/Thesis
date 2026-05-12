#!/usr/bin/env bash

set -euo pipefail

DEFAULT_JOB_IDS=(
  67009763
  67009764
  67009765
  67009766
  67009767
)

if (( $# > 0 )); then
  JOB_IDS=("$@")
else
  JOB_IDS=("${DEFAULT_JOB_IDS[@]}")
fi

job_csv="$(IFS=,; echo "${JOB_IDS[*]}")"

echo "== squeue =="
squeue -j "$job_csv"
echo

echo "== sacct =="
sacct -j "$job_csv" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode,Submit,Start,End
