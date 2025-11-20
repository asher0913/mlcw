#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
TRAIN_SUBSET="${TRAIN_SUBSET:-10000}"
TEST_SUBSET="${TEST_SUBSET:-2000}"
TASK3_VARIANTS_RAW="${TASK3_VARIANTS:-rf}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run_all] NVIDIA GPU was requested but nvidia-smi is unavailable." >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[run_all] NVIDIA GPU check failed. Please run this script on a CUDA-enabled machine." >&2
  exit 1
fi

IFS=',' read -ra _TASK3_LIST <<< "${TASK3_VARIANTS_RAW}"
TASK3_ARGS=()
for variant in "${_TASK3_LIST[@]}"; do
  trimmed="$(echo "$variant" | xargs)"
  [[ -z "$trimmed" ]] && continue
  TASK3_ARGS+=("$trimmed")
done
if [[ ${#TASK3_ARGS[@]} -eq 0 ]]; then
  TASK3_ARGS=("rf")
fi

# Install dependencies if requested (default: install).
if [[ "${INSTALL_DEPS:-1}" -eq 1 ]]; then
  echo "[run_all] Installing Python dependencies..."
  "${PYTHON_BIN}" -m pip install --user -r "${PROJECT_ROOT}/requirements.txt"
else
  echo "[run_all] Skipping dependency installation."
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "[run_all] Task 3 backends: ${TASK3_ARGS[*]}"
echo "[run_all] Launching coursework experiments..."
"${PYTHON_BIN}" -m mlcw.run_pipeline \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --train-subset "${TRAIN_SUBSET}" \
  --test-subset "${TEST_SUBSET}" \
  --task3-backends "${TASK3_ARGS[@]}" \
  "$@"

echo "[run_all] Finished."
