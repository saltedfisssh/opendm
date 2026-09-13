#!/usr/bin/env bash
# Small-data end-to-end verification; intentionally separate from full-data runs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export OPENDM_PYTHON="${OPENDM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export SOURCE="${SOURCE:-/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0}"
export DATA_ROOT="${SMOKE_DATA_ROOT:-/tmp/robotwin2_ablation_smoke}"
export OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-$REPO_ROOT/user_checkpoints/robotwin2_smoke}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/DM05}"
export NPROC_PER_NODE=1 NNODES=1 NODE_RANK=0 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1
LOG_ROOT="${SMOKE_LOG_ROOT:-$REPO_ROOT/results/robotwin2_smoke}"
ASSETS="${ROBOTWIN_ASSETS:-/kpfs_ssd/data/wzy/dexbotic-benchmark/RoboTwin/assets}"
RUNGS=("$@")
if (( ${#RUNGS[@]} == 0 )); then RUNGS=(s0 s1 s2 s2_pair s3a s3 s4); fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory missing: $MODEL_PATH; set MODEL_PATH to the shared initialization." >&2
  exit 2
fi
# Reusing saved smoke checkpoints would resume instead of testing initialization.
for rung in "${RUNGS[@]}"; do
  case "$rung" in s0|s1|s2|s2_pair|s3a|s3|s4) ;; *) echo "Unknown rung: $rung" >&2; exit 2 ;; esac
  if [[ -d "$OUTPUT_ROOT/$rung" ]]; then
    echo "Output already exists: $OUTPUT_ROOT/$rung; choose a fresh SMOKE_OUTPUT_ROOT." >&2
    exit 2
  fi
done
if [[ -d "$DATA_ROOT/norm_stats" ]]; then
  echo "Smoke statistics already exist: $DATA_ROOT/norm_stats; choose a fresh SMOKE_DATA_ROOT." >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
"$OPENDM_PYTHON" -u script/robotwin2_prepare_ablation.py \
  --source "$SOURCE" --out-root "$DATA_ROOT" --assets "$ASSETS" \
  --limit "${SMOKE_EPISODES:-20}" --workers "${PREP_WORKERS:-10}" --rung "${RUNGS[@]}" \
  2>&1 | tee "$LOG_ROOT/prepare.log"
"$OPENDM_PYTHON" -u script/robotwin2_compute_norm_stats.py \
  --source "$SOURCE" --data-root "$DATA_ROOT" --workers 2 --batch-size 128 --rung "${RUNGS[@]}" \
  2>&1 | tee "$LOG_ROOT/norm.log"
for rung in "${RUNGS[@]}"; do
  bash script/train_robotwin2_ablation.sh "$rung" \
    --trainer-config.num-train-steps 2 --trainer-config.dataloader-num-workers 2 \
    --trainer-config.save-steps 2 --trainer-config.save-total-limit 1 \
    --trainer-config.save-only-model 2>&1 | tee "$LOG_ROOT/train_$rung.log"
done
