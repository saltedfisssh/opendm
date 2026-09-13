#!/usr/bin/env bash
# One rung per invocation; run the invocations sequentially on the same GPUs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUNG="${1:-s0}"
if (( $# )); then shift; fi
case "$RUNG" in
  s0) REP=joint; MODE=vector ;;
  s1) REP=eef_local; MODE=vector ;;
  s2) REP=eef_local; MODE=se3 ;;
  s2_pair) REP=eef_local_pair; MODE=se3 ;;
  s3a) REP=eef_unified; MODE=se3 ;;
  s3) REP=eef_unified_pair; MODE=se3 ;;
  s4) REP=eef_gravity_random; MODE=se3 ;;
  *) echo "Unknown rung: $RUNG" >&2; exit 2 ;;
esac
# Training uses OpenDM's environment; RoboTwin's .venv is for simulator checks.
PYTHON="${OPENDM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2_ablation}"
SOURCE="${SOURCE:-/kpfs_ssd/data/wzy/data/Dexmal/robotwin2-full/robotwin2.0}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-8}" --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" playground/dm05_robotwin2_ablation.py \
  --trainer-config.per-device-train-batch-size "${BATCH_SIZE:-2}" \
  --trainer-config.gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-8}" \
  --task train --data-config.dataset-name "robotwin2_ablation_$RUNG" \
  --data-config.relative-mode "$MODE" \
  --data-config.jsonl-dir "$DATA_ROOT/$REP/jsonl/train" \
  --data-config.image-dir "$SOURCE/video" --data-config.norm-stats-root "$DATA_ROOT/norm_stats" \
  --model-config.model-name-or-path "${MODEL_PATH:-$REPO_ROOT/checkpoints/DM05}" \
  --trainer-config.output-dir "${OUTPUT_ROOT:-$REPO_ROOT/user_checkpoints/robotwin2_ablation}/$RUNG" \
  "$@"
