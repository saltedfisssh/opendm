#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="http://127.0.0.1:9999/process_frame"
DATASET="${ROOT_DIR}/data/arrange_the_fruits_piper/merged_binarized"
EPISODE=0
FRAME=0
EXPECTED_HORIZON=50
STATES=""
DRY_RUN=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: tests/curl_piper.sh [options]

Send a real two-camera Piper observation to an OpenDM inference service.

Options:
  --url URL          Service URL (default: http://127.0.0.1:9999/process_frame)
  --dataset PATH     LeRobot v2/v2.1 dataset root
  --episode N        Episode index (default: 0)
  --frame N          Frame index (default: 0)
  --horizon N        Expected response horizon (default: 50)
  --states JSON      Override the 7-D state instead of reading parquet
  --dry-run          Prepare and validate inputs without sending the request
  -h, --help         Show this help

Reading state from parquet requires pyarrow in the active Python environment.
EOF
}

while (($#)); do
  case "$1" in
    --url) BASE_URL="${2:?missing value for --url}"; shift 2 ;;
    --dataset) DATASET="${2:?missing value for --dataset}"; shift 2 ;;
    --episode) EPISODE="${2:?missing value for --episode}"; shift 2 ;;
    --frame) FRAME="${2:?missing value for --frame}"; shift 2 ;;
    --horizon) EXPECTED_HORIZON="${2:?missing value for --horizon}"; shift 2 ;;
    --states) STATES="${2:?missing value for --states}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in "${EPISODE}" "${FRAME}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Episode and frame must be non-negative integers" >&2; exit 2; }
done
[[ "${EXPECTED_HORIZON}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--horizon must be a positive integer" >&2
  exit 2
}
for command in jq ffmpeg curl "${PYTHON_BIN}"; do
  command -v "${command}" >/dev/null || { echo "Required command not found: ${command}" >&2; exit 2; }
done
[[ -f "${DATASET}/meta/info.json" ]] || { echo "Invalid LeRobot dataset: ${DATASET}" >&2; exit 2; }
[[ "${BASE_URL}" == */process_frame ]] || BASE_URL="${BASE_URL%/}/process_frame"

INFO="${DATASET}/meta/info.json"
CHUNK_SIZE="$(jq -er '.chunks_size // 1000' "${INFO}")"
CHUNK="$((EPISODE / CHUNK_SIZE))"
printf -v CHUNK_NAME '%03d' "${CHUNK}"
printf -v EPISODE_NAME 'episode_%06d' "${EPISODE}"
PARQUET="${DATASET}/data/chunk-${CHUNK_NAME}/${EPISODE_NAME}.parquet"
[[ -f "${PARQUET}" ]] || { echo "Parquet file not found: ${PARQUET}" >&2; exit 2; }

mapfile -t VIDEO_KEYS < <(jq -er '.features | to_entries[] | select(.value.dtype == "video") | .key' "${INFO}")
[[ "${#VIDEO_KEYS[@]}" == 2 ]] || {
  echo "Expected exactly two video features, found ${#VIDEO_KEYS[@]}" >&2
  exit 2
}

if [[ -z "${STATES}" ]]; then
  STATES="$(${PYTHON_BIN} - "${PARQUET}" "${FRAME}" <<'PY'
import json
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    raise SystemExit("pyarrow is required; activate the OpenDM environment or pass --states JSON")

path, frame = sys.argv[1], int(sys.argv[2])
table = pq.read_table(path, columns=["observation.state", "frame_index"])
rows = table.to_pylist()
matches = [row for row in rows if int(row["frame_index"]) == frame]
if not matches:
    raise SystemExit(f"frame {frame} is not present in {path}")
print(json.dumps([float(value) for value in matches[0]["observation.state"]]))
PY
)"
fi
jq -e 'type == "array" and length == 7 and all(.[]; type == "number")' <<<"${STATES}" >/dev/null || {
  echo "Piper state must be a JSON array containing seven numbers: ${STATES}" >&2
  exit 2
}
STATES="$(jq -c . <<<"${STATES}")"

PROMPT="$(jq -er 'select(.task_index == 0) | .task' "${DATASET}/meta/tasks.jsonl" | head -n 1)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/curl_piper.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

curl_args=(
  --silent --show-error --fail-with-body -X POST "${BASE_URL}"
  -F "text=${PROMPT}"
  -F 'robot_type=Piper'
  -F "states=${STATES}"
)

for index in 0 1; do
  video="${DATASET}/videos/chunk-${CHUNK_NAME}/${VIDEO_KEYS[$index]}/${EPISODE_NAME}.mp4"
  image="${TMP_DIR}/images_$((index + 1)).jpg"
  [[ -f "${video}" ]] || { echo "Video not found: ${video}" >&2; exit 2; }
  ffmpeg -loglevel error -i "${video}" -vf "select=eq(n\,${FRAME})" -frames:v 1 -y "${image}"
  [[ -s "${image}" ]] || { echo "Could not extract frame ${FRAME} from ${video}" >&2; exit 1; }
  curl_args+=(-F "image=@${image}")
done

echo "Piper inference request"
echo "  sample:     episode=${EPISODE}, frame=${FRAME}"
echo "  prompt:     ${PROMPT}"
echo "  state:      ${STATES}"
echo "  cameras:    ${VIDEO_KEYS[*]}"
echo "  descriptor: PIPER (6 joints + gripper)"
echo "  endpoint:   ${BASE_URL}"
if ((DRY_RUN)); then
  echo "Dry-run passed: request inputs are valid."
  exit 0
fi

response="$(curl "${curl_args[@]}")"
jq -e --argjson horizon "${EXPECTED_HORIZON}" '
  .response | type == "array" and length == $horizon and
  all(.[]; type == "array" and length == 7 and all(.[]; type == "number" and isnan == false and isinfinite == false))
' <<<"${response}" >/dev/null || {
  echo "Invalid response; expected ${EXPECTED_HORIZON}x7 finite actions:" >&2
  jq . <<<"${response}" >&2 || echo "${response}" >&2
  exit 1
}

echo "Inference passed: response shape is ${EXPECTED_HORIZON}x7 and all actions are finite."
jq '.response as $actions | {
  shape: [($actions | length), ($actions[0] | length)],
  first_action: $actions[0],
  last_action: $actions[-1]
}' <<<"${response}"
