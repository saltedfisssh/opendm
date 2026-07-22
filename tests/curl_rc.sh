#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="http://127.0.0.1:9999/process_frame"
EMBODIMENT="arx5"
MANIFEST="${TABLE30V2_MANIFEST:-${ROOT_DIR}/data/table30v2_dexdata_binary/manifest.json}"
SAMPLE_LINE=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: tests/curl_rc.sh [options]

Send a real Table30v2 observation to a running OpenDM inference service.

Options:
  --embodiment NAME  arx5, ur5, aloha, dos_w1, or all (default: arx5)
  --url URL          Service URL (default: http://127.0.0.1:9999/process_frame)
  --manifest PATH    Converted Table30v2 manifest.json
  --sample-line N    JSONL line to test, starting at 1 (default: 1)
  --dry-run          Extract inputs and print request details without HTTP POST
  -h, --help         Show this help

For --embodiment all, each embodiment can use a different service:
  ARX5_URL=... UR5_URL=... ALOHA_URL=... DOS_W1_URL=... tests/curl_rc.sh --embodiment all

The inference service must match the embodiment:
  arx5:   output_action_dim=7,  image_keys=images_1 images_2 images_3
  ur5:    output_action_dim=7,  image_keys=images_1 images_2
  aloha:  output_action_dim=14, image_keys=images_1 images_2 images_3
  dos_w1: output_action_dim=14, image_keys=images_1 images_2 images_3
EOF
}

while (($#)); do
  case "$1" in
    --embodiment) EMBODIMENT="${2:?missing value for --embodiment}"; shift 2 ;;
    --url) BASE_URL="${2:?missing value for --url}"; shift 2 ;;
    --manifest) MANIFEST="${2:?missing value for --manifest}"; shift 2 ;;
    --sample-line) SAMPLE_LINE="${2:?missing value for --sample-line}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${EMBODIMENT}" in
  arx5|ur5|aloha|dos_w1|all) ;;
  *) echo "Invalid embodiment: ${EMBODIMENT}" >&2; exit 2 ;;
esac
[[ "${SAMPLE_LINE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--sample-line must be a positive integer" >&2
  exit 2
}
for command in jq ffmpeg curl; do
  command -v "${command}" >/dev/null || {
    echo "Required command not found: ${command}" >&2
    exit 2
  }
done
[[ -f "${MANIFEST}" ]] || {
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Pass the full-data manifest with --manifest." >&2
  exit 2
}
if [[ "${BASE_URL}" != */process_frame ]]; then
  BASE_URL="${BASE_URL%/}/process_frame"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/curl_rc.XXXXXX")"
# trap 'rm -rf "${TMP_DIR}"' EXIT

url_for() {
  case "$1" in
    arx5) echo "${ARX5_URL:-${BASE_URL}}" ;;
    ur5) echo "${UR5_URL:-${BASE_URL}}" ;;
    aloha) echo "${ALOHA_URL:-${BASE_URL}}" ;;
    dos_w1) echo "${DOS_W1_URL:-${BASE_URL}}" ;;
  esac
}

test_embodiment() {
  local embodiment="$1" url="$2" dataset_name="table30v2_${1}"
  local jsonl_dir image_dir image_count action_dim robot_type jsonl sample

  jsonl_dir="$(jq -er --arg name "${dataset_name}" '.datasets[$name].jsonl_dir' "${MANIFEST}")" || {
    echo "Dataset ${dataset_name} is not present in ${MANIFEST}" >&2
    return 1
  }
  image_dir="$(jq -er --arg name "${dataset_name}" '.datasets[$name].image_dir' "${MANIFEST}")"
  image_count="$(jq -er --arg name "${dataset_name}" '.datasets[$name].image_keys | length' "${MANIFEST}")"
  action_dim="$(jq -er --arg name "${dataset_name}" '.datasets[$name].state_dim' "${MANIFEST}")"
  jsonl="$(find "${jsonl_dir}" -type f -name '*.jsonl' -print -quit)"
  [[ -n "${jsonl}" ]] || { echo "No episode JSONL found under ${jsonl_dir}" >&2; return 1; }
  sample="$(sed -n "${SAMPLE_LINE}p" "${jsonl}")"
  [[ -n "${sample}" ]] || { echo "Line ${SAMPLE_LINE} does not exist in ${jsonl}" >&2; return 1; }

  case "${embodiment}" in
    arx5) robot_type="ARX5" ;;
    ur5) robot_type="UR5" ;;
    aloha) robot_type="ALOHA" ;;
    dos_w1) robot_type="DOS W1" ;;
  esac

  local prompt states state_dim
  prompt="$(jq -er '.prompt' <<<"${sample}")"
  states="$(jq -c '.state' <<<"${sample}")"
  state_dim="$(jq -er '.state | length' <<<"${sample}")"
  [[ "${state_dim}" == "${action_dim}" ]] || {
    echo "State dimension mismatch: sample=${state_dim}, manifest=${action_dim}" >&2
    return 1
  }

  local -a curl_args=(
    --silent --show-error --fail-with-body -X POST "${url}"
    -F "text=${prompt}" -F "robot_type=${robot_type}" -F "states=${states}"
  )
  local index image_meta video_path frame_idx image_path
  for ((index = 1; index <= image_count; index++)); do
    image_meta="$(jq -cer --arg key "images_${index}" '.[$key]' <<<"${sample}")"
    video_path="${image_dir}/$(jq -er '.url' <<<"${image_meta}")"
    frame_idx="$(jq -er '.frame_idx' <<<"${image_meta}")"
    image_path="${TMP_DIR}/${embodiment}_images_${index}.jpg"
    [[ -f "${video_path}" ]] || { echo "Video not found: ${video_path}" >&2; return 1; }
    ffmpeg -loglevel error -i "${video_path}" \
      -vf "select=eq(n\,${frame_idx})" -frames:v 1 -y "${image_path}"
    [[ -s "${image_path}" ]] || {
      echo "Failed to extract frame ${frame_idx} from ${video_path}" >&2
      return 1
    }
    curl_args+=(-F "image=@${image_path}")
  done

  echo "=== ${embodiment} ==="
  echo "dataset:      ${dataset_name}"
  echo "sample:       ${jsonl}:${SAMPLE_LINE}"
  echo "robot_type:   ${robot_type}"
  echo "state/action: ${action_dim}D"
  echo "images:       ${image_count}"
  echo "url:          ${url}"
  if ((DRY_RUN)); then
    echo "dry-run: request prepared successfully"
    return 0
  fi

  local response response_dim
  response="$(curl "${curl_args[@]}")"
  jq -e '.response | type == "array" and length > 0' <<<"${response}" >/dev/null || {
    echo "Invalid response: ${response}" >&2
    return 1
  }
  response_dim="$(jq -r '.response[0] | length' <<<"${response}")"
  [[ "${response_dim}" == "${action_dim}" ]] || {
    echo "Action dimension mismatch: expected ${action_dim}, got ${response_dim}" >&2
    echo "Response: ${response}" >&2
    return 1
  }
  jq . <<<"${response}"
  echo
}

if [[ "${EMBODIMENT}" == "all" ]]; then
  status=0
  for embodiment in arx5 ur5 aloha dos_w1; do
    url="$(url_for "${embodiment}")"
    [[ "${url}" == */process_frame ]] || url="${url%/}/process_frame"
    test_embodiment "${embodiment}" "${url}" || status=1
  done
  exit "${status}"
fi

url="$(url_for "${EMBODIMENT}")"
[[ "${url}" == */process_frame ]] || url="${url%/}/process_frame"
test_embodiment "${EMBODIMENT}" "${url}"
