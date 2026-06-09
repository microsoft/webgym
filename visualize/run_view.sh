#!/bin/bash
# Launch the trajectory viewer (Gradio).
#
# Usage:
#   AREAL_EXPERIMENTS=/path/to/experiments \
#       bash visualize/run_view.sh --split train   # view train trajectories
#   AREAL_EXPERIMENTS=/path/to/experiments \
#       bash visualize/run_view.sh --split test    # view test trajectories
#   AREAL_EXPERIMENTS=/path/to/experiments \
#       bash visualize/run_view.sh --split test --port 7861
#
# AREAL_EXPERIMENTS must point at the AReaL experiments root that contains
# logs/root/<exp>/<trial>/{rollout,test-rollout}/<version>/ . Pick that up
# from your launch script or set it explicitly.
set -euo pipefail

SPLIT="test"
PORT=7860
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        *)       EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${AREAL_EXPERIMENTS:-}" ]]; then
    echo "AREAL_EXPERIMENTS environment variable is required." >&2
    echo "It should point at the experiments root, e.g." >&2
    echo "  export AREAL_EXPERIMENTS=/path/to/areal/experiments" >&2
    exit 1
fi
LOGS_DIR="${AREAL_EXPERIMENTS}/logs/root"

# Find the most recent experiment with the requested split
if [[ "${SPLIT}" == "test" ]]; then
    SUBDIR="test-rollout"
elif [[ "${SPLIT}" == "train" ]]; then
    SUBDIR="rollout"
else
    echo "Unknown split: ${SPLIT} (use train or test)" >&2
    exit 1
fi

# Find the most recently modified rollout directory
ROLLOUT_DIR=$(find "${LOGS_DIR}" -maxdepth 3 -name "${SUBDIR}" -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)

if [[ -z "${ROLLOUT_DIR}" ]]; then
    echo "No ${SUBDIR} directory found under ${LOGS_DIR}" >&2
    echo "Available experiments:" >&2
    ls "${LOGS_DIR}" 2>/dev/null >&2 || true
    exit 1
fi

echo "Viewing: ${ROLLOUT_DIR}"
echo "Port:    ${PORT}"
echo ""

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
exec python3 "${SCRIPT_DIR}/view_trajs.py" "${ROLLOUT_DIR}" --port "${PORT}" "${EXTRA_ARGS[@]}"
