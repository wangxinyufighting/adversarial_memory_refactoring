#!/usr/bin/env bash
# Certify constructed memories have sufficient answer-source coverage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

MEMORY_DIR="${MEMORY_DIR:-}"
GRAPHS="${GRAPHS:-}"
OUTPUT="${OUTPUT:-}"
TOP_K="${TOP_K:-10}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/online_grpo.yaml}"

show_usage() {
    cat << EOF
Usage: $(basename "$0") --memory-dir DIR --graphs DIR [options]

Certify that constructed memories have adequate coverage of answer sources.

Required:
  --memory-dir DIR         Directory containing <case_id>.json memory files
  --graphs DIR             CaseGraph directory or file

Optional:
  --output FILE            Certification report JSON (default: <memory-dir>/../certification_report.json)
  --top-k N               Top-K chunks to retrieve (default: 10)
  --training-config FILE   Config for retriever settings (default: configs/online_grpo.yaml)

Example:
  ./scripts/certify_memories.sh \\
    --memory-dir outputs/memory_states \\
    --graphs outputs/case_graphs_test \\
    --output outputs/certification.json
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --memory-dir) MEMORY_DIR="$2"; shift 2 ;;
        --graphs) GRAPHS="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --training-config) TRAINING_CONFIG="$2"; shift 2 ;;
        -h|--help) show_usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; show_usage >&2; exit 1 ;;
    esac
done

if [[ -z "$MEMORY_DIR" || -z "$GRAPHS" ]]; then
    echo "Error: --memory-dir and --graphs are required" >&2
    show_usage >&2
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    MEMORY_PARENT="$(dirname "$MEMORY_DIR")"
    OUTPUT="$MEMORY_PARENT/certification_report.json"
fi

export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

echo "Certifying memories..."
echo "  Memory dir: $MEMORY_DIR"
echo "  Graphs: $GRAPHS"
echo "  Output: $OUTPUT"
echo "  Top-K: $TOP_K"

python3 -m case_graph.certification_cli \
    --memory-dir "$MEMORY_DIR" \
    --graphs "$GRAPHS" \
    --output "$OUTPUT" \
    --top-k "$TOP_K" \
    --training-config "$TRAINING_CONFIG"

echo "Certification complete: $OUTPUT"
