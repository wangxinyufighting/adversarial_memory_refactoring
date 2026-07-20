#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/merge_actor_checkpoint.sh ACTOR_PATH [OUTPUT_PATH]

Arguments:
  ACTOR_PATH   verl FSDP actor checkpoint directory.
  OUTPUT_PATH  Hugging Face output directory. Defaults to ACTOR_PATH_hf.

Environment:
  PYTHON_BIN                 Python executable. Defaults to python3.
  TRUST_REMOTE_CODE=1        Pass --trust-remote-code to the merger.
  USE_CPU_INITIALIZATION=0   Disable CPU model initialization during merge.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 2
fi

ACTOR_PATH="${1%/}"
OUTPUT_PATH="${2:-${ACTOR_PATH}_hf}"
OUTPUT_PATH="${OUTPUT_PATH%/}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$ACTOR_PATH" ]]; then
  echo "Actor checkpoint directory not found: $ACTOR_PATH" >&2
  exit 1
fi

if [[ ! -f "$ACTOR_PATH/fsdp_config.json" ]]; then
  echo "Missing FSDP metadata: $ACTOR_PATH/fsdp_config.json" >&2
  exit 1
fi

if [[ ! -f "$ACTOR_PATH/huggingface/config.json" ]]; then
  echo "Missing Hugging Face model config: $ACTOR_PATH/huggingface/config.json" >&2
  exit 1
fi

if [[ "$OUTPUT_PATH" == "$ACTOR_PATH" ]]; then
  echo "OUTPUT_PATH must be different from ACTOR_PATH." >&2
  exit 1
fi

if [[ -d "$OUTPUT_PATH" ]] && [[ -n "$(find "$OUTPUT_PATH" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output directory is not empty: $OUTPUT_PATH" >&2
  echo "Choose a new output path to avoid mixing model files." >&2
  exit 1
fi

# Infer world size from the complete shard set. This also repairs stale
# fsdp_config.json metadata, while preserving its original contents in .bak.
WORLD_SIZE="$("$PYTHON_BIN" - "$ACTOR_PATH" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path

actor_path = Path(sys.argv[1])
pattern = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")
shards_by_world_size = {}

for shard_path in actor_path.iterdir():
    match = pattern.match(shard_path.name)
    if match is None:
        continue
    world_size = int(match.group(1))
    rank = int(match.group(2))
    shards_by_world_size.setdefault(world_size, set()).add(rank)

if not shards_by_world_size:
    raise SystemExit(f"No model_world_size_*_rank_*.pt shards found in {actor_path}")
if len(shards_by_world_size) != 1:
    details = ", ".join(
        f"world_size={size}: ranks={sorted(ranks)}"
        for size, ranks in sorted(shards_by_world_size.items())
    )
    raise SystemExit(f"Mixed checkpoint shard sets found in {actor_path}: {details}")

world_size, ranks = next(iter(shards_by_world_size.items()))
expected_ranks = set(range(world_size))
if ranks != expected_ranks:
    missing = sorted(expected_ranks - ranks)
    extra = sorted(ranks - expected_ranks)
    raise SystemExit(
        f"Incomplete checkpoint for world_size={world_size}; "
        f"missing ranks={missing}, unexpected ranks={extra}"
    )

config_path = actor_path / "fsdp_config.json"
with config_path.open(encoding="utf-8") as handle:
    config = json.load(handle)

configured_world_size = config.get("world_size")
if configured_world_size != world_size:
    backup_path = actor_path / "fsdp_config.json.bak"
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    config["world_size"] = world_size
    temporary_path = actor_path / "fsdp_config.json.tmp"
    temporary_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)
    print(
        f"Corrected fsdp_config.json world_size from "
        f"{configured_world_size!r} to {world_size}; backup: {backup_path}",
        file=sys.stderr,
    )

print(world_size)
PY
)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

MERGE_ARGS=(
  -m verl.model_merger merge
  --backend fsdp
  --local_dir "$ACTOR_PATH"
  --target_dir "$OUTPUT_PATH"
)

if [[ "${USE_CPU_INITIALIZATION:-1}" == "1" ]]; then
  MERGE_ARGS+=(--use_cpu_initialization)
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  MERGE_ARGS+=(--trust-remote-code)
fi

echo "Merging FSDP actor checkpoint"
echo "  Source:     $ACTOR_PATH"
echo "  World size: $WORLD_SIZE"
echo "  Output:     $OUTPUT_PATH"

"$PYTHON_BIN" "${MERGE_ARGS[@]}"

"$PYTHON_BIN" - "$OUTPUT_PATH" <<'PY'
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
if not (output_path / "config.json").is_file():
    raise SystemExit(f"Merge output is missing config.json: {output_path}")

weight_patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
if not any(any(output_path.glob(pattern)) for pattern in weight_patterns):
    raise SystemExit(f"Merge output contains no model weight files: {output_path}")
PY

echo "Merged model ready: $OUTPUT_PATH"
