#!/usr/bin/env bash
set -euo pipefail

python3 -m case_graph.trace_to_grpo_data_cli "$@"
