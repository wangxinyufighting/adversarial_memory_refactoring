#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"
PYTHON_BIN=${PYTHON_BIN:-python}

"${PYTHON_BIN}" -m case_graph.evaluation_cli "$@"
