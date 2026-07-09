#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

python3 -m unittest discover -s tests -p 'test_*evaluation*.py'
python3 -m unittest discover -s tests -p 'test_unifiedmem_baseline.py'
