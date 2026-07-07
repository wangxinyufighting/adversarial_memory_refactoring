#!/usr/bin/env bash
set -euo pipefail

# Complete test script for online GRPO memory refactoring training
# This script demonstrates the full workflow from graph preparation to training

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=============================================="
echo "Online GRPO Training - Complete Test Script"
echo "=============================================="
echo ""

# Configuration
GRAPHS_SOURCE="outputs/case_graphs_deepseek_test"
TRAIN_GRAPHS="outputs/case_graphs_train"
VAL_GRAPHS="outputs/case_graphs_val"
TEST_GRAPHS="outputs/case_graphs_test"
MODEL_PATH="${MODEL_PATH:-/mnt/local2/wxy/models/Qwen3-0.6B}"
OUTPUT_BASE="outputs/online_grpo_experiment"

# Step 1: Check if graphs exist
echo "Step 1: Checking for case graphs..."
if [ ! -d "${GRAPHS_SOURCE}" ]; then
    echo "Error: Source graphs directory not found: ${GRAPHS_SOURCE}"
    echo "Please run: ./scripts/build_case_graphs.sh first"
    exit 1
fi

GRAPH_COUNT=$(find "${GRAPHS_SOURCE}" -name "*.case_graph.json" | wc -l)
echo "Found ${GRAPH_COUNT} case graphs in ${GRAPHS_SOURCE}"

if [ "${GRAPH_COUNT}" -eq 0 ]; then
    echo "Error: No case graphs found"
    exit 1
fi
echo ""

# Step 2: Split graphs into train/val/test
echo "Step 2: Splitting graphs into train/val/test sets..."
if [ ! -d "${TRAIN_GRAPHS}" ]; then
    python3 scripts/split_case_graphs.py "${GRAPHS_SOURCE}" \
        --output-base outputs \
        --train-ratio 0.6 \
        --val-ratio 0.2 \
        --seed 42
    echo "Graphs split complete"
else
    echo "Split graphs already exist, skipping..."
fi
echo ""

# Step 3: Verify split
echo "Step 3: Verifying split..."
TRAIN_COUNT=$(find "${TRAIN_GRAPHS}" -name "*.case_graph.json" 2>/dev/null | wc -l || echo 0)
VAL_COUNT=$(find "${VAL_GRAPHS}" -name "*.case_graph.json" 2>/dev/null | wc -l || echo 0)
TEST_COUNT=$(find "${TEST_GRAPHS}" -name "*.case_graph.json" 2>/dev/null | wc -l || echo 0)

echo "Train set: ${TRAIN_COUNT} graphs"
echo "Val set: ${VAL_COUNT} graphs"
echo "Test set: ${TEST_COUNT} graphs"
echo ""

# Step 4: Run unit tests
echo "Step 4: Running unit tests..."
python3 -m pytest tests/test_online_memory_dataset.py -v
if [ $? -ne 0 ]; then
    echo "Error: Unit tests failed"
    exit 1
fi
echo "All tests passed!"
echo ""

# Step 5: Test dataset initialization
echo "Step 5: Testing dataset initialization..."
python3 -m case_graph.online_memory_cli \
    --graphs "${TRAIN_GRAPHS}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_BASE}/init_test" \
    --config configs/online_grpo.yaml \
    --episodes-per-case 10

if [ $? -ne 0 ]; then
    echo "Error: Dataset initialization failed"
    exit 1
fi
echo "Dataset initialization successful!"
echo ""

# Step 6: Run training (placeholder)
echo "Step 6: Running online GRPO training..."
echo "Note: This is a placeholder training run (verl integration pending)"
echo ""

TRAIN_OUTPUT="${OUTPUT_BASE}/training"
mkdir -p "${TRAIN_OUTPUT}"

echo "Training configuration:"
echo "  Graphs: ${TRAIN_GRAPHS}"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${TRAIN_OUTPUT}"
echo "  Episodes per case: 100"
echo "  Rollout N: 4"
echo "  Batch size: 4"
echo ""

# For now, just test the setup without full verl integration
export GRAPHS_DIR="${TRAIN_GRAPHS}"
export MODEL_PATH="${MODEL_PATH}"
export OUTPUT_DIR="${TRAIN_OUTPUT}"
export EPISODES_PER_CASE=10  # Small number for testing
export ROLLOUT_N=4
export TRAIN_BATCH_SIZE=2

echo "Running training setup test..."
python3 -m case_graph.online_memory_cli \
    --graphs "${TRAIN_GRAPHS}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${TRAIN_OUTPUT}" \
    --config configs/online_grpo.yaml \
    --episodes-per-case 10 \
    --rollout-n 4 \
    --train-batch-size 2

echo ""
echo "=============================================="
echo "Complete Test Script Finished"
echo "=============================================="
echo ""
echo "Summary:"
echo "  ✓ Case graphs found and split"
echo "  ✓ Unit tests passed"
echo "  ✓ Dataset initialization successful"
echo "  ✓ Training setup tested"
echo ""
echo "Next steps:"
echo "  1. Implement full verl trainer integration in online_memory_trainer.py"
echo "  2. Run full training: bash scripts/run_online_memory_grpo.sh"
echo "  3. Monitor training logs and memory state evolution"
echo "  4. Evaluate on validation and test sets"
echo ""
echo "Output locations:"
echo "  Train graphs: ${TRAIN_GRAPHS}/"
echo "  Val graphs: ${VAL_GRAPHS}/"
echo "  Test graphs: ${TEST_GRAPHS}/"
echo "  Training output: ${TRAIN_OUTPUT}/"
echo "  Initial memory states: ${TRAIN_OUTPUT}/initial_memory_states/"
echo ""
