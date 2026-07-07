#!/usr/bin/env bash
set -euo pipefail

# Split case graphs into train/val/test sets

GRAPHS_DIR=${1:-outputs/case_graphs_deepseek_test}
OUTPUT_BASE=${2:-outputs}
TRAIN_RATIO=${TRAIN_RATIO:-0.6}
VAL_RATIO=${VAL_RATIO:-0.2}
SEED=${SEED:-42}

echo "Splitting graphs from: ${GRAPHS_DIR}"
echo "Train ratio: ${TRAIN_RATIO}"
echo "Val ratio: ${VAL_RATIO}"
echo "Test ratio: $((1 - TRAIN_RATIO - VAL_RATIO))"

# Create output directories
mkdir -p "${OUTPUT_BASE}/case_graphs_train"
mkdir -p "${OUTPUT_BASE}/case_graphs_val"
mkdir -p "${OUTPUT_BASE}/case_graphs_test"

# Get all graph files and shuffle
GRAPH_FILES=($(ls "${GRAPHS_DIR}"/*.case_graph.json | sort | shuf --random-source=<(yes "${SEED}")))
TOTAL=${#GRAPH_FILES[@]}

# Calculate split points
TRAIN_COUNT=$(python3 -c "import math; print(math.floor($TOTAL * $TRAIN_RATIO))")
VAL_COUNT=$(python3 -c "import math; print(math.floor($TOTAL * $VAL_RATIO))")
TEST_COUNT=$((TOTAL - TRAIN_COUNT - VAL_COUNT))

echo ""
echo "Total graphs: ${TOTAL}"
echo "Train: ${TRAIN_COUNT} graphs"
echo "Val: ${VAL_COUNT} graphs"
echo "Test: ${TEST_COUNT} graphs"

# Split files
TRAIN_FILES=("${GRAPH_FILES[@]:0:$TRAIN_COUNT}")
VAL_FILES=("${GRAPH_FILES[@]:$TRAIN_COUNT:$VAL_COUNT}")
TEST_FILES=("${GRAPH_FILES[@]:$((TRAIN_COUNT + VAL_COUNT)):$TEST_COUNT}")

# Copy files
echo ""
echo "Copying to train set..."
for f in "${TRAIN_FILES[@]}"; do
    cp "$f" "${OUTPUT_BASE}/case_graphs_train/"
    echo "  $(basename $f)"
done

echo ""
echo "Copying to val set..."
for f in "${VAL_FILES[@]}"; do
    cp "$f" "${OUTPUT_BASE}/case_graphs_val/"
    echo "  $(basename $f)"
done

echo ""
echo "Copying to test set..."
for f in "${TEST_FILES[@]}"; do
    cp "$f" "${OUTPUT_BASE}/case_graphs_test/"
    echo "  $(basename $f)"
done

echo ""
echo "Split complete!"
echo "Train: ${OUTPUT_BASE}/case_graphs_train/ (${TRAIN_COUNT} files)"
echo "Val: ${OUTPUT_BASE}/case_graphs_val/ (${VAL_COUNT} files)"
echo "Test: ${OUTPUT_BASE}/case_graphs_test/ (${TEST_COUNT} files)"
