#!/bin/bash
# Launch Co-Training V2: Adversarial Attacker and Defender Training

set -e

# Default paths - override with environment variables
GRAPHS_DIR=${GRAPHS_DIR:-"outputs/longmemeval_split/case_graphs_train"}
VAL_GRAPHS_DIR=${VAL_GRAPHS_DIR:-"outputs/longmemeval_split/case_graphs_val"}
ATTACKER_MODEL=${ATTACKER_MODEL:-"/mnt/local2/wxy/models/Qwen3-0.6B"}
DEFENDER_MODEL=${DEFENDER_MODEL:-"/mnt/local2/wxy/models/Qwen3-0.6B"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/cotrain_v2"}
CONFIG_FILE=${CONFIG_FILE:-"configs/cotrain_v2.yaml"}
COTRAIN_ROUNDS=${COTRAIN_ROUNDS:-10}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Co-Training V2: Attacker & Defender"
echo "=========================================="
echo "Training graphs: $GRAPHS_DIR"
echo "Validation graphs: $VAL_GRAPHS_DIR"
echo "Attacker model: $ATTACKER_MODEL"
echo "Defender model: $DEFENDER_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Config: $CONFIG_FILE"
echo "Rounds: $COTRAIN_ROUNDS"
echo "=========================================="

# Setup environment
export PYTHONPATH="${ROOT_DIR}/compat:${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}"

# Update config with environment variables if provided
if [ -n "$COTRAIN_ROUNDS" ]; then
    echo "Setting cotrain_rounds to $COTRAIN_ROUNDS"
fi

# Run co-training
python3 -m case_graph.cotrain_v2_cli \
    --config "$CONFIG_FILE" \
    --train-graphs "$GRAPHS_DIR" \
    --val-graphs "$VAL_GRAPHS_DIR" \
    --attacker-init "$ATTACKER_MODEL" \
    --defender-init "$DEFENDER_MODEL" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Co-Training Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Final attacker: $OUTPUT_DIR/final_attacker_results.json"
echo "Final defender: $OUTPUT_DIR/final_defender_results.json"
