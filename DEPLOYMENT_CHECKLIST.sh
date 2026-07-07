#!/usr/bin/env bash
# Deployment Checklist for Remote GPU Server
# ==========================================

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════╗
║     Remote GPU Server Deployment - Checklist                 ║
╚═══════════════════════════════════════════════════════════════╝

📋 PRE-DEPLOYMENT CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ Files to Transfer
  ├─ case_graph/online_memory_dataset.py
  ├─ case_graph/online_memory_trainer.py
  ├─ case_graph/online_memory_cli.py
  ├─ case_graph/grpo_adapter.py (modified)
  ├─ case_graph/pipeline.py (modified)
  ├─ configs/online_grpo.yaml
  ├─ scripts/run_online_memory_grpo.sh
  ├─ scripts/test_online_grpo_complete.sh
  ├─ scripts/split_case_graphs.py
  ├─ tests/test_online_memory_dataset.py
  ├─ outputs/case_graphs_train/ (3 graphs)
  ├─ outputs/case_graphs_val/ (1 graph)
  └─ outputs/case_graphs_test/ (1 graph)

□ Environment Setup
  ├─ Python 3.8+
  ├─ PyTorch with CUDA support
  ├─ verl library installed
  ├─ All dependencies from requirements.txt
  └─ GPU drivers and CUDA toolkit

□ API Keys (if needed)
  ├─ OPENAI_API_KEY (for FrozenLLMAttacker)
  ├─ DEEPSEEK_API_KEY (alternative)
  └─ WANDB_API_KEY (for logging)

□ Model Files
  ├─ Base model downloaded (e.g., Qwen3-0.6B)
  └─ Model path accessible from training script

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 DEPLOYMENT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Transfer Files to Server
--------------------------------

# Option A: rsync (recommended)
rsync -avz --progress \
  /Users/ganning/Documents/project_python/大模型记忆/adversarial_memory_refactoring/ \
  user@remote-server:/path/to/project/

# Option B: scp
scp -r /Users/ganning/Documents/project_python/大模型记忆/adversarial_memory_refactoring/ \
  user@remote-server:/path/to/project/

# Option C: git (if using version control)
git add .
git commit -m "Add online GRPO implementation"
git push
# Then on server: git pull


Step 2: Setup Environment on Server
-----------------------------------

ssh user@remote-server

cd /path/to/project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set PYTHONPATH
export PYTHONPATH="/path/to/project:${PYTHONPATH}"


Step 3: Verify Installation
---------------------------

# Run tests
python3 -m pytest tests/test_online_memory_dataset.py -v

# Should see: 10 passed in 0.0Xs


Step 4: Test Dataset Initialization
-----------------------------------

python3 -m case_graph.online_memory_cli \
  --graphs outputs/case_graphs_train \
  --model-path /path/to/model \
  --output-dir outputs/test_init \
  --config configs/online_grpo.yaml \
  --episodes-per-case 10


Step 5: Run Small-Scale Test
-----------------------------

# Test with 1 graph, 10 episodes
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/Qwen3-0.6B \
OUTPUT_DIR=outputs/test_small \
EPISODES_PER_CASE=10 \
ROLLOUT_N=2 \
TRAIN_BATCH_SIZE=2 \
bash scripts/run_online_memory_grpo.sh


Step 6: Full Training
---------------------

# Launch full training on GPU
GRAPHS_DIR=outputs/case_graphs_train \
MODEL_PATH=/path/to/Qwen3-0.6B \
OUTPUT_DIR=outputs/online_grpo_production \
EPISODES_PER_CASE=100 \
ROLLOUT_N=4 \
TRAIN_BATCH_SIZE=4 \
TOTAL_EPOCHS=3 \
bash scripts/run_online_memory_grpo.sh


Step 7: Monitor Training
------------------------

# Watch logs
tail -f outputs/online_grpo_production/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor memory evolution
ls -lh outputs/online_grpo_production/checkpoint_*/memory_states/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  CONFIGURATION TUNING FOR GPU SERVER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Edit configs/online_grpo.yaml:

# For single GPU (e.g., A100 40GB)
verl:
  actor_rollout_ref:
    rollout:
      tensor_model_parallel_size: 1
      gpu_memory_utilization: 0.8

  trainer:
    n_gpus_per_node: 1
    nnodes: 1

# For multi-GPU (e.g., 4x A100)
verl:
  actor_rollout_ref:
    rollout:
      tensor_model_parallel_size: 4
      gpu_memory_utilization: 0.8

  trainer:
    n_gpus_per_node: 4
    nnodes: 1

# For larger batch sizes
train_batch_size: 8
ppo_mini_batch_size: 4
rollout_n: 8

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: CUDA out of memory
Solution:
  - Reduce batch_size
  - Reduce rollout_n
  - Lower gpu_memory_utilization
  - Use smaller model

Issue: Import errors
Solution:
  - Check PYTHONPATH is set correctly
  - Verify all dependencies installed
  - Run: pip install -e .

Issue: API rate limits (FrozenLLMAttacker)
Solution:
  - Add sleep between attacks
  - Use local model instead
  - Pre-generate attacks offline

Issue: Slow training
Solution:
  - Enable tensor parallelism
  - Increase batch size if memory allows
  - Use faster inference backend (vllm)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 EXPECTED OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

outputs/online_grpo_production/
├── training_config.yaml
├── initial_memory_states/
│   ├── 58bf7951_ep0.json
│   ├── 1e043500_ep0.json
│   └── 51a45a95_ep0.json
├── checkpoint_step100/
│   ├── memory_states/
│   │   ├── 58bf7951_ep30.json
│   │   ├── 1e043500_ep35.json
│   │   └── 51a45a95_ep35.json
│   └── training_metadata.json
└── checkpoint_final/
    └── memory_states/

Training logs should show:
  ✓ Batch processing
  ✓ UID tracking per episode
  ✓ Reward computation
  ✓ Memory commits
  ✓ Checkpoint saving

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ POST-TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ Verify checkpoints saved
□ Check memory evolution (compare initial vs final)
□ Run validation on val set
□ Run evaluation on test set
□ Compare with offline GRPO baseline
□ Document results

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 READY FOR DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All files implemented and tested locally.
Transfer to GPU server and follow steps above.

For questions, refer to:
  • ONLINE_GRPO_README.md
  • scripts/QUICK_START.sh

Good luck with training! 🚀

EOF
