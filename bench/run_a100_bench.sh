#!/bin/bash
# A100 before/after benchmark for lingbot_vla_v2 acceleration work.
# before = pristine PR head (eager + fp32 attention hardcoded)
# after  = patched (sdpa default + bf16 attention + invariant hoisting + MoE sync-free)
set -u
cd ~/accel-bench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lvla-lerobot
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

CKPT=${CKPT:-$HOME/lvla_scratch/lingbot-vla-v2-6b-lerobot-v2}
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -1 | cut -d, -f1)
echo "Using GPU $GPU ($(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU) free)"
export CUDA_VISIBLE_DEVICES=$GPU

run() {  # run <tag> <repo> <mode> [extra args...]
  local tag=$1 repo=$2 mode=$3; shift 3
  echo "=== $tag / $mode ==="
  PYTHONPATH=$HOME/accel-bench/$repo/src python ~/accel-bench/bench_lingbot_v2.py "$mode" \
    --ckpt "$CKPT" "$@" 2>&1 | grep -v Warning | tail -40
  echo
}

ITERS=${ITERS:-20}
BATCH=${BATCH:-8}

# --- inference ---
run before repo-before infer --iters $ITERS                       # eager + fp32 (old code)
run after  repo-after  infer --iters $ITERS --attn sdpa           # new defaults
run abl_sdpa_fp32 repo-after infer --iters $ITERS --attn sdpa --attn-fp32   # backend-only gain

# --- training (fwd+bwd, no optimizer) ---
run before repo-before train --iters 10 --batch $BATCH
run after  repo-after  train --iters 10 --batch $BATCH --attn sdpa

# --- parity check (reference sdpa+fp32 vs bf16 / flex_cached) ---
run after  repo-after  check-flex
echo DONE
