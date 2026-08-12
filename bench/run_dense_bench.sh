#!/bin/bash
# Full before/after bench for the dense-MoE change. Waits for the host to go
# idle (liuyue's LIBERO eval: load<20 and GPUs 0-3 free), then runs the whole
# inference + training matrix and drops JSON-able logs in ~/accel-bench/dense_bench/.
set -u
cd ~/accel-bench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lvla-lerobot
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
OUT=~/accel-bench/dense_bench
mkdir -p "$OUT"

# ---- wait for idle (max ~12h) ----
for i in $(seq 1 360); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -4 | awk '{s+=$1} END {print s}')
  if awk -v l="$L" -v g="$G" 'BEGIN{exit !(l<20 && g<1000)}'; then break; fi
  sleep 120
done
L=$(cut -d' ' -f1 /proc/loadavg)
G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -4 | awk '{s+=$1} END {print s}')
echo "start: $(date) load=$L gpu0-3=${G}MiB" | tee "$OUT/status.txt"

CKPT=$HOME/lvla_scratch/lingbot-vla-v2-6b-lerobot-v2
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES=$GPU
echo "using GPU $GPU" | tee -a "$OUT/status.txt"

run() {  # run <tag> <repo> <mode> [extra args...]
  local tag=$1 repo=$2 mode=$3; shift 3
  echo "=== $tag / $mode ($*) ==="
  PYTHONPATH=$HOME/accel-bench/$repo/src python ~/accel-bench/bench_lingbot_v2.py "$mode" \
    --ckpt "$CKPT" "$@" > "$OUT/${tag}_${mode}.log" 2>&1
  echo "exit=$? $(grep -o '"total_ms": [0-9.]*\|"step_ms": [0-9.]*' "$OUT/${tag}_${mode}.log" | head -1)"
}

# --- inference ---
run before             repo-before infer --iters 20
run after_dense        repo-after  infer --iters 20 --attn sdpa
run after_dense_compile repo-after infer --iters 20 --attn sdpa --compile

# --- training (fwd+bwd) ---
run before             repo-before train --iters 10 --batch 4
run after_dense        repo-after  train --iters 10 --batch 4 --attn sdpa
run after_dense_gckpt  repo-after  train --iters 10 --batch 16 --attn sdpa --grad-ckpt

# --- parity ---
run after_dense        repo-after  check-flex

echo "ALLDONE $(date)" | tee -a "$OUT/status.txt"
