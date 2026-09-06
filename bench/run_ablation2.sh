#!/bin/bash
# Ablation round 2: locked core = dense MoE + bf16 + compile + 10 steps + sdpa.
# Arms vary compile mode (and later: compile scope, preprocess). LIBERO real
# frames; fidelity = drift vs fp32/eager reference, same noise seed.
# Results: ~/accel-bench/ablation2_<tag>/
set -u
cd ~/accel-bench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lvla-lerobot
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

TAG=${1:-contended}
OUT=~/accel-bench/ablation2_$TAG
mkdir -p "$OUT"
CKPT=$HOME/lvla_scratch/lingbot-vla-v2-6b-lerobot-v2
DATA=$HOME/accel-bench/libero_data
H5="$DATA/task1.hdf5 $DATA/task2.hdf5"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES=$GPU
L=$(cut -d' ' -f1 /proc/loadavg)
echo "ablation2 start: $(date) load=$L gpu=$GPU" | tee "$OUT/status.txt"

FID=~/accel-bench/bench_libero_fidelity.py
REF=$OUT/ref.npz

run() {  # run <name> [args...]
  local name=$1; shift
  echo "=== $name ($*) ==="
  PYTHONPATH=$HOME/accel-bench/repo-after/src python "$FID" \
    --ckpt "$CKPT" --hdf5 $H5 --frames-per-demo 8 --max-demos 2 --iters 3 \
    "$@" > "$OUT/${name}.log" 2>&1
  echo "exit=$? $(grep -o '"model_ms_mean": [0-9.]*\|"drift_rel_l2_mean": [0-9.e-]*' "$OUT/${name}.log" | tr '\n' ' ')"
}

# R0: full-precision reference (eager attention + fp32 + grouped MoE, no compile)
run R0_reference --attn eager --attn-fp32 --moe-dense-max-tokens 0 --save-ref "$REF"

# C1: locked core, default compile mode
run C1_core_compile_default --attn sdpa --no-attn-fp32 --compile --ref "$REF"

# C2: locked core, max-autotune compile mode
run C2_core_compile_maxautotune --attn sdpa --no-attn-fp32 --compile \
  --compile-mode max-autotune-no-cudagraphs --ref "$REF"

# D1 (diagnostic): compile + grouped triton MoE (why dense is locked in)
run D1_diag_compile_grouped --attn sdpa --no-attn-fp32 --compile \
  --moe-dense-max-tokens 0 --ref "$REF"

echo "ABLATION2_ALLDONE $(date)" | tee -a "$OUT/status.txt"
