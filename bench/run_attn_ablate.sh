#!/bin/bash
# sdpa vs eager attention inference comparison, full locked stack (compile+prefix+gpu-preprocess).
set -u
SM=$HOME/accel-bench/attn_ablate
mkdir -p "$SM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lvla-lerobot
export HF_HUB_OFFLINE=1
CKPT=$HOME/lvla_scratch/robotwin-6b-lerobot
: > "$SM/STATUS"

arm () { # name extra-flags...  (sequential on GPU 4 for clean timing)
  local name=$1; shift
  ( cd ~/accel-bench && CUDA_VISIBLE_DEVICES=4 python bench_lingbot_v2.py infer \
      --ckpt "$CKPT" --iters 10 --warmup 3 \
      --compile --compile-mode max-autotune-no-cudagraphs --compile-prefix --gpu-preprocess \
      "$@" ) > "$SM/$name.log" 2>&1
  echo "$name exit=$?" >> "$SM/STATUS"
}

arm A_sdpa_bf16 --attn sdpa            # 当前默认(锁定栈)
arm B_eager_bf16 --attn eager          # 去掉 sdpa
arm C_eager_fp32 --attn eager --attn-fp32   # 上游原始行为
arm D_sdpa_fp32 --attn sdpa --attn-fp32     # 隔离 dtype 因素
echo "ALL DONE" >> "$SM/STATUS"
