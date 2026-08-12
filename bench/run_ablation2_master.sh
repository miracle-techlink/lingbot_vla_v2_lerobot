#!/bin/bash
# Master unattended ablation runner (locked core: dense MoE + bf16 + compile +
# 10 denoise steps + sdpa). Self-contained: downloads data, smoke-tests, runs
# all arms, aggregates a summary table. Safe to re-run (skips finished arms).
# Results: ~/accel-bench/ablation2_contended/
set -u
exec > >(tee -a ~/accel-bench/ablation2_master.log) 2>&1
cd ~/accel-bench
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lvla-lerobot
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

OUT=~/accel-bench/ablation2_contended
DATA=~/accel-bench/libero_data
mkdir -p "$OUT" "$DATA"
CKPT=$HOME/lvla_scratch/lingbot-vla-v2-6b-lerobot-v2
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES=$GPU
echo "=== master start $(date) load=$(cut -d' ' -f1 /proc/loadavg) gpu=$GPU ==="

# ---------- 1. LIBERO data (2 task files from libero_spatial) ----------
T1=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
T2=pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
dl() {  # dl <task> <outfile>
  local t=$1 out=$2
  [ -s "$out" ] && [ "$(stat -c%s "$out")" -gt 100000000 ] && { echo "have $out"; return 0; }
  for host in hf-mirror.com huggingface.co; do
    curl -sL --max-time 590 -o "$out" \
      "https://$host/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial/${t}_demo.hdf5" \
      && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 100000000 ] && { echo "downloaded $out from $host"; return 0; }
  done
  echo "FAILED to download $t"; return 1
}
dl "$T1" "$DATA/task1.hdf5" && dl "$T2" "$DATA/task2.hdf5" || { echo "DATA_FAIL"; exit 1; }

FID=~/accel-bench/bench_libero_fidelity.py
REF=$OUT/ref.npz

run() {  # run <name> [args...]  — skips if log already has model_ms_mean
  local name=$1; shift
  if grep -q '"model_ms_mean"' "$OUT/${name}.log" 2>/dev/null; then echo "skip $name (done)"; return 0; fi
  echo "=== $name $(date) ($*) ==="
  PYTHONPATH=$HOME/accel-bench/repo-after/src python "$FID" \
    --ckpt "$CKPT" --hdf5 "$DATA/task1.hdf5" "$DATA/task2.hdf5" \
    --frames-per-demo 8 --max-demos 2 --iters 3 \
    "$@" > "$OUT/${name}.log" 2>&1
  echo "exit=$? $(grep -o '"model_ms_mean": [0-9.]*' "$OUT/${name}.log" | head -1)"
}

# ---------- 2. smoke test (1 frame, eager, fast) ----------
echo "--- smoke ---"
PYTHONPATH=$HOME/accel-bench/repo-after/src python "$FID" \
  --ckpt "$CKPT" --hdf5 "$DATA/task1.hdf5" --frames-per-demo 1 --max-demos 1 --iters 1 \
  --attn sdpa --no-attn-fp32 > "$OUT/smoke.log" 2>&1 \
  && echo "smoke OK" || { echo "SMOKE_FAIL — see $OUT/smoke.log"; exit 1; }

# ---------- 3. reference + arms ----------
run R0_reference --attn eager --attn-fp32 --moe-dense-max-tokens 0 --save-ref "$REF"
run C1_core_compile_default --attn sdpa --no-attn-fp32 --compile --ref "$REF"
run C2_core_compile_maxautotune --attn sdpa --no-attn-fp32 --compile \
  --compile-mode max-autotune-no-cudagraphs --ref "$REF"
run D1_diag_compile_grouped --attn sdpa --no-attn-fp32 --compile \
  --moe-dense-max-tokens 0 --ref "$REF"

# ---------- 4. aggregate ----------
python - "$OUT" <<'EOF'
import json, re, sys, glob, os
out = sys.argv[1]
rows = []
for log in sorted(glob.glob(os.path.join(out, "*.log"))):
    name = os.path.basename(log)[:-4]
    if name == "smoke":
        continue
    txt = open(log).read()
    m = re.search(r"\{[^{}]*\"model_ms_mean\"[^{}]*\}", txt, re.S)
    if not m:
        rows.append((name, None))
        continue
    try:
        rows.append((name, json.loads(m.group(0))))
    except Exception:
        rows.append((name, None))
lines = ["# Ablation round 2 (locked: dense MoE + bf16 + compile + 10 steps + sdpa)",
         "", f"host load at start: see status.txt; frames=32, iters=3/arm", "",
         "| arm | model ms/chunk | preprocess ms | drift rel L2 | drift max abs | cosine |",
         "|---|---|---|---|---|---|"]
for name, d in rows:
    if d is None:
        lines.append(f"| {name} | FAILED | | | | |")
        continue
    lines.append(
        f"| {name} | {d['model_ms_mean']:.1f} | {d['preprocess_ms_mean']:.1f} | "
        f"{d.get('drift_rel_l2_mean', 0):.4f} | {d.get('drift_max_abs', 0):.4f} | "
        f"{d.get('drift_cosine_mean', 1):.6f} |"
    )
open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

echo "ABLATION2_ALLDONE $(date)" | tee "$OUT/status.txt"
