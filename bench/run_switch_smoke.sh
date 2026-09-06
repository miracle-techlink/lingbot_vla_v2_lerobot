#!/bin/bash
# Switch-launch smoke matrix for lingbot_vla_v2 (pr3967-combined).
# Every arm must exit 0 and produce its expected artifact (loss lines / infer JSON).
set -u
SM=$HOME/accel-bench/switch_smoke
mkdir -p "$SM"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lvla-lerobot
export HF_HUB_OFFLINE=1
CKPT=$HOME/lvla_scratch/robotwin-6b-lerobot
DS=$HOME/lvla_scratch/datasets/gpudad_so101_pick_cube_v2/gpudad_so101_pick_cube_v2
RC=$HOME/lvla_scratch/configs/gpudad_so101_robot_config.yaml
NS=$HOME/lvla_scratch/configs/gpudad_so101_norm_stats.json

train_arm () { # name gpu extra-flags...
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu lerobot-train \
    --dataset.repo_id=gpudad/so101_pick_cube_v2 --dataset.root="$DS" \
    --policy.type=lingbot_vla_v2 --policy.pretrained_path="$CKPT" \
    --policy.robot_config_path="$RC" --policy.norm_stats_path="$NS" \
    --policy.dtype=bfloat16 --policy.push_to_hub=false \
    --batch_size=1 --steps=3 --log_freq=1 --num_workers=2 \
    --save_checkpoint=false \
    --output_dir="$SM/out_$name" --job_name="$name" --wandb.enable=false "$@" \
    > "$SM/$name.log" 2>&1
  echo "$name exit=$?" >> "$SM/STATUS"
}

infer_arm () { # name gpu extra-flags...
  local name=$1 gpu=$2; shift 2
  ( cd ~/accel-bench && CUDA_VISIBLE_DEVICES=$gpu python bench_lingbot_v2.py infer \
      --ckpt "$CKPT" --iters 3 --warmup 1 "$@" ) > "$SM/$name.log" 2>&1
  echo "$name exit=$?" >> "$SM/STATUS"
}

bake_arm () { # I6: rollout-path launch — bake flags into config.json, from_pretrained, predict
  local name=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python - > "$SM/$name.log" 2>&1 << 'PYEOF'
import json, os, pathlib, shutil, sys, time
import torch
src = pathlib.Path.home() / "lvla_scratch/robotwin-6b-lerobot"
dst = pathlib.Path.home() / "accel-bench/switch_smoke/ckpt_baked"
if not dst.exists():
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.suffix == ".safetensors":
            os.symlink(f, dst / f.name)
        else:
            shutil.copy(f, dst / f.name)
cfg = json.loads((dst / "config.json").read_text())
cfg.update({
    "compile_predict_velocity": True,
    "compile_predict_velocity_mode": "max-autotune-no-cudagraphs",
    "compile_prefix": True,
    "preprocess_device": "cuda",
    "num_steps": 10,
})
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
print("baked config:", {k: cfg[k] for k in ["compile_predict_velocity", "compile_predict_velocity_mode", "compile_prefix", "preprocess_device", "num_steps"]})
sys.path.insert(0, str(pathlib.Path.home() / "accel-bench"))
from bench_lingbot_v2 import make_obs
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy
from lerobot.policies.factory import make_pre_post_processors
policy = LingbotVLAV2Policy.from_pretrained(str(dst))
policy.to("cuda").eval()
pre, _ = make_pre_post_processors(
    policy.config, pretrained_path=str(dst),
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
obs = make_obs(str(dst))
for i in range(3):
    batch = pre(obs)
    t0 = time.perf_counter()
    with torch.no_grad():
        a = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    print(f"iter{i} {(time.perf_counter() - t0) * 1e3:.1f}ms action={tuple(a.shape)} finite={bool(a.isfinite().all())}")
print("I6_OK")
PYEOF
  echo "$name exit=$?" >> "$SM/STATUS"
}

: > "$SM/STATUS"
echo "=== batch1: T0 T1 T2 T3 ===" >> "$SM/STATUS"
train_arm T0_control 4 &
train_arm T1_fused 5 --policy.optimizer_fused=true &
train_arm T2_gradckpt 6 --policy.gradient_checkpointing=true &
train_arm T3_moe0 7 --policy.moe_dense_max_tokens=0 &
wait

echo "=== batch2: T4 T5 I0 I1 ===" >> "$SM/STATUS"
train_arm T4_sdpa_efficient 4 --policy.sdpa_backend=EFFICIENT_ATTENTION &
train_arm T5_sdpa_cudnn 5 --policy.sdpa_backend=CUDNN_ATTENTION &
infer_arm I0_control 6 &
infer_arm I1_gpu_preprocess 7 --gpu-preprocess &
wait

echo "=== batch3: I2 I3 I5 I6 ===" >> "$SM/STATUS"
infer_arm I2_compile 4 --compile --compile-mode max-autotune-no-cudagraphs &
infer_arm I3_compile_prefix 5 --compile --compile-mode max-autotune-no-cudagraphs --compile-prefix --gpu-preprocess &
infer_arm I5_moe0 6 --moe-dense-max-tokens 0 &
bake_arm I6_baked_rollout_path 7 &
wait

echo "=== batch4: I4 ===" >> "$SM/STATUS"
infer_arm I4_numsteps5 4 --compile --compile-mode max-autotune-no-cudagraphs --compile-prefix --gpu-preprocess --num-steps 5 &
wait

echo "=== ALL DONE ===" >> "$SM/STATUS"
