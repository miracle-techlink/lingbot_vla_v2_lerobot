# LingBot-VLA 2.0 with `lerobot-train` / `lerobot-rollout`: Entry Points & Switch Guide

> Branch `pr3967-combined`, 2026-08-14. Audience: lerobot community users who want to
> train LingBot-VLA 2.0 (Qwen3-VL-4B + 36-layer sparse-MoE action expert) and deploy it
> with the stock community entry points — no custom harnesses.
> Every flag below is verified against the source and launch-tested on 8×A100-80GB.

> **TL;DR — final configs (sdpa is kept everywhere; it was never voted out):**
> **Training** = sdpa (auto backend) + bf16 + dense dual-GEMM MoE (default threshold 512)
> + fused AdamW (`--policy.optimizer_fused=true`) + no grad-ckpt; B=3 single GPU, B=2 per rank DDP.
> **Inference** = sdpa + bf16 + `compile_predict_velocity` (max-autotune-no-cudagraphs)
> + `compile_prefix` + `preprocess_device=cuda` + `num_steps=10`, baked once into the ckpt.
> Rejected: forcing `sdpa_backend` (CUDNN → backward NaN, EFFICIENT → no kernel),
> `eager` for training (real A/B: +2.2% slower, +2.9 GB, B=4 OOM), `attention_fp32`.

---

## 1. The two entry points

| Task | Entry | Notes |
|---|---|---|
| Training | `lerobot-train` | multi-GPU via `accelerate launch --num_processes=N $(which lerobot-train)` |
| On-robot inference | `lerobot-rollout` | policy loaded from `--policy.path` (a saved checkpoint dir) |

One config object (`LingbotVLAV2Config`) drives both. Training flags are passed on the
`lerobot-train` CLI as `--policy.<flag>=<value>`; inference-time flags are best **baked
into the checkpoint's `config.json` once** (§4), so rollout stays a single `--policy.path`.

## 2. Training quickstart

### 2.1 Single GPU (80 GB)

```bash
lerobot-train \
  --dataset.repo_id=<user>/<dataset> \
  --dataset.root=<local dataset path> \
  --dataset.streaming=false \
  --policy.type=lingbot_vla_v2 \
  --policy.pretrained_path=<converted lingbot-vla-v2 ckpt> \
  --policy.robot_config_path=<robot_config.yaml> \
  --policy.norm_stats_path=<norm_stats.json> \
  --policy.dtype=bfloat16 \
  --policy.optimizer_fused=true \
  --policy.push_to_hub=false \
  --batch_size=3 \
  --steps=30000 \
  --log_freq=50 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --output_dir=~/output_lerobot_train/<run_name> \
  --job_name=<run_name> \
  --wandb.enable=false
```

### 2.2 Multi-GPU (community-standard accelerate)

Same command, three changes:

```bash
accelerate launch --num_processes=4 $(which lerobot-train) \
  ... same flags ... \
  --batch_size=2            # PER-RANK semantics!
```

**`batch_size` is per-rank**: effective global batch = `num_processes × batch_size`
(printed at startup). DDP all-reduce gradient buckets add ~11 GB per rank, so the
per-rank ceiling is **B=2** (B=3 OOMs on 80 GB).

### 2.3 Mandatory overrides when the dataset ≠ the checkpoint's native robot

`--policy.robot_config_path` / `--policy.norm_stats_path` map your dataset's raw keys
(camera names, state/action slices) onto the checkpoint's canonical layout
(`arm.position` 14-dim slot; `camera_top` / `camera_wrist_left` / `camera_wrist_right`).
They override what is embedded in the checkpoint. **If your dataset's camera names differ
from the checkpoint's native ones and you omit these, training dies in a
FeatureTransform assertion.** Missing canonical camera slots are masked automatically.

## 3. Switch reference (training)

All passed as `--policy.<name>=<value>` on `lerobot-train`.

| Flag | Default | What it does | Guidance |
|---|---|---|---|
| `optimizer_fused` | `false` | single-kernel fused AdamW | **turn on**: −5.4% step time, loss-identical |
| `attention_implementation` | `sdpa` | attention kernel path (`sdpa`/`eager`/`fa2`/`flex`/`flex_cached`) | keep `sdpa`; `eager` A/B'd: +2.2% slower, +2.9 GB, B=4 OOM (§6) |
| `attention_fp32` | `false` | fp32 upcast inside attention | off; +8–11% slower for parity-level numerics |
| `gradient_checkpointing` | `false` | recompute activations | **don't** at B≤4 (+38% slower); only if OOM forces it |
| `moe_dense_max_tokens` | `512` | dense dual-GEMM MoE below this token count; `0` = per-expert loop | keep default; `0` only for ablation |
| `sdpa_backend` | `null` (auto) | force a specific SDPA kernel | **NEVER set**: `CUDNN_ATTENTION` → backward NaN (torch 2.8, bf16+bool mask+GQA); `EFFICIENT_ATTENTION` → no available kernel. Launch-verified. |
| `num_steps` | `10` | flow-matching denoise steps | locked at 10 |

## 4. Inference: bake once, then just `--policy.path`

Training checkpoints don't carry inference optimizations. Bake them once:

```python
import json, pathlib
p = pathlib.Path("<run>/checkpoints/last/pretrained_model/config.json")
c = json.loads(p.read_text())
c.update({
    "compile_predict_velocity": True,
    "compile_predict_velocity_mode": "max-autotune-no-cudagraphs",
    "compile_prefix": True,
    "preprocess_device": "cuda",
    "num_steps": 10,
})
p.write_text(json.dumps(c, indent=2))
```

This exact baked path is launch-tested end-to-end (`from_pretrained` →
`predict_action_chunk`, steady-state 258 ms on the 3-camera 512 px workload after a
one-time ~44 s inductor compile).

| Baked key | Effect |
|---|---|
| `compile_predict_velocity` (+ `_mode`) | torch.compile the 10 denoise steps; the main inference speedup |
| `compile_prefix` | also compile vision+prefix fill (requires the above) |
| `preprocess_device: "cuda"` | GPU image preprocessing (−62% preprocess time); training ignores this and stays on CPU |
| `num_steps` | denoise steps; smaller = linearly faster, lower action quality |

## 5. Rollout quickstart

```bash
lerobot-rollout \
  --policy.path=<run>/checkpoints/last/pretrained_model \
  --robot.type=<robot, e.g. so101_follower / rebot_b601_follower> \
  --robot.port=<port> \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/v4l/by-id/<id>, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {...} }" \
  --strategy.type=base \
  --inference.type=sync \
  --task="<exactly the training task string>" \
  --fps=30 --duration=60 \
  --display_data=false --play_sounds=false
```

Add `--dataset.repo_id=... --dataset.single_task=... --dataset.push_to_hub=false` to
record the rollout. Camera keys must match the robot_config `origin_keys` used in
training **exactly**. First run on a new machine recompiles once (minutes).

## 6. Performance anchors (A100-80GB, contended host, 2026-08)

| Scenario | Number |
|---|---|
| Inference e2e, full baked stack (3-cam 512 px) | 436 ms (official eager 1705 ms; official compile 1113 ms) |
| Inference e2e, 1-cam SO101 | ~300 ms |
| Train 1 GPU, sdpa+bf16 B=3 | 1.51 s/step · 1.99 samples/s · 72.0 GB |
| Train 4 GPU, B=2/rank | 1.00 s/step · 8.0 samples/s → 30k steps ≈ 8.3 h |

**Attention A/B verdict (300-step real-dataset run, 2026-08-14).** A synthetic-benchmark
arm had suggested `eager`+bf16 was faster (−6.9%) and lighter (−10 GB); that was a
tiled-identical-sample artifact. In the real `lerobot-train` run: loss curves are
statistically identical (max |Δ| ≈ 0.012 over 300 steps, no NaN), but `eager` is
**+2.2% slower** (0.999 vs 0.978 s/update), **+2.9 GB** heavier (68.5 vs 65.7 GB),
and B=4 **OOMs** (79.3/80 GB) where the bench said it would fit. `sdpa`+bf16 stays
the recommended training default; don't override `attention_implementation`.

## 7. Hard-won don'ts

1. **Never `--policy.sdpa_backend=...`** (any value). Auto backend is the only safe choice.
2. **Don't enable `gradient_checkpointing`** at B≤4 — you pay 38% for memory you don't need.
3. **Don't forget `robot_config_path`/`norm_stats_path`** when dataset cameras ≠ checkpoint cameras.
4. **Don't change `num_steps` casually** — 10 is the validated quality/latency point.
5. First on-robot rollout: `--duration=10` sanity run before long tasks.

---

*Chinese companion docs (same repo, `docs/accel/`): full command handbook
(`lingbot_vla_train_rollout_commands.md`), launch-verification report
(`lingbot_vla_switch_table.pdf`), reBot arm field guide
(`lingbot_vla_rebot_guide.pdf`).*
