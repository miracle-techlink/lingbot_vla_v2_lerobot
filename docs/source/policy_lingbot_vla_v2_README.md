# LingBot-VLA 2.0

`lingbot_vla_v2` is the LeRobot policy wrapper for LingBot-VLA 2.0. It combines a
Qwen3-VL vision-language backbone with a sparse MoE Qwen2 action expert and flow-matching
continuous action generation over the canonical 55-D robot state/action space.

Use this policy through the standard LeRobot interfaces: `lerobot-train`,
`make_policy_config`, `make_policy`, `from_pretrained`, and `predict_action_chunk` /
`select_action`.

## Install

Install LeRobot with the optional LingBot-VLA v2 dependencies:

```bash
pip install -e ".[training,lingbot-v2]"
```

The default config expects a Qwen3-VL processor/tokenizer and a LingBot-VLA v2 checkpoint:

```text
Qwen/Qwen3-VL-4B-Instruct
robbyant/lingbot-vla-v2-6b
```

Use local paths with `--policy.processor_path`, `--policy.tokenizer_path`, or
`--policy.path` when running offline.

## Train

First convert the raw upstream checkpoint into the self-contained LeRobot format:

```bash
python -m lerobot.policies.lingbot_vla_v2.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-vla-v2-6b-lerobot \
  --robot-config-path <robot_config.yaml> \
  --norm-stats-path <norm_stats.json>
```

Then use the normal LeRobot training CLI, initializing from the converted checkpoint with
`--policy.path`. Do not also pass `--policy.type`; LeRobot infers `lingbot_vla_v2` from the
checkpoint's `config.json`.

```bash
lerobot-train \
  --dataset.repo_id=<repo_id> \
  --dataset.root=<dataset_root> \
  --policy.path=<converted-lingbot-vla-v2-6b> \
  --policy.robot_config_path=<robot_config.yaml> \
  --policy.norm_stats_path=<norm_stats.json> \
  --policy.processor_path=<qwen3_vl_processor_or_model_path> \
  --policy.tokenizer_path=<qwen3_vl_processor_or_model_path> \
  --policy.image_max_pixels=262144 \
  --policy.image_min_pixels=131072 \
  --policy.device=cuda \
  --batch_size=1 \
  --steps=5000 \
  --save_freq=2500 \
  --output_dir=outputs/train/lingbot_vla_v2
```

For an offline smoke test, add `--policy.push_to_hub=false`, `--save_checkpoint=false`,
and `--num_workers=0`, then use local paths for the processor, tokenizer, and checkpoint.

Required data assets:

- `robot_config_path`: maps dataset state/action/image keys into LingBot-VLA v2 canonical slots.
- `norm_stats_path`: stores per-slot normalization stats used by the LingBot feature transform.
- `processor_path` / `tokenizer_path`: Qwen3-VL processor/tokenizer path or Hub id. Use local
  paths when the training node cannot reach the Hugging Face Hub.

The robot config maps dataset keys into the canonical LingBot slots. The norm-stats JSON is
used by the LingBot feature transform, so the saved LeRobot processor pipeline does not use
the generic LeRobot normalizer/unnormalizer steps.

## Adapting to a New Embodiment

Fine-tuning on a robot the checkpoint was not converted for only requires two new assets —
a robot-config YAML and a norm-stats JSON — passed as `--policy.robot_config_path` /
`--policy.norm_stats_path`. Explicit paths take precedence over the assets embedded in the
checkpoint (a warning is logged when they differ), and checkpoints saved during fine-tuning
embed the new assets so they remain self-contained.

Single-arm example (7-DoF: 6 arm joints + gripper, absolute joint angles, `front` + `wrist`
cameras) — the filled dims are packed from position 0 of each canonical slot, unfilled slots
are zero-padded and masked out of the loss:

```yaml
# rebot.yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state: { start: 0, end: 6 }
  - observation.state.effector.position:
      origin_keys:
        - observation.state: { start: 6, end: 7 }
actions:
  - action.arm.position:
      origin_keys:
        - action: { start: 0, end: 6 }
      subtract_state: false # absolute actions; true only for state-relative deltas
  - action.effector.position:
      origin_keys:
        - action: { start: 6, end: 7 }
      subtract_state: false
images: # unmapped canonical views (camera_wrist_right) are zero-filled
  - observation.images.camera_top:
      origin_keys: observation.images.front
  - observation.images.camera_wrist_left:
      origin_keys: observation.images.wrist
norm_stats: rebot_norm_stats.json
```

The norm-stats JSON holds per-slot `mean`/`std` over the filled dims (the default
`canonical_norm_type` is `meanstd`), e.g. `{"norm_stats": {"action.arm.position": {"mean":
[...6], "std": [...6]}, ...}}` — derivable from a LeRobot dataset's `meta/stats.json`.

Deploy the fine-tuned checkpoint on the robot with `lerobot-rollout`
(`--strategy.type=episodic` to record evaluation episodes); camera names must match the
`origin_keys` above. `select_action` returns actions in the robot's raw action space.

See `docs/source/lingbot_vla_v2.mdx` for the full walkthrough.

## Resume

Resume from a saved LeRobot checkpoint by passing the checkpoint's `train_config.json`:

```bash
lerobot-train \
  --resume=true \
  --config_path=outputs/train/lingbot_vla_v2/checkpoints/005000/pretrained_model/train_config.json \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/lingbot_vla_v2_resume
```

## Load A Policy

```python
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

policy = LingbotVLAV2Policy.from_pretrained(
    "outputs/train/lingbot_vla_v2/checkpoints/005000/pretrained_model"
)
policy.to("cuda").eval()
```

For batched LeRobot observations, use `select_action`. For open-loop action chunks, use
`predict_action_chunk`; it returns a `(batch, chunk_size, action_dim)` tensor after the
policy postprocessing path has mapped canonical actions back to raw dataset action keys.

## Tests

Run the lightweight registration and config tests:

```bash
pytest -q tests/policies/lingbot_vla_v2/test_lingbot_vla_v2.py
```

Run the feature-transform tests when a local Qwen3-VL processor is available:

```bash
export LINGBOT_VLA_V2_QWEN3VL=/path/to/Qwen3-VL-4B-Instruct
pytest -q tests/policies/lingbot_vla_v2/test_feature_transform.py
```

Run the optional Triton grouped-MoE parity test on a CUDA machine with Triton installed:

```bash
pytest -q tests/policies/lingbot_vla_v2/test_triton_moe.py
```

## Citation

If you use this policy, please cite the upstream LingBot-VLA 2.0 project and LeRobot.
The upstream project is available at:

```text
https://github.com/Robbyant/lingbot-vla-v2
```

## Implementation Notes

The Qwen3-VL backbone adaptation and sparse-MoE action expert are vendored from the upstream
LingBot-VLA 2.0 implementation and Hugging Face Transformers, with Apache-2.0 license headers
retained. FlashAttention is optional. The default attention implementation is `sdpa` in the
model dtype (bf16); `eager`, `fa2`, `flex`, and `flex_cached` can be selected through the policy
config, and `attention_fp32=true` restores the original fp32-attention parity path.

For MoE inference, the fused expert path tries the optional upstream Triton kernel first, then
the in-tree Triton grouped-GEMM backend, and finally the grouped-by-expert eager fallback. The
training path uses the eager fallback for autograd stability; the eager fallback groups routes with
a single argsort (one host sync per layer) instead of per-expert nonzero scans.
