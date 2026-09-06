# Accelerate lingbot_vla_v2 inference & training (sdpa/bf16 attention, dense small-T MoE, torch.compile, GPU preprocessing)

## Motivation

The upstream lingbot_vla_v2 policy runs its dual-stream (VLM + action expert)
attention in eager fp32 and dispatches MoE experts through routed
argsort/gather/scatter machinery even at the tiny token counts of
flow-matching inference (T ≈ chunk+1 = 51). End-to-end action-chunk inference
measured **1039.5 ms** on an A100 — too slow for real-time control — and
training steps carried avoidable host-device syncs from the MoE monitoring
metrics. This branch is a purely additive acceleration pass: every optimization
is gated behind a config flag, numerics are validated against the original
paths, and a LIBERO open-loop fidelity bench guards against behavioral drift.

## Key results (A100)

| Metric | Before | After |
|---|---|---|
| Inference, end-to-end action chunk | 1039.5 ms | ~300 ms |
| Training step, B=4 | 2762.6 ms | 1704.6 ms |
| LIBERO open-loop action drift (all suites) | — | all ≤ 0.76% (threshold 2%) |
| Dense MoE vs eager, fp32 parity | — | max abs diff 1.4e-6 |

## Commit groups (in review order)

**Core attention + denoise-loop perf**
- `4d692d25` perf: sdpa+bf16 attention (`attention_implementation="sdpa"`,
  `attention_fp32` parity switch), denoise-loop invariant hoisting
  (masks / position ids / rotary embeddings computed once per chunk instead of
  per step), sync-free MoE (`moe_metrics_interval` — metrics .item() syncs once
  every N steps instead of every step).

**torch.compile of the hot paths**
- `b0544e9b` feat: `compile_predict_velocity` flag — compile the per-step
  denoise (suffix) forward.
- `7eb22d83` feat: `compile_predict_velocity_mode` — `"default"` |
  `"max-autotune-no-cudagraphs"` inductor mode.
- `569929b8` feat: `compile_prefix` — also compile vision tower + prefix KV
  fill (runs once per chunk); requires `compile_predict_velocity=True`.

**MoE**
- `9de3cc41` perf: dense two-GEMM pure-torch MoE for small token counts
  (`moe_dense_max_tokens`, default 512): every expert computed with two plain
  matmuls, routing weights folded into the down GEMM — no argsort/gather/
  scatter, static shapes, no torch.compile graph breaks. fp32 parity vs the
  eager path: max abs diff 1.4e-6.

**Data path**
- `21ef4187` feat: GPU image-preprocess fast path (`preprocess_device`, one
  batched HF image-processor call with on-device outputs) + episode-level
  tokenizer cache (task string is constant within an episode; bounded cache
  keyed by full prompt).

**Docs**
- `e6f9f6d7` docs: new-embodiment adaptation guide with a single-arm example.

**Benchmarks / validation tooling**
- `a94b3c8c` bench: stage-level latency/throughput scripts
  (`bench_lingbot_v2.py`, `bench_compile_profile.py`, `bench_deploy_native.py`,
  `bench/a100.py` + runners).
- `87baf779` bench: LIBERO open-loop fidelity bench
  (`bench_libero_fidelity.py`) + compile-mode flag.
- `2b4c8fe4` bench: unattended ablation runners (locked dense+bf16+compile
  core).

**Hygiene**
- `e62d64d0` style: ruff format + lint fixes (no behavior change; dense-MoE
  path re-verified numerically against a manual per-expert reference,
  max abs diff 1.2e-10).

## Default-behavior changes (disclosed honestly)

Most optimizations are opt-in, but three defaults intentionally change
relative to the original PR, because they are validated numerically and are
where most of the win comes from:

| Config | Original PR | This branch | Revert to original |
|---|---|---|---|
| `attention_implementation` / `vit_attn_implementation` | `"eager"` | `"sdpa"` (+ bf16 tensor cores; `attention_fp32=False`) | set `"eager"` and `attention_fp32=True` |
| `moe_dense_max_tokens` | — (routed dispatch only) | `512` (dense path **on** by default; fp32 parity 1.4e-6) | set `0` |
| `moe_metrics_interval` | 1 (sync every step) | `50` | set `1` |

Strictly opt-in (default = original behavior): `preprocess_device=None`,
`compile_predict_velocity=False`, `compile_predict_velocity_mode="default"`,
`compile_prefix=False`.

## Testing

- `python bench/bench_lingbot_v2.py --ckpt <ckpt>` — stage-level latency /
  throughput (vision / prefix / per-step denoise / e2e); `--compile-mode`
  selects the inductor mode.
- `python bench/bench_libero_fidelity.py ...` — LIBERO open-loop fidelity:
  replays dataset actions through the policy and reports per-suite action
  drift; all suites ≤ 0.76% against a 2% threshold with the full
  dense+bf16+compile stack enabled.
- `python bench/check_gpu_preprocess.py ...` — CPU vs GPU preprocessing
  parity + timing for the `preprocess_device` fast path.
- `bench/run_ablation2_master.sh` — unattended ablation matrix (dense on/off
  × bf16 × compile modes), used to produce the numbers above.
- Dense MoE algebra re-checked against a manual per-expert reference on CPU
  (max abs diff 1.2e-10).

## Risks & rollback

- Every optimization has a config switch; the table above shows how to restore
  the original PR behavior exactly (`attention_implementation="eager"`,
  `attention_fp32=True`, `moe_dense_max_tokens=0`, `moe_metrics_interval=1`,
  all compile/preprocess flags off).
- Numerical differences are limited to floating-point reassociation (bf16
  tensor-core attention, dense-MoE reduction order); LIBERO drift bench
  bounds the behavioral impact at ≤ 0.76%.
- torch.compile paths require fixed shapes across calls and pay a one-time
  compile cost (minutes) on first use; triton kernels are optional — the
  in-tree grouped-GEMM and the dense path both fall back to pure torch.
- The GPU preprocessing fast path is inference-only; training (augmentation)
  and depth-align paths always stay on CPU regardless of `preprocess_device`.
