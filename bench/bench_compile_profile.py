#!/usr/bin/env python3
"""Per-stage latency for the ACCELERATED lerobot port WITH torch.compile enabled.

Probe design that does NOT disturb dynamo:
  - get_image_features / embed_prefix are only called OUTSIDE the compiled
    predict_velocity, so plain CUDA-event wrappers are safe there.
  - core.forward is called both outside (prefix fill) and inside (denoise)
    the compiled region; the wrapper passes through untouched when
    torch.compiler.is_compiling() (constant-folded away by dynamo), and only
    times the eager prefix call.
  - the denoise loop calls model._compiled_predict_velocity from EAGER code;
    we pre-compile predict_velocity ourselves and install a timing wrapper as
    _compiled_predict_velocity, giving per-step denoise latency for free.
  - MoE kernel time is attributed from a torch.profiler trace by kernel name
    (_grouped_gemm_kernel / _moe_*); MoE only runs in the denoise expert
    stream, so a global name match is stage-clean.

Usage: python bench_compile_profile.py --ckpt <ckpt> [--iters 3]
"""

import argparse
import collections
import json
import os
import sys
import time

import torch

# handle_kv_cache specializes on layer_idx (36 layers) — the default recompile
# limit (8) can fall back to eager mid-run; raise it before any compile.
import torch._dynamo

torch._dynamo.config.recompile_limit = 64
try:
    torch._dynamo.config.accumulated_recompile_limit = 512
except AttributeError:
    pass

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--iters", type=int, default=3)
p.add_argument("--num-steps", type=int, default=10)
args = p.parse_args()

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_lingbot_v2 import make_obs


class CudaTimer:
    def __init__(self):
        self.pairs = []
        self._open = None
        self.calls = 0

    def start(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._open = e

    def stop(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.pairs.append((self._open, e))
        self._open = None
        self.calls += 1

    @property
    def elapsed_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.pairs)

    def reset(self):
        self.pairs.clear()
        self.calls = 0


class Timers(dict):
    def __missing__(self, key):
        self[key] = CudaTimer()
        return self[key]


policy = LingbotVLAV2Policy.from_pretrained(args.ckpt)
policy.config.num_steps = args.num_steps
policy.model._use_compile_predict_velocity = True
policy.to("cuda").eval()

preprocessor, _ = make_pre_post_processors(
    policy.config,
    pretrained_path=args.ckpt,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
batch = preprocessor(make_obs(args.ckpt))

fm = policy.model
core = fm.qwenvl_with_expert
T = Timers()

# vision (embed_prefix path only — never inside compiled code)
orig_gif = core.get_image_features


def gif_timed(*a, **kw):
    T["vision"].start()
    try:
        return orig_gif(*a, **kw)
    finally:
        T["vision"].stop()


core.get_image_features = gif_timed

# embed_prefix total (eager, once per inference)
orig_ep = fm.embed_prefix


def ep_timed(*a, **kw):
    T["embed_prefix"].start()
    try:
        return orig_ep(*a, **kw)
    finally:
        T["embed_prefix"].stop()


fm.embed_prefix = ep_timed

# NOTE: no wrapper on core.forward — any Python frame there is traced by
# dynamo inside predict_velocity and forces a per-layer graph break (measured:
# 22k kernels/infer, denoise 845ms vs 580ms unwrapped). prefix_fwd is derived
# by subtraction below instead.

# per-step denoise: pre-compile predict_velocity with the same options the
# model uses and install a timing wrapper as _compiled_predict_velocity
compiled_pv = torch.compile(
    fm.predict_velocity,
    fullgraph=False,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def pv_timed(*a, **kw):
    T["denoise_step"].start()
    try:
        return compiled_pv(*a, **kw)
    finally:
        T["denoise_step"].stop()


fm._compiled_predict_velocity = pv_timed

print("warming up (compile) ...", flush=True)
with torch.no_grad():
    for _ in range(3):
        policy.predict_action_chunk(batch)
torch.cuda.synchronize()
for t in T.values():
    t.reset()
print("compile done", flush=True)

# ---- clean totals (no profiler) ---------------------------------------------
with torch.no_grad():
    for _ in range(2):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    for t in T.values():
        t.reset()
    t0 = time.perf_counter()
    N_CLEAN = 10
    for _ in range(N_CLEAN):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    clean_total_ms = (time.perf_counter() - t0) / N_CLEAN * 1e3
    stage_ms = {k: t.elapsed_ms / N_CLEAN for k, t in T.items()}
    stage_calls = {k: t.calls / N_CLEAN for k, t in T.items()}

# ---- profiler pass for kernel-level attribution ------------------------------
from torch.profiler import ProfilerActivity, profile

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()

trace_path = os.environ.get("TRACE_OUT", "/tmp/compile_profile.json")
prof.export_chrome_trace(trace_path)

evs = json.load(open(trace_path))["traceEvents"]
kernels = [(e["name"], e["dur"]) for e in evs if e.get("ph") == "X" and e.get("cat") == "kernel"]

MOE_PAT = ("_moe_", "moe_", "grouped_gemm")
ATTN_PAT = ("flash_fwd", "flash_attn", "memory_efficient", "fmha", "scaled_dot_product")


def classify(kname):
    kl = kname.lower()
    if any(p in kl for p in MOE_PAT):
        return "moe"
    if any(p in kl for p in ATTN_PAT):
        return "attention"
    return "other"


n_inf = args.iters
out = {
    "iters_clean": N_CLEAN,
    "num_steps": args.num_steps,
    "clean_total_ms": clean_total_ms,
}
for k in stage_ms:
    out[f"{k}_ms"] = stage_ms[k]
    out[f"{k}_calls"] = stage_calls[k]
out["denoise_loop_ms"] = stage_ms.get("denoise_step", 0.0)
# prefix fill is not directly probeable without disturbing dynamo; derive it:
# total = embed_prefix + prefix_fwd + denoise_loop + postprocess(~0.7ms)
out["prefix_fwd_derived_ms"] = (
    clean_total_ms - stage_ms.get("embed_prefix", 0.0) - stage_ms.get("denoise_step", 0.0)
)
for cls in ("moe", "attention", "other"):
    out[f"kernels_{cls}_ms"] = sum(d for n, d in kernels if classify(n) == cls) / n_inf / 1e3
out["kernels_total_ms"] = sum(d for _, d in kernels) / n_inf / 1e3
out["kernel_count_per_infer"] = len(kernels) / n_inf

agg = collections.Counter()
for n, d in kernels:
    agg[n[:70]] += d
out["top_kernels_us_per_infer"] = {n: round(d / n_inf, 1) for n, d in agg.most_common(12)}

print("RESULT " + json.dumps(out, indent=2))
if os.environ.get("BENCH_OUT"):
    with open(os.environ["BENCH_OUT"], "w") as f:
        json.dump(out, f, indent=2)
