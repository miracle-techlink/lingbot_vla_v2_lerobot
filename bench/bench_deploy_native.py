#!/usr/bin/env python3
"""Deploy-faithful benchmark for upstream robbyant/lingbot-vla-v2.

Reproduces the README path: deploy.lingbot_vla_v2_policy.LingbotVLAv2Server
(bf16, robotwin robot config, real 6B checkpoint), timing each stage:
  preprocess (_prepare_model_input: resize + FeatureTransform + image proc + tokenizer)
  model      (sample_actions_batch -> sample_actions GPU time)
  unapply    (_unapply_batched_actions: action un-normalize)
and, in eager mode, model-internal probes:
  vision / fwd_prefix / fwd_denoise / attention / moe / rope / embed_suffix

Usage (on A100, conda env lvla-native):
  CUDA_VISIBLE_DEVICES=4 python bench_deploy_native.py [--compile] [--iters 20]
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("QWEN3VL_PATH", os.path.expanduser("~/lvla_scratch/Qwen3-VL-4B-Instruct"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = "/home/liuyue/lingbot-vla-v2-upstream"
sys.path.insert(0, REPO)
os.chdir(REPO)  # robot configs / norm stats are resolved relative to repo root

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--compile", action="store_true", help="README fast path: --use_compile")
p.add_argument("--iters", type=int, default=20)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--num-steps", type=int, default=0, help="override denoise steps (0 = keep ckpt default 10)")
args = p.parse_args()

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

MODEL = os.path.expanduser("~/lvla_scratch/ttt_policy_deploy/a/b/model")

server = LingbotVLAv2Server(
    MODEL,
    use_length=50,
    chunk_ret=True,  # every infer() call runs a real forward
    use_bf16=True,
    use_compile=args.compile,
)
server.reset(robo_name="robotwin")
if args.num_steps:
    server.vla.config.num_steps = args.num_steps
    server.vla.model.config.num_steps = args.num_steps


# ---------------- stage timers (host wall around the three deploy stages) ----
class Wall:
    def __init__(self):
        self.t = 0.0
        self.n = 0

    def reset(self):
        self.t = 0.0
        self.n = 0


pre_t, mdl_t, un_t = Wall(), Wall(), Wall()

orig_prep = server._prepare_model_input


def prep(obs):
    t0 = time.perf_counter()
    r = orig_prep(obs)
    pre_t.t += time.perf_counter() - t0
    pre_t.n += 1
    return r


server._prepare_model_input = prep

orig_sab = server.vla.sample_actions_batch


def sab(*a, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = orig_sab(*a, **kw)
    torch.cuda.synchronize()
    mdl_t.t += time.perf_counter() - t0
    mdl_t.n += 1
    return r


server.vla.sample_actions_batch = sab

orig_un = server._unapply_batched_actions


def un(*a, **kw):
    t0 = time.perf_counter()
    r = orig_un(*a, **kw)
    un_t.t += time.perf_counter() - t0
    un_t.n += 1
    return r


server._unapply_batched_actions = un


# ---------------- model-internal probes (eager only; they break dynamo) ------
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


class Timers(dict):
    def __missing__(self, key):
        self[key] = CudaTimer()
        return self[key]


T = Timers()
if not args.compile:
    fm = server.vla.model
    core = fm.qwenvl_with_expert
    patches = []

    def wrap(obj, name, key):
        orig = getattr(obj, name)

        def wrapper(*a, **kw):
            T[key].start()
            try:
                return orig(*a, **kw)
            finally:
                T[key].stop()

        patches.append((obj, name, orig))
        setattr(obj, name, wrapper)

    wrap(core, "get_image_features", "vision")
    orig_forward = core.forward

    def forward_probe(*a, **kw):
        ie = kw.get("inputs_embeds")
        if ie is None and len(a) > 4:
            ie = a[4]
        key = "fwd_denoise" if (ie is not None and ie[0] is None) else "fwd_prefix"
        T[key].start()
        try:
            return orig_forward(*a, **kw)
        finally:
            T[key].stop()

    patches.append((core, "forward", orig_forward))
    core.forward = forward_probe

    attn = core.attention_interface

    def attn_probe(*a, **kw):
        T["attention"].start()
        try:
            return attn(*a, **kw)
        finally:
            T["attention"].stop()

    core.attention_interface = attn_probe
    patches.append((core, "attention_interface", attn))

    if hasattr(core, "apply_mrope"):
        wrap(core, "apply_mrope", "rope")
    if hasattr(fm, "embed_suffix"):
        wrap(fm, "embed_suffix", "embed_suffix")

    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2TokenMoeBlock

    orig_moe = Qwen2TokenMoeBlock.forward

    def moe_probe(self, *a, **kw):
        T["moe"].start()
        try:
            return orig_moe(self, *a, **kw)
        finally:
            T["moe"].stop()

    Qwen2TokenMoeBlock.forward = moe_probe
    patches.append((Qwen2TokenMoeBlock, "forward", orig_moe))

# ---------------- observation (robotwin: 3 cams + 14-dim state + prompt) ----
rng = np.random.default_rng(0)


def make_obs():
    return {
        "observation.state": rng.random(14).astype(np.float32),
        "observation.images.cam_high": (rng.random((480, 640, 3)) * 255).astype(np.uint8),
        "observation.images.cam_left_wrist": (rng.random((480, 640, 3)) * 255).astype(np.uint8),
        "observation.images.cam_right_wrist": (rng.random((480, 640, 3)) * 255).astype(np.uint8),
        "task": "pick up the block",
    }


infer_times = []
for i in range(args.warmup + args.iters):
    obs = make_obs()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = server.infer(obs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if i == args.warmup - 1:
        pre_t.reset()
        mdl_t.reset()
        un_t.reset()
        for t in T.values():
            t.pairs.clear()
            t.calls = 0
    if i >= args.warmup:
        infer_times.append(dt)

n = len(infer_times)
res = {
    "side": "native-deploy",
    "compile": args.compile,
    "num_steps": server.vla.config.num_steps,
    "iters": n,
    "infer_total_ms": sum(infer_times) / n * 1e3,
    "preprocess_ms": pre_t.t / max(pre_t.n, 1) * 1e3,
    "model_ms": mdl_t.t / max(mdl_t.n, 1) * 1e3,
    "unapply_ms": un_t.t / max(un_t.n, 1) * 1e3,
}
for key, t in T.items():
    res[f"{key}_ms"] = t.elapsed_ms / n
    res[f"{key}_calls"] = t.calls / n
res["denoise_per_step_ms"] = res.get("fwd_denoise_ms", 0) / max(server.vla.config.num_steps, 1)
print("RESULT " + json.dumps(res, indent=2), flush=True)
if os.environ.get("BENCH_OUT"):
    with open(os.environ["BENCH_OUT"], "w") as f:
        json.dump(res, f, indent=2)
