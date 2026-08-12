#!/usr/bin/env python3
"""Stage-level latency/throughput benchmark for the lingbot_vla_v2 policy.

Measures the full inference chain (preprocess -> vision -> prefix KV fill ->
per-step denoise -> postprocess) and the training step (fwd+bwd), with
cumulative sub-timers for attention / MoE / rope so each part's share of the
total is reported.

Sub-timers use CUDA event pairs recorded on the current stream (no per-call
host sync, so timing overhead does not distort the pipeline).

Usage:
  python bench_lingbot_v2.py infer --ckpt /path/to/ckpt [--iters 20] [--num-steps 10]
  python bench_lingbot_v2.py train --ckpt /path/to/ckpt [--batch 8] [--iters 10]
  python bench_lingbot_v2.py check-flex --ckpt /path/to/ckpt   # flex_cached vs sdpa parity
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
    LingbotVLAV2Policy,
    QwenvlWithExpertV2Model,
)
from lerobot.policies.lingbot_vla_v2.qwen2_action_expert import Qwen2TokenMoeBlock


class CudaTimer:
    """Accumulates GPU time via event pairs; elapsed resolved after a final sync."""

    def __init__(self, device="cuda"):
        self.pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.device = device
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


class WallTimer:
    def __init__(self):
        self.total = 0.0
        self.calls = 0
        self._t0 = None

    def start(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self.total += time.perf_counter() - self._t0
        self._t0 = None
        self.calls += 1

    @property
    def elapsed_ms(self):
        return self.total * 1e3


class Timers(dict):
    def __missing__(self, key):
        self[key] = CudaTimer()
        return self[key]


def install_probes(policy, T: Timers):
    """Monkeypatch stage boundaries with cumulative timers. Returns restore fn."""
    fm = policy.model
    core: QwenvlWithExpertV2Model = fm.qwenvl_with_expert
    patches = []

    def wrap(obj, name, key, bind=None):
        orig = getattr(obj, name)

        def wrapper(*a, **kw):
            T[key].start()
            try:
                return orig(*(bind or ()) + a, **kw)
            finally:
                T[key].stop()

        patches.append((obj, name, orig))
        setattr(obj, name, wrapper)

    # vision tower (patch embed + ViT blocks)
    wrap(core, "get_image_features", "vision")
    # joint dual-stream forward: split prefix-fill vs denoise by call index
    state = {"n": 0}
    orig_forward = core.forward

    def forward_probe(*a, **kw):
        # prefix fill: inputs_embeds=[prefix, None]; denoise: [None, suffix];
        # training: [prefix, suffix]
        ie = kw.get("inputs_embeds", a[5] if len(a) > 5 else None)
        if ie is not None and ie[0] is None:
            key = "fwd_denoise"
        elif ie is not None and ie[1] is None:
            key = "fwd_prefix"
        else:
            key = "fwd_joint"
        T[key].start()
        try:
            return orig_forward(*a, **kw)
        finally:
            T[key].stop()

    patches.append((core, "forward", orig_forward))
    core.forward = forward_probe

    # attention backend (instance attr set at init)
    attn = core.attention_interface

    def attn_probe(*a, **kw):
        T["attention"].start()
        try:
            return attn(*a, **kw)
        finally:
            T["attention"].stop()

    core.attention_interface = attn_probe
    patches.append((core, "attention_interface", attn))

    # flex_cached bypasses attention_interface
    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as mod

    if hasattr(mod, "flex_attention_with_block_mask"):
        orig_flex = mod.flex_attention_with_block_mask

        def flex_probe(*a, **kw):
            T["attention"].start()
            try:
                return orig_flex(*a, **kw)
            finally:
                T["attention"].stop()

        mod.flex_attention_with_block_mask = flex_probe
        patches.append((mod, "flex_attention_with_block_mask", orig_flex))

    # MoE blocks (training + inference eager/triton all funnel through forward)
    orig_moe = Qwen2TokenMoeBlock.forward

    def moe_probe(self, *a, **kw):
        T["moe"].start()
        try:
            return orig_moe(self, *a, **kw)
        finally:
            T["moe"].stop()

    Qwen2TokenMoeBlock.forward = moe_probe
    patches.append((Qwen2TokenMoeBlock, "forward", orig_moe))

    # mrope (per-layer rotary compute)
    wrap(core, "apply_mrope", "rope")
    # action unapply / postprocess
    wrap(policy, "_postprocess_actions", "postprocess")
    # suffix embedding (per denoise step)
    wrap(fm, "embed_suffix", "embed_suffix")

    def restore():
        for obj, name, orig in patches:
            setattr(obj, name, orig)

    return restore


def make_obs(ckpt, seed=0):
    """Synthetic observation matching the checkpoint's robot_config: source camera
    keys and state dim are read from the embedded policy_preprocessor.json so the
    same script works for 3-cam bi-arm and 1-cam SO101 checkpoints alike."""
    import os

    cfg = json.load(open(os.path.join(ckpt, "policy_preprocessor.json")))
    ft = {s["registry_name"]: s["config"] for s in cfg["steps"]}["lingbot_vla_v2_feature_transform"]
    rc = ft["robot_config"]
    img_keys = []
    for entry in rc.get("images", []):
        for info in entry.values():
            ok = info.get("origin_keys")
            if isinstance(ok, str):
                img_keys.append(ok)
    state_dim = 0
    for entry in rc.get("states", []):
        for info in entry.values():
            oks = info.get("origin_keys")
            if isinstance(oks, list):
                for d in oks:
                    for k, sl in d.items():
                        if k == "observation.state":
                            state_dim = max(state_dim, sl["end"])
    rng = np.random.default_rng(seed)
    img = rng.random((3, 480, 640)).astype(np.float32)
    obs = {k: torch.from_numpy(img.copy()) for k in img_keys}
    obs["observation.state"] = torch.from_numpy(np.zeros(state_dim, dtype=np.float32))
    obs["task"] = "stack the bowls"
    return obs


def load_policy(args):
    policy = LingbotVLAV2Policy.from_pretrained(args.ckpt)
    core = policy.model.qwenvl_with_expert
    if args.attn:
        policy.config.attention_implementation = args.attn
        core.config.attention_implementation = args.attn
        core.attention_interface = core.get_attention_interface()
    if args.vit_attn:
        policy.config.vit_attn_implementation = args.vit_attn
    if getattr(args, "attn_fp32", None) is not None:
        core.config.attention_fp32 = args.attn_fp32
    if getattr(args, "grad_ckpt", False):
        core.config.gradient_checkpointing = True
    if getattr(args, "moe_dense_max_tokens", None) is not None:
        # blocks read the threshold at construction; override live
        for m in policy.modules():
            if hasattr(m, "_dense_max_tokens"):
                m._dense_max_tokens = args.moe_dense_max_tokens
    policy.to("cuda").eval()
    return policy


def bench_infer(args):
    policy = load_policy(args)
    policy.config.num_steps = args.num_steps
    if getattr(args, "compile", False):
        policy.model._use_compile_predict_velocity = True
        # handle_kv_cache specializes on layer_idx (36 layers); the default
        # recompile limit (8) can fall back to eager mid-graph
        import torch._dynamo as _dynamo
        _dynamo.config.recompile_limit = 64
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    T = Timers()
    if not getattr(args, "compile", False):
        # probes wrap hot functions in a way dynamo recompiles on; skip them for
        # the compile experiment (we only need total latency there)
        install_probes(policy, T)

    lat_total, lat_pre = [], []
    for i in range(args.warmup + args.iters):
        obs = make_obs(args.ckpt, seed=i)
        t0 = time.perf_counter()
        batch = preprocessor(obs)
        t_pre = time.perf_counter() - t0
        with torch.no_grad():
            t0 = time.perf_counter()
            actions = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            lat_total.append(time.perf_counter() - t0)
        lat_pre.append(t_pre)
        if i < args.warmup:
            # reset timers after warmup
            for t in list(T.values()):
                t.pairs.clear()
                t.calls = 0
            lat_total.clear()
            lat_pre.clear()

    n = len(lat_total)
    total = sum(lat_total) / n * 1e3
    out = {
        "mode": "infer",
        "attn": policy.model.qwenvl_with_expert.config.attention_implementation,
        "attn_fp32": getattr(policy.model.qwenvl_with_expert.config, "attention_fp32", None),
        "num_steps": args.num_steps,
        "iters": n,
        "total_ms": total,
        "preprocess_ms": sum(lat_pre) / n * 1e3,
        "action_shape": list(actions.shape),
    }
    for key, t in T.items():
        out[f"{key}_ms"] = t.elapsed_ms / n
        out[f"{key}_calls"] = t.calls / n
    out["denoise_per_step_ms"] = out.get("fwd_denoise_ms", 0) / max(args.num_steps, 1)
    print(json.dumps(out, indent=2))
    return out


def bench_train(args):
    policy = load_policy(args)
    policy.train()
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    # one real preprocessed sample, then tile to batch B
    sample = preprocessor(make_obs(args.ckpt))
    B = args.batch
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            reps = [B] + [1] * (v.ndim - 1)
            batch[k] = v.repeat(*reps) if v.shape[0] == 1 else v
    chunk = policy.config.chunk_size
    adim = policy.config.max_action_dim
    torch.manual_seed(42)
    batch["action"] = torch.randn(B, chunk, adim, device="cuda", dtype=torch.bfloat16) * 0.1
    # deterministic flow-matching noise/time so before/after losses are comparable
    g = torch.Generator(device="cuda").manual_seed(1234)
    batch["noise"] = torch.randn(B, chunk, adim, device="cuda", dtype=torch.bfloat16, generator=g)
    batch["time"] = torch.rand(B, device="cuda", generator=g).to(torch.bfloat16) * 0.999 + 0.001
    # joint_mask: mark only the first 14 dims valid (SO101-style)
    jm = torch.zeros(B, chunk, adim, device="cuda", dtype=torch.bfloat16)
    jm[:, :, :14] = 1.0
    batch["joint_mask"] = jm
    batch = {k: (v.cuda() if torch.is_tensor(v) and not v.is_cuda else v) for k, v in batch.items()}

    T = Timers()
    install_probes(policy, T)
    wall = WallTimer()
    for i in range(args.warmup + args.iters):
        if i == args.warmup:
            for t in list(T.values()):
                t.pairs.clear()
                t.calls = 0
            torch.cuda.reset_peak_memory_stats()
        wall.start()
        try:
            loss, loss_dict = policy.forward(batch)
            loss.backward()
        except torch.OutOfMemoryError:
            torch.cuda.synchronize()
            summ = torch.cuda.memory_summary()
            print("OOM at batch", B)
            print("\n".join(summ.splitlines()[:12]))
            raise
        policy.zero_grad(set_to_none=True)
        wall.stop()
    n = args.iters
    out = {
        "mode": "train",
        "attn": policy.model.qwenvl_with_expert.config.attention_implementation,
        "attn_fp32": getattr(policy.model.qwenvl_with_expert.config, "attention_fp32", None),
        "grad_ckpt": getattr(policy.model.qwenvl_with_expert.config, "gradient_checkpointing", None),
        "batch": B,
        "iters": n,
        "step_ms": wall.elapsed_ms / n,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "loss": float(loss_dict["loss"]),
    }
    for key, t in T.items():
        out[f"{key}_ms"] = t.elapsed_ms / n
        out[f"{key}_calls"] = t.calls / n
    print(json.dumps(out, indent=2))
    return out


def check_flex(args):
    """Parity check: sdpa+fp32 attention (original behavior) as reference, compared
    against sdpa+bf16 and flex_cached+bf16 — same weights, same noise."""
    policy = load_policy(args)
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    batch = preprocessor(make_obs(args.ckpt))
    noise = torch.randn(1, policy.config.n_action_steps, policy.config.max_action_dim,
                        device="cuda", dtype=next(policy.parameters()).dtype)

    def run(attn, fp32):
        core = policy.model.qwenvl_with_expert
        core.config.attention_implementation = attn
        core.config.attention_fp32 = fp32
        core.attention_interface = core.get_attention_interface()
        with torch.no_grad():
            return policy.predict_action_chunk(batch, noise=noise.clone())

    ref = run("sdpa", True)
    out = {"reference": "sdpa+fp32", "action_abs_max": ref.abs().max().item()}
    for attn, fp32 in [("sdpa", False), ("eager", True), ("flex_cached", False)]:
        try:
            a = run(attn, fp32)
            diff = (ref - a).abs().max().item()
            out[f"{attn}+{'fp32' if fp32 else 'bf16'}"] = {
                "max_abs_diff": diff, "rel": diff / max(ref.abs().max().item(), 1e-9)}
        except Exception as exc:
            out[f"{attn}+{'fp32' if fp32 else 'bf16'}"] = f"FAILED: {type(exc).__name__}: {exc}"
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["infer", "train", "check-flex"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attn", default=None, help="override attention_implementation")
    p.add_argument("--vit-attn", default=None)
    p.add_argument("--attn-fp32", action=argparse.BooleanOptionalAction, default=None,
                   help="force fp32 attention upcast (parity path) on/off")
    p.add_argument("--grad-ckpt", action="store_true", help="enable gradient checkpointing")
    p.add_argument("--moe-dense-max-tokens", type=int, default=None,
                   help="override moe_dense_max_tokens on all MoE blocks (0 = disable dense path)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile predict_velocity (inductor, cudagraphs off)")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    {"infer": bench_infer, "train": bench_train, "check-flex": check_flex}[args.mode](args)


if __name__ == "__main__":
    main()
