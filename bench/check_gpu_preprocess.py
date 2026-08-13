#!/usr/bin/env python3
"""Equivalence check for the GPU preprocessing fast path (C4/M2 acceptance).

Runs the same observation through the lingbot_vla_v2 preprocessor twice —
default per-camera CPU path vs batched on-GPU path — and reports:
  1. element-wise diffs of every preprocessed tensor (images/lang/state/masks)
  2. end-to-end action parity (fixed noise, same weights): rel L2 + max abs
  3. preprocess wall time of both paths

Usage:
  python check_gpu_preprocess.py --ckpt /path/to/ckpt [--iters 20]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from bench_lingbot_v2 import make_obs
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy


def build_preprocessor(policy, ckpt, device):
    policy.config.preprocess_device = device
    pre, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return pre


def diff_batches(a, b):
    out = {}
    for k in a:
        ta, tb = a[k], b.get(k)
        if not torch.is_tensor(ta):
            continue
        if tb is None or ta.shape != tb.shape or ta.dtype != tb.dtype:
            out[k] = (
                f"MISMATCH {tuple(ta.shape)}/{ta.dtype} vs "
                f"{getattr(tb, 'shape', None)}/{getattr(tb, 'dtype', None)}"
            )
            continue
        da = ta.float().abs().max().item() if ta.numel() else 0.0
        d = (ta.float() - tb.float()).abs().max().item() if ta.numel() else 0.0
        out[k] = {"max_abs_diff": d, "ref_abs_max": da, "rel": d / max(da, 1e-9)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    policy = LingbotVLAV2Policy.from_pretrained(args.ckpt)
    policy.to("cuda").eval()

    pre_cpu = build_preprocessor(policy, args.ckpt, None)
    pre_gpu = build_preprocessor(policy, args.ckpt, "cuda")

    obs = make_obs(args.ckpt, seed=0)
    batch_cpu = pre_cpu(obs)
    batch_gpu = pre_gpu(obs)
    report = {"tensor_diffs": diff_batches(batch_cpu, batch_gpu)}

    # action parity with fixed noise
    noise = torch.randn(
        1,
        policy.config.n_action_steps,
        policy.config.max_action_dim,
        device="cuda",
        dtype=next(policy.parameters()).dtype,
    )
    with torch.no_grad():
        a_cpu = policy.predict_action_chunk(batch_cpu, noise=noise.clone())
        a_cpu2 = policy.predict_action_chunk(batch_cpu, noise=noise.clone())
        a_gpu = policy.predict_action_chunk(batch_gpu, noise=noise.clone())
    d = (a_cpu - a_gpu).abs().max().item()
    d_self = (a_cpu - a_cpu2).abs().max().item()
    report["action_parity"] = {
        "max_abs_diff": d,
        "rel": d / max(a_cpu.abs().max().item(), 1e-9),
        "self_rerun_max_abs_diff": d_self,
        "self_rerun_rel": d_self / max(a_cpu.abs().max().item(), 1e-9),
    }

    # preprocess timing both paths
    for name, pre in [("cpu", pre_cpu), ("gpu", pre_gpu)]:
        times = []
        for i in range(args.warmup + args.iters):
            o = make_obs(args.ckpt, seed=i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pre(o)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            if i < args.warmup:
                times.clear()
        report[f"preprocess_ms_{name}"] = sum(times) / len(times) * 1e3

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
