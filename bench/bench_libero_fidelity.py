#!/usr/bin/env python3
"""Open-loop fidelity + latency benchmark on real LIBERO episode frames.

Feeds real camera frames from LIBERO demo HDF5 files through the policy's
normal observation path and measures, per configuration arm:
  - latency: model time per action chunk (CUDA-event precise) + preprocess wall
  - fidelity: action drift vs a saved full-precision reference
    (rel L2, max abs, cosine) computed with the SAME fixed noise seed

The reference is produced once (--save-ref ref.npz with the reference flags,
e.g. --attn eager --attn-fp32 --moe-dense-max-tokens 0) and every arm reloads
it (--ref ref.npz), so arms never pay the fp32 reference cost.

LIBERO -> SO101 input mapping (fidelity testbed, not task success):
  agentview_rgb -> the ckpt's single front camera key (preprocessor resizes)
  joint_states[:6] -> observation.state (SO101 is 6-dim)
  hdf5 filename  -> task string (underscores -> spaces)

Usage:
  python bench_libero_fidelity.py --ckpt CKPT --hdf5 file1.hdf5 [file2.hdf5 ...] \
      --save-ref ref.npz [reference flags]
  python bench_libero_fidelity.py --ckpt CKPT --hdf5 ... --ref ref.npz [arm flags]
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
)


def read_ckpt_spec(ckpt):
    """Camera keys and state dim expected by the checkpoint's preprocessor."""
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
    return img_keys, state_dim


def load_libero_frames(hdf5_paths, frames_per_demo, max_demos):
    """Yield (image HWC uint8, state6 float32, task) samples spread across demos."""
    import h5py

    samples = []
    for path in hdf5_paths:
        task = path.rsplit("/", 1)[-1].replace("_demo.hdf5", "").replace("_", " ")
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())[:max_demos]
            for dk in demos:
                demo = f["data"][dk]
                imgs = demo["obs"]["agentview_rgb"]  # [T, H, W, 3] uint8
                # 7 joints + 2 gripper = 9 raw dims; truncated/padded to the ckpt's
                # state dim by the caller (fixed mapping across arms)
                joints = np.concatenate(
                    [demo["obs"]["joint_states"][:], demo["obs"]["gripper_states"][:]], axis=1
                ).astype(np.float32)  # [T, 9]
                T = imgs.shape[0]
                idxs = np.linspace(0, T - 1, frames_per_demo).astype(int)
                for i in idxs:
                    samples.append((imgs[i].copy(), joints[i], task))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hdf5", nargs="+", required=True)
    p.add_argument("--frames-per-demo", type=int, default=8)
    p.add_argument("--max-demos", type=int, default=2)
    p.add_argument("--iters", type=int, default=3, help="timed repeats per frame")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ref", default=None, help="load reference actions npz")
    p.add_argument("--save-ref", default=None, help="save this run as reference npz")
    # arm knobs (same surface as bench_lingbot_v2)
    p.add_argument("--attn", default=None)
    p.add_argument("--attn-fp32", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--moe-dense-max-tokens", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument(
        "--compile-prefix", action="store_true", help="also compile embed_prefix + prefix KV fill (C3)"
    )
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="cast the whole model to this dtype after load",
    )
    p.add_argument(
        "--gpu-preprocess",
        action="store_true",
        help="run image preprocessing on GPU (batched single processor call)",
    )
    p.add_argument("--noise-seed", type=int, default=1234)
    args = p.parse_args()

    policy = LingbotVLAV2Policy.from_pretrained(args.ckpt)
    if args.dtype:
        policy.model.to(getattr(torch, args.dtype))
    core = policy.model.qwenvl_with_expert
    if args.attn:
        policy.config.attention_implementation = args.attn
        core.config.attention_implementation = args.attn
        core.attention_interface = core.get_attention_interface()
    if args.attn_fp32 is not None:
        core.config.attention_fp32 = args.attn_fp32
    if args.moe_dense_max_tokens is not None:
        for m in policy.modules():
            if hasattr(m, "_dense_max_tokens"):
                m._dense_max_tokens = args.moe_dense_max_tokens
    if args.compile:
        policy.model._use_compile_predict_velocity = True
        policy.model._compile_predict_velocity_mode = args.compile_mode
        if args.compile_prefix:
            policy.model._use_compile_prefix = True
        import torch._dynamo as _dynamo

        _dynamo.config.recompile_limit = 64
    policy.config.num_steps = args.num_steps
    if args.gpu_preprocess:
        policy.config.preprocess_device = "cuda"
    policy.to("cuda").eval()

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    img_keys, state_dim = read_ckpt_spec(args.ckpt)
    samples = load_libero_frames(args.hdf5, args.frames_per_demo, args.max_demos)
    print(f"loaded {len(samples)} frames from {len(args.hdf5)} hdf5 files", flush=True)

    ref = np.load(args.ref)["actions"] if args.ref else None
    actions_out = []
    lat_model, lat_pre = [], []

    g = torch.Generator(device="cuda").manual_seed(args.noise_seed)
    for si, (img, state, task) in enumerate(samples):
        obs = {k: torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0 for k in img_keys}
        st = np.zeros(state_dim, dtype=np.float32)
        st[: min(state_dim, state.shape[0])] = state[: min(state_dim, state.shape[0])]
        obs["observation.state"] = torch.from_numpy(st)
        obs["task"] = task
        t0 = time.perf_counter()
        batch = preprocessor(obs)
        t_pre = time.perf_counter() - t0

        n = args.warmup + args.iters if si == 0 else args.iters
        lat = []
        for it in range(n):
            noise = torch.randn(
                1,
                policy.config.n_action_steps,
                policy.config.max_action_dim,
                device="cuda",
                dtype=next(policy.parameters()).dtype,
                generator=g,
            )
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                act = policy.predict_action_chunk(batch, noise=noise.clone())
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
        keep = lat[args.warmup :] if si == 0 else lat
        lat_model.extend(keep)
        lat_pre.append(t_pre)
        actions_out.append(act.float().cpu().numpy()[0])
        if si % 4 == 0:
            print(f"frame {si}/{len(samples)} model={np.mean(lat) * 1e3:.1f}ms", flush=True)

    actions = np.stack(actions_out)
    out = {
        "mode": "libero_fidelity",
        "attn": core.config.attention_implementation,
        "attn_fp32": getattr(core.config, "attention_fp32", None),
        "dense": getattr(args, "moe_dense_max_tokens", None),
        "compile": bool(args.compile),
        "compile_mode": args.compile_mode if args.compile else None,
        "num_steps": args.num_steps,
        "frames": len(samples),
        "model_ms_mean": float(np.mean(lat_model) * 1e3),
        "model_ms_p50": float(np.percentile(lat_model, 50) * 1e3),
        "preprocess_ms_mean": float(np.mean(lat_pre) * 1e3),
    }
    if args.save_ref:
        np.savez_compressed(args.save_ref, actions=actions)
        out["saved_ref"] = args.save_ref
    if ref is not None:
        assert ref.shape == actions.shape, f"ref {ref.shape} vs arm {actions.shape}"
        diff = actions - ref
        ref_norm = np.linalg.norm(ref.reshape(len(ref), -1), axis=1) + 1e-9
        arm_norm = np.linalg.norm(actions.reshape(len(actions), -1), axis=1)
        out["drift_rel_l2_mean"] = float(
            np.mean(np.linalg.norm(diff.reshape(len(diff), -1), axis=1) / ref_norm)
        )
        out["drift_max_abs"] = float(np.abs(diff).max())
        cos = np.sum(actions.reshape(len(actions), -1) * ref.reshape(len(ref), -1), axis=1) / (
            arm_norm * ref_norm
        )
        out["drift_cosine_mean"] = float(np.mean(cos))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
