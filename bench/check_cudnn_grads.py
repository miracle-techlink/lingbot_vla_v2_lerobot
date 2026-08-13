#!/usr/bin/env python3
"""Single-step gradient comparison: auto SDPA backend vs forced cuDNN backend.

Same weights, same synthetic batch (built via bench_lingbot_v2 helpers), one
fwd+bwd each, then compare all parameter gradients element-wise.
"""
import sys

import torch

sys.path.insert(0, "/home/liuyue/accel-bench")
from bench_lingbot_v2 import make_obs  # noqa: E402

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import (  # noqa: E402
    LingbotVLAV2Policy,
)

CKPT = "/home/liuyue/lvla_scratch/robotwin-6b-lerobot"
B = 2


def build_batch(policy):
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    sample = preprocessor(make_obs(CKPT))
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            reps = [B] + [1] * (v.ndim - 1)
            batch[k] = v.repeat(*reps) if v.shape[0] == 1 else v
    chunk, adim = policy.config.chunk_size, policy.config.max_action_dim
    torch.manual_seed(42)
    batch["action"] = torch.randn(B, chunk, adim, device="cuda", dtype=torch.bfloat16) * 0.1
    g = torch.Generator(device="cuda").manual_seed(1234)
    batch["noise"] = torch.randn(B, chunk, adim, device="cuda", dtype=torch.bfloat16, generator=g)
    batch["time"] = torch.rand(B, device="cuda", generator=g).to(torch.bfloat16) * 0.999 + 0.001
    jm = torch.zeros(B, chunk, adim, device="cuda", dtype=torch.bfloat16)
    jm[:, :, :14] = 1.0
    batch["joint_mask"] = jm
    return {k: (v.cuda() if torch.is_tensor(v) and not v.is_cuda else v) for k, v in batch.items()}


def run(policy, batch, backend):
    core = policy.model.qwenvl_with_expert
    core.config.sdpa_backend = backend
    core.attention_interface = core.get_attention_interface()
    policy.zero_grad(set_to_none=True)
    loss, _ = policy.forward(batch)
    loss.backward()
    grads = {}
    for n, p in policy.named_parameters():
        if p.grad is not None:
            grads[n] = p.grad.detach().float().clone()
    policy.zero_grad(set_to_none=True)
    return float(loss), grads


def main():
    policy = LingbotVLAV2Policy.from_pretrained(CKPT)
    policy.to("cuda").train()
    batch = build_batch(policy)

    loss_auto, g_auto = run(policy, batch, None)
    loss_cudnn, g_cudnn = run(policy, batch, "CUDNN_ATTENTION")

    print(f"loss auto={loss_auto:.8f} cudnn={loss_cudnn:.8f} rel={abs(loss_auto-loss_cudnn)/abs(loss_auto):.2e}")
    nan_cudnn = [n for n, g in g_cudnn.items() if torch.isnan(g).any()]
    nan_auto = [n for n, g in g_auto.items() if torch.isnan(g).any()]
    print(f"NaN grads: auto={len(nan_auto)} cudnn={len(nan_cudnn)}")
    if nan_cudnn:
        print("first NaN grad (cudnn):", nan_cudnn[0])
    worst = []
    for n in g_auto:
        a, b = g_auto[n], g_cudnn[n]
        denom = a.abs().max().clamp_min(1e-12)
        worst.append(((a - b).abs().max() / denom, n))
    worst.sort(reverse=True)
    print("top-5 grad rel max-diff:")
    for d, n in worst[:5]:
        print(f"  {float(d):.3e}  {n}")
    med = sorted(float(d) for d, _ in worst)[len(worst) // 2]
    print(f"median grad rel max-diff: {med:.3e} over {len(worst)} tensors")


if __name__ == "__main__":
    main()
