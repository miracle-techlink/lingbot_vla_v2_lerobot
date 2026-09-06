#!/usr/bin/env python
"""拼 rollout 帧条图 + 画 30k 完整 loss 曲线 (学术风格, 对齐昨天 loss_curve_rtw30k.png)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

BASE = "/home/nvidia/reports/robotwin_eval_report"

# -- 1) rollout strip: 4 frames side by side with time labels --
frames = [(0, "t=0s (step 0)"), (33, "t≈16s (step 165)"), (66, "t≈33s (step 330)"), (95, "t≈47s (step 475)")]
imgs = [Image.open(f"{BASE}/figs/rollout_f{p}.png") for p, _ in frames]
w, h = imgs[0].size
pad, label_h = 8, 34
strip = Image.new("RGB", (w * len(imgs) + pad * (len(imgs) + 1), h + label_h + pad), "white")
draw = ImageDraw.Draw(strip)
for i, (im, (_, label)) in enumerate(zip(imgs, frames)):
    x = pad + i * (w + pad)
    strip.paste(im, (x, label_h + pad))
    draw.text((x + w // 2 - 60, 8), label, fill="black")
strip.save(f"{BASE}/figs/rollout_strip.png", dpi=(150, 150))
print("strip:", strip.size)

# -- 2) loss curve 1k→30k --
recs = json.load(open(f"{BASE}/data/rtw_loss_full_30k.json"))
steps = [s for s, _ in recs]
loss = [l for _, l in recs]

plt.rcParams.update({"font.size": 15, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
ax.plot(steps, loss, lw=0.7, color="#1f5fa6", alpha=0.35, label="raw (every 20 steps)")
win = 25
if len(loss) > win:
    sm = [sum(loss[max(0, i - win) : i + 1]) / len(loss[max(0, i - win) : i + 1]) for i in range(len(loss))]
    ax.plot(steps, sm, lw=1.8, color="#1f5fa6", label=f"moving avg ({win * 20} steps)")
ax.axvspan(4200, 6000, color="#b3502d", alpha=0.08)
ax.annotate("crash-restart window\n(watchdog resume from ckpt 006000)", xy=(5100, 0.30), xytext=(8000, 0.315),
            fontsize=12, color="#b3502d", arrowprops=dict(arrowstyle="->", color="#b3502d", lw=1.0))
ax.annotate(f"final loss {loss[-1]:.3f}", xy=(steps[-1], loss[-1]), xytext=(steps[-1] - 6500, loss[-1] + 0.028),
            fontsize=12, arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_xlabel("training step")
ax.set_ylabel("training loss (L1 flow-matching)")
ax.set_xlim(0, 30500)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(frameon=False, fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{BASE}/figs/loss_curve_30k.png")
print("loss curve saved:", len(recs), "records, final", loss[-1])
