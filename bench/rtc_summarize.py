#!/usr/bin/env python3
"""Aggregate rtc_bench runs into the deliverable table.

Usage: python3 summarize.py  (run on the 4090 box, reads /root/d4090_rtc/logs)
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/root/lerobot-port/bench")
sys.path.insert(0, "/root/lerobot-port/src")

import numpy as np  # noqa: E402

LOGS = "/root/d4090_rtc/logs"
TICK_MS = 1000.0 / 30


def load(arm, run):
    ticks = [json.loads(l) for l in open(f"{LOGS}/{arm}_run{run}_ticks.jsonl")]
    infs = [json.loads(l) for l in open(f"{LOGS}/{arm}_run{run}_inferences.jsonl")]
    return ticks, infs


def reactions(arm, ticks, infs, leg_len):
    jumps = [leg_len * i for i in range(1, 7)]
    obs_tick_of = {c["chunk_id"]: c["obs_tick"] for c in infs}
    id_key = "chunk_id" if arm == "sync" else "epoch"
    out = []
    for jt in jumps:
        post = {cid for cid, ot in obs_tick_of.items() if ot >= jt}
        for rec in ticks:
            if rec["tick"] >= jt and rec.get("action") is not None and rec.get(id_key) in post:
                out.append((rec["tick"] - jt) * TICK_MS)
                break
    return out


def boundary_stats(arm, ticks):
    id_key = "chunk_id" if arm == "sync" else "epoch"
    prev, boundary, internal = None, [], []
    for rec in ticks:
        a = rec.get("action")
        if a is not None and prev is not None and prev.get("action") is not None:
            d = float(np.linalg.norm(np.array(a) - np.array(prev["action"])))
            if rec.get(id_key) != prev.get(id_key):
                boundary.append(d)
            elif not rec.get("freeze"):
                internal.append(d)
        prev = rec
    return boundary, internal


def summarize(arm, run, leg_len):
    ticks, infs = load(arm, run)
    r = reactions(arm, ticks, infs, leg_len)
    b, i = boundary_stats(arm, ticks)
    walls = [c["wall_ms"] for c in infs][3:]  # skip warmup-ish chunks
    gpus = [c["gpu_ms"] for c in infs][3:]
    row = {
        "run": f"{arm}_run{run}",
        "chunk_wall_ms": float(np.median(walls)),
        "chunk_gpu_ms": float(np.median(gpus)),
        "reactions_ms": [round(x, 1) for x in r],
        "reaction_median_ms": float(np.median(r)),
        "boundary_jump_median": float(np.median(b)),
        "internal_step_median": float(np.median(i)),
    }
    if arm == "sync":
        starts = [c["start_tick"] for c in infs]
        row["freeze_ticks_per_cycle"] = float(np.median([c["end_tick"] - c["start_tick"] + 1 for c in infs]))
        row["period_ticks"] = float(np.median(np.diff(starts)))
        row["freeze_frac"] = sum(1 for t in ticks if t.get("freeze")) / len(ticks)
        # analytic phase-uniform reaction: with jump phase p ~ U[0,P) relative to the
        # freeze-start, reaction(p) = P + F - p ticks -> median = F + P/2.
        P, F = row["period_ticks"], row["freeze_ticks_per_cycle"]
        row["reaction_analytic_phaseuniform_ms"] = (F + P / 2) * TICK_MS if P and F else None
    else:
        ends = [c["end_tick"] for c in infs]
        row["inference_interval_ticks"] = float(np.median(np.diff(ends)))
        if arm == "rtc":
            ds = [c["delay_used"] for c in infs][3:]
            row["delay_used_median"] = float(np.median(ds))
            row["delay_used_all"] = ds
            row["real_delay_median"] = float(np.median([c["real_delay"] for c in infs][3:]))
    return row


def main():
    rows = []
    for arm, run, leg in [("sync", 2, 426), ("sync", 3, 440), ("async", 2, 426), ("rtc", 2, 426)]:
        if os.path.exists(f"{LOGS}/{arm}_run{run}_ticks.jsonl"):
            try:
                rows.append(summarize(arm, run, leg))
            except Exception as e:
                rows.append({"run": f"{arm}_run{run}", "error": str(e)})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
