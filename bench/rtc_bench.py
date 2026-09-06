#!/usr/bin/env python3
"""sync / async / RTC scheduling benchmark for LingBot-VLA 2.0 (4x4090 box, GPU0).

Simulated 30fps control loop driven by REAL rebot_shakehands observations.
The world = sequential dataset frames; at fixed ticks the frame source jumps to
a distant segment (observation discontinuity). "Equivalent reaction latency" =
ticks from the jump until the executed action stream first carries an action
whose chunk was inferred from a post-jump observation (chunk provenance is
tracked exactly, per chunk).

Arms:
  sync   execute 25 actions, then blocking inference (freeze = hold last action).
  async  background inference thread; queue threshold 30; new chunk REPLACES the
         queue from index 0 (no guidance, no delay skip).
  rtc    async + lerobot RTCProcessor.denoise_step guidance in normalized action
         space; leftover prefix truncated to execution_horizon=12; adaptive
         inference_delay = ceil(max_recent_latency / tick); merge skips
         real_delay = ceil(wall_latency / tick) actions of the new chunk.

Modes:
  sanity    bitwise identity check: RTC path (delay=0, no leftover) == sync path.
  overhead  guided vs unguided single-chunk latency (analytic RTC cost).
  run       one arm of the simulated loop:  --arm {sync,async,rtc}

All artifacts under /root/d4090_rtc/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import types

import numpy as np
import torch

sys.path.insert(0, "/root/lerobot-port/bench")
sys.path.insert(0, "/root/lerobot-port/src")

from bench_lingbot_v2 import load_policy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.rtc.action_queue import ActionQueue  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.latency_tracker import LatencyTracker  # noqa: E402
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402

CKPT = "/root/ckpts/shakehands-lerobot"
DS = "/root/datasets/rebot_shakehands"
OUT = "/root/d4090_rtc"

FPS = 30
TICK = 1.0 / FPS
# 7 legs / 6 jumps. LEG_LEN is env-overridable: the sync arm's cycle locks at
# ~35.08 ticks and 421 is nearly resonant with it (421 = 12*35.08 + 0.04), which
# put all 6 run1 jumps at the same cycle phase. 426 gives a +5 tick phase sweep.
LEG_LEN = int(os.environ.get("RTC_LEG_LEN", "421"))
N_LEGS = int(os.environ.get("RTC_N_LEGS", "7"))
TOTAL_TICKS = LEG_LEN * N_LEGS
JUMP_TICKS = [LEG_LEN * i for i in range(1, N_LEGS)]
# (episode, seg_start_frame) per leg: distant segments, cross-episode jumps
LEGS = [(10, 120), (40, 120), (70, 120), (30, 120), (55, 120), (5, 120), (85, 120)]
SEG_FRAMES = LEG_LEN + 20          # preload margin

USE_LEN = 25                       # sync arm: actions executed per chunk
QUEUE_TRIGGER = 30                 # async/rtc: infer when qsize() <= this
EXEC_HORIZON = 12                  # rtc: leftover prefix length / guidance horizon end
TASK = "shake hands"


# ---------------------------------------------------------------------------
# World: real dataset frames, segmented, with jump schedule
# ---------------------------------------------------------------------------

def decode_segment(path, ts0, n):
    import av

    c = av.open(path)
    s = c.streams.video[0]
    c.seek(int(ts0 / s.time_base), stream=s, any_frame=False, backward=True)
    frames = []
    for f in c.decode(s):
        if f.time is None or f.time < ts0 - 1e-6:
            continue
        frames.append(f.to_ndarray(format="rgb24"))
        if len(frames) == n:
            break
    c.close()
    if len(frames) < n:
        raise RuntimeError(f"short decode {path} ts={ts0}: got {len(frames)}/{n}")
    return np.stack(frames)  # (n, 480, 640, 3) uint8


class World:
    """tick -> (episode, local_frame) -> raw observation tensors."""

    def __init__(self):
        import pandas as pd

        ep_meta = pd.read_parquet(f"{DS}/meta/episodes/chunk-000/file-000.parquet")
        data = pd.read_parquet(f"{DS}/data/chunk-000/file-000.parquet")
        self.segs = {}
        for ep, start in LEGS:
            row = ep_meta[ep_meta.episode_index == ep].iloc[0]
            i0 = int(row["dataset_from_index"]) + start
            states = np.stack(data["observation.state"].iloc[i0 : i0 + SEG_FRAMES].to_numpy()).astype(np.float32)
            entry = {"state": states}
            for cam in ("front", "wrist"):
                key = f"observation.images.{cam}"
                fidx = int(row[f"videos/{key}/file_index"])
                ts0 = float(row[f"videos/{key}/from_timestamp"]) + start / FPS
                path = f"{DS}/videos/{key}/chunk-000/file-{fidx:03d}.mp4"
                t0 = time.perf_counter()
                entry[cam] = decode_segment(path, ts0, SEG_FRAMES)
                print(f"[world] ep{ep} {cam}: decoded {SEG_FRAMES} frames in {time.perf_counter()-t0:.1f}s", flush=True)
            self.segs[ep] = entry
        self.t0 = None

    def start_clock(self):
        self.t0 = time.perf_counter()

    def tick_now(self):
        return int((time.perf_counter() - self.t0) / TICK)

    def frame_of_tick(self, tick):
        leg = min(tick // LEG_LEN, N_LEGS - 1)
        ep, start = LEGS[leg]
        return ep, start + (tick - leg * LEG_LEN)

    def obs_at_tick(self, tick):
        leg = min(tick // LEG_LEN, N_LEGS - 1)
        ep, start = LEGS[leg]
        i = tick - leg * LEG_LEN
        seg = self.segs[ep]
        obs = {
            "observation.images.front": torch.from_numpy(seg["front"][i].copy()).permute(2, 0, 1).float().div_(255.0),
            "observation.images.wrist": torch.from_numpy(seg["wrist"][i].copy()).permute(2, 0, 1).float().div_(255.0),
            "observation.state": torch.from_numpy(seg["state"][i].copy()),
            "task": TASK,
        }
        return obs


# ---------------------------------------------------------------------------
# Policy / model helpers
# ---------------------------------------------------------------------------

def load():
    args = types.SimpleNamespace(
        ckpt=CKPT, dtype=None, attn=None, vit_attn=None, attn_fp32=None,
        sdpa_backend=None, grad_ckpt=False, moe_dense_max_tokens=None,
    )
    policy = load_policy(args)
    policy.config.num_steps = 10
    policy.model._use_compile_predict_velocity = True
    policy.model._compile_predict_velocity_mode = "max-autotune-no-cudagraphs"
    policy.model._use_compile_prefix = True
    # Freeze params for ALL arms. This is what an inference deployment does, and for
    # the RTC guided path it is numerically exact, not an approximation:
    # RTCProcessor.denoise_step sets x_t.requires_grad_(True) AFTER the denoiser
    # forward, so no edge from x_t into the network graph exists and
    # autograd.grad(x1_t, x_t, err) == err regardless of whether params require grad.
    # With params unfrozen, autograd additionally executes a dead full-network
    # backward per denoise step whose saved activations OOM a 24GB 4090 (measured:
    # 23.5GB allocated, OOM on the first guided step's grad-mode forward).
    for p in policy.parameters():
        p.requires_grad_(False)
    import torch._dynamo as _dynamo

    _dynamo.config.recompile_limit = 64
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, preprocessor


def make_noise(policy, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dtype = next(policy.parameters()).dtype
    return torch.randn(
        1, policy.config.n_action_steps, policy.config.max_action_dim,
        generator=g, device="cuda", dtype=dtype,
    )


def rtc_sample(model, rtc_processor, inputs, noise, prev_left_over, inference_delay, execution_horizon):
    """Replica of FlowMatchingV2.sample_actions with the RTC guidance hook.

    - prefix fill: compiled prefix fn, under no_grad (identical to sync path).
    - denoise loop: compiled predict_velocity wrapped by RTCProcessor.denoise_step.
      Under grad (guided) steps the loop-invariant `_denoise_cache` is NOT shared
      across steps: cached tensors created in step k's graph would be referenced by
      step k+1's graph and autograd.grad(retain_graph=False) frees step graphs per
      call -> cross-step reuse would crash. Local cache per step instead.
    Bitwise note: with prev_left_over=None under no_grad this replicates
    sample_actions exactly (same accumulated bf16 time schedule, same dt tensor,
    same fresh per-call denoise cache).
    """
    images, img_masks, lang_tokens, lang_masks, state, grid = inputs
    bsize = state.shape[0]
    device, dtype = state.device, state.dtype

    prefix_fn = getattr(model, "_compiled_prefix", None)
    pv = getattr(model, "_compiled_predict_velocity", None)
    assert prefix_fn is not None and pv is not None, "warm up sample_actions first"

    with torch.no_grad():
        prefix_pad_masks, prefix_position_ids, past_key_values = prefix_fn(
            images, img_masks, lang_tokens, lang_masks, grid
        )

    dt = torch.tensor(-1.0 / model.config.num_steps, dtype=dtype, device=device)
    x_t = noise
    time_v = torch.tensor(1.0, dtype=dtype, device=device)
    time_values = []
    for _ in range(model.config.num_steps):
        time_values.append(time_v)
        time_v = time_v + dt

    guided = prev_left_over is not None
    denoise_cache = {}  # used only on the unguided path (mirrors sample_actions)
    for step_time in time_values:
        expanded_time = step_time.expand(bsize)

        def partial(input_x_t, et=expanded_time):
            return pv(
                state,
                prefix_pad_masks,
                past_key_values,
                input_x_t,
                et,
                prefix_position_ids=prefix_position_ids,
                _denoise_cache=None if guided else denoise_cache,
            )

        if guided:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_left_over,
                inference_delay=inference_delay,
                time=step_time,
                original_denoise_step_partial=partial,
                execution_horizon=execution_horizon,
            )
            v_t = v_t.to(x_t.dtype)  # guidance math upcasts to f32; keep x_t bf16
        else:
            with torch.no_grad():
                v_t = partial(x_t)
        x_t = x_t + dt * v_t
    return x_t


def normalize_prev(prev, target_steps):
    """lerobot rollout/inference/rtc.py _normalize_prev_actions_length (verbatim semantics)."""
    steps = prev.shape[0]
    if steps == target_steps:
        return prev
    if steps > target_steps:
        return prev[:target_steps]
    padded = torch.zeros((target_steps, prev.shape[1]), dtype=prev.dtype, device=prev.device)
    padded[:steps] = prev
    return padded


class TaggedQueue(ActionQueue):
    """ActionQueue replace-merge (RTC-mode semantics) + epoch (chunk_id) tagging.

    get()/merge_replace() hold the same lock, so the epoch a popped action belongs
    to is exact (no boundary mis-tag).
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.epoch = -1

    def get(self):
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None, self.epoch
            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone(), self.epoch

    def merge_replace(self, original_actions, processed_actions, real_delay, chunk_id):
        with self.lock:
            d = max(0, min(real_delay, len(original_actions), len(processed_actions)))
            self.original_queue = original_actions[d:].clone()
            self.queue = processed_actions[d:].clone()
            self.last_index = 0
            self.epoch = chunk_id


class JsonlLog:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)

    def write(self, rec):
        self.f.write(json.dumps(rec) + "\n")

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Inference (shared by all arms): preprocess -> sample -> postprocess, timed
# ---------------------------------------------------------------------------

def run_one_inference(policy, preprocessor, obs, noise, rtc_processor=None, prev=None, delay=0):
    """Returns (actions_norm (1,50,55), processed (1,50,7), timing dict)."""
    t0 = time.perf_counter()
    batch = preprocessor(obs)
    t_pre = time.perf_counter()
    inputs = policy._extract_model_inputs(batch)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    if rtc_processor is not None and prev is not None:
        actions = rtc_sample(policy.model, rtc_processor, inputs, noise, prev, delay, EXEC_HORIZON)
    else:
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state, grid = inputs
            actions = policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, image_grid_thw=grid,
            )
    e1.record()
    t_post0 = time.perf_counter()
    processed = policy._postprocess_actions(actions, batch)
    wall = time.perf_counter() - t0
    torch.cuda.synchronize()
    return actions, processed, {
        "pre_ms": (t_pre - t0) * 1e3,
        "gpu_ms": e0.elapsed_time(e1),
        "post_ms": (time.perf_counter() - t_post0) * 1e3,
        "wall_ms": wall * 1e3,
    }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_sync(world, policy, preprocessor, log_dir, run_id):
    tick_log = JsonlLog(f"{log_dir}/sync_run{run_id}_ticks.jsonl")
    inf_log = JsonlLog(f"{log_dir}/sync_run{run_id}_inferences.jsonl")
    chunk_id = -1
    chunk = None
    ptr = 0
    last_a = None
    state = "FREEZE"
    tick = 0
    world.start_clock()
    while tick < TOTAL_TICKS:
        target = world.t0 + tick * TICK
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        if state == "FREEZE":
            k = tick
            obs = world.obs_at_tick(k)
            cid = chunk_id + 1
            _, processed, tm = run_one_inference(policy, preprocessor, obs, make_noise(policy, 1000 + cid))
            end = world.tick_now()
            inf_log.write({
                "chunk_id": cid, "start_tick": k, "end_tick": end, "obs_tick": k,
                **{k2: round(v, 2) for k2, v in tm.items()},
            })
            chunk = processed[0]
            chunk_id = cid
            ptr = 0
            for ft in range(k, min(end + 1, TOTAL_TICKS)):
                tick_log.write({
                    "tick": ft, "chunk_id": chunk_id - 1, "freeze": True,
                    "action": last_a if last_a is not None else None,
                })
            tick = end + 1
            state = "EXECUTE"
            continue
        a = chunk[ptr].tolist()
        ptr += 1
        last_a = a
        tick_log.write({"tick": tick, "chunk_id": chunk_id, "freeze": False, "action": a})
        tick += 1
        if ptr == USE_LEN:
            state = "FREEZE"
    tick_log.close()
    inf_log.close()


def run_async(world, policy, preprocessor, log_dir, run_id, use_rtc):
    name = "rtc" if use_rtc else "async"
    tick_log = JsonlLog(f"{log_dir}/{name}_run{run_id}_ticks.jsonl")
    inf_log = JsonlLog(f"{log_dir}/{name}_run{run_id}_inferences.jsonl")
    cfg = RTCConfig(enabled=True, execution_horizon=EXEC_HORIZON)
    queue = TaggedQueue(cfg)
    rtc_processor = RTCProcessor(cfg) if use_rtc else None
    # Windowed p95 for the RTC inference_delay. lerobot's LatencyTracker.max() is a
    # monotone max over ALL history since reset, so a single transient spike pins
    # delay forever (observed: early grad-path warmup spikes -> delay stuck at 17 >
    # horizon 12, degenerating the LINEAR schedule into a hard clamp). p95 over the
    # last 20 walls tracks the steady state instead.
    from collections import deque

    recent_walls = deque(maxlen=20)
    stop = threading.Event()
    err = []

    def loop():
        cid = 0
        while not stop.is_set():
            if queue.qsize() > QUEUE_TRIGGER:
                time.sleep(0.005)
                continue
            try:
                start_tick = world.tick_now()
                idx_before = queue.get_action_index()
                prev = None
                delay = 0
                if use_rtc:
                    prev = queue.get_left_over()
                    if recent_walls:
                        delay = math.ceil(float(np.quantile(np.asarray(recent_walls), 0.95)) / TICK)
                    if prev is not None:
                        prev = normalize_prev(prev, EXEC_HORIZON)
                obs = world.obs_at_tick(start_tick)
                actions, processed, tm = run_one_inference(
                    policy, preprocessor, obs, make_noise(policy, 1000 + cid),
                    rtc_processor=rtc_processor, prev=prev, delay=delay,
                )
                wall_s = tm["wall_ms"] / 1e3
                recent_walls.append(wall_s)
                real_delay = math.ceil(wall_s / TICK) if use_rtc else 0
                queue.merge_replace(actions[0], processed[0], real_delay, cid)
                inf_log.write({
                    "chunk_id": cid, "start_tick": start_tick, "end_tick": world.tick_now(),
                    "obs_tick": start_tick, "delay_used": delay, "real_delay": real_delay,
                    "leftover_len": int(prev.shape[0]) if prev is not None else 0,
                    "idx_before": idx_before,
                    **{k2: round(v, 2) for k2, v in tm.items()},
                })
                cid += 1
            except Exception as e:  # noqa: BLE001
                import traceback

                err.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                stop.set()
                return

    world.start_clock()
    th = threading.Thread(target=loop, daemon=True, name="infer")
    th.start()
    for tick in range(TOTAL_TICKS):
        target = world.t0 + tick * TICK
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        a, ep = queue.get()
        tick_log.write({
            "tick": tick, "epoch": ep, "qsize": queue.qsize(),
            "action": a.tolist() if a is not None else None,
        })
    stop.set()
    th.join(timeout=5)
    tick_log.close()
    inf_log.close()
    if err:
        print("INFERENCE THREAD ERROR:\n", err[0], flush=True)
        raise RuntimeError(err[0])


# ---------------------------------------------------------------------------
# Sanity + overhead modes
# ---------------------------------------------------------------------------

def mode_sanity(world, policy, preprocessor):
    obs = world.obs_at_tick(0)
    batch = preprocessor(obs)
    inputs = policy._extract_model_inputs(batch)
    images, img_masks, lang_tokens, lang_masks, state, grid = inputs

    # NOTE: sample_actions mutates its noise arg in place (x_t += dt*v_t) — every
    # call below gets a FRESH seeded noise tensor (make_noise is deterministic).
    print("[sanity] warmup (compiles no-grad prefix+velocity)...", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(2):
            policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                        noise=make_noise(policy, 100 + i), image_grid_thw=grid)
    torch.cuda.synchronize()
    print(f"[sanity] warmup done in {time.perf_counter()-t0:.1f}s", flush=True)

    with torch.no_grad():
        ref = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                          noise=make_noise(policy, 42), image_grid_thw=grid)
    cfg = RTCConfig(enabled=True, execution_horizon=EXEC_HORIZON)
    rtc = RTCProcessor(cfg)
    out = rtc_sample(policy.model, rtc, inputs, make_noise(policy, 42), None, 0, EXEC_HORIZON)
    eq = torch.equal(ref, out)
    print(f"[sanity] rtc(delay=0,no leftover) == sync sample_actions: torch.equal={eq}, "
          f"max_abs_diff={(ref-out).abs().max().item():.3e}", flush=True)

    ref_post = policy._postprocess_actions(ref, batch)
    pac = policy.predict_action_chunk(batch, noise=make_noise(policy, 42))
    print(f"[sanity] postprocess(sample_actions) == predict_action_chunk: "
          f"torch.equal={torch.equal(ref_post, pac)}, "
          f"max_abs_diff={(ref_post-pac).abs().max().item():.3e}", flush=True)

    print("[sanity] first guided call (compiles grad path; may take minutes)...", flush=True)
    prev = ref[0, :EXEC_HORIZON].clone()
    t0 = time.perf_counter()
    g = rtc_sample(policy.model, rtc, inputs, make_noise(policy, 42), prev, 4, EXEC_HORIZON)
    torch.cuda.synchronize()
    print(f"[sanity] first guided call ok in {time.perf_counter()-t0:.1f}s; "
          f"diff vs unguided max_abs={(g-ref).abs().max().item():.3e} "
          f"(must be > 0 -> guidance active)", flush=True)
    if not eq:
        print("[sanity] FAILED: bitwise identity does not hold", flush=True)
        sys.exit(1)
    print("[sanity] PASS", flush=True)


def mode_overhead(world, policy, preprocessor, iters=10):
    obs = world.obs_at_tick(0)
    batch = preprocessor(obs)
    inputs = policy._extract_model_inputs(batch)
    images, img_masks, lang_tokens, lang_masks, state, grid = inputs
    cfg = RTCConfig(enabled=True, execution_horizon=EXEC_HORIZON)
    rtc = RTCProcessor(cfg)

    with torch.no_grad():
        for i in range(2):
            policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                        noise=make_noise(policy, 100 + i), image_grid_thw=grid)
    torch.cuda.synchronize()

    def timed(fn, n):
        ts = []
        for _ in range(n):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            w0 = time.perf_counter()
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append((e0.elapsed_time(e1), (time.perf_counter() - w0) * 1e3))
        gpu = sorted(t[0] for t in ts)
        wall = sorted(t[1] for t in ts)
        return gpu[len(gpu) // 2], wall[len(wall) // 2]

    with torch.no_grad():
        ref = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                          noise=make_noise(policy, 42), image_grid_thw=grid)
    torch.cuda.reset_peak_memory_stats()
    g_gpu, g_wall = timed(lambda: rtc_sample(policy.model, rtc, inputs, make_noise(policy, 42), None, 0, EXEC_HORIZON), iters)
    mem_unguided = torch.cuda.max_memory_allocated() / 2**30
    print(f"[overhead] unguided (rtc harness, no leftover): gpu={g_gpu:.1f}ms wall={g_wall:.1f}ms "
          f"peak_mem={mem_unguided:.1f}GiB", flush=True)

    prev = ref[0, :EXEC_HORIZON].clone()
    t0 = time.perf_counter()
    rtc_sample(policy.model, rtc, inputs, make_noise(policy, 42), prev, 8, EXEC_HORIZON)  # grad-path compile
    torch.cuda.synchronize()
    print(f"[overhead] guided-path first call (compile): {time.perf_counter()-t0:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()
    r_gpu, r_wall = timed(lambda: rtc_sample(policy.model, rtc, inputs, make_noise(policy, 42), prev, 8, EXEC_HORIZON), iters)
    mem_guided = torch.cuda.max_memory_allocated() / 2**30
    print(f"[overhead] guided delay=8 horizon={EXEC_HORIZON}: gpu={r_gpu:.1f}ms wall={r_wall:.1f}ms "
          f"(overhead {r_gpu/max(g_gpu,1e-9)-1:+.1%}) peak_mem={mem_guided:.1f}GiB", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(log_dir):
    arms = {}
    for fn in sorted(os.listdir(log_dir)):
        if fn.endswith("_ticks.jsonl"):
            arm = fn.rsplit("_run", 1)[0]
            run = fn.split("_run")[1].split("_")[0]
            arms.setdefault((arm, run), {})["ticks"] = os.path.join(log_dir, fn)
        elif fn.endswith("_inferences.jsonl"):
            arm = fn.rsplit("_run", 1)[0]
            run = fn.split("_run")[1].split("_")[0]
            arms.setdefault((arm, run), {})["inf"] = os.path.join(log_dir, fn)

    summary = {}
    for (arm, run), files in sorted(arms.items()):
        ticks = [json.loads(l) for l in open(files["ticks"])]
        infs = [json.loads(l) for l in open(files["inf"])]
        obs_tick_of = {c["chunk_id"]: c["obs_tick"] for c in infs}
        key = f"{arm}_run{run}"
        s = {"n_ticks": len(ticks), "n_inferences": len(infs)}

        walls = [c["wall_ms"] for c in infs]
        gpus = [c["gpu_ms"] for c in infs]
        if walls:
            s["chunk_wall_ms_median"] = float(np.median(walls[1:])) if len(walls) > 1 else walls[0]
            s["chunk_gpu_ms_median"] = float(np.median(gpus[1:])) if len(gpus) > 1 else gpus[0]
            s["chunk_wall_ms_all"] = [round(w, 1) for w in walls]

        # reaction latency per jump: first executed action from a chunk whose obs is post-jump
        id_key = "chunk_id" if arm == "sync" else "epoch"
        reactions = []
        for jt in JUMP_TICKS:
            post_ids = {cid for cid, ot in obs_tick_of.items() if ot >= jt}
            first = None
            for rec in ticks:
                if rec["tick"] < jt or rec.get("action") is None:
                    continue
                if rec.get(id_key) in post_ids:
                    first = rec["tick"]
                    break
            if first is not None:
                reactions.append((first - jt) * TICK * 1e3)
        s["reaction_ms_per_jump"] = [round(r, 1) for r in reactions]
        if reactions:
            s["reaction_ms_median"] = float(np.median(reactions))

        # boundary vs internal action deltas (7-dim processed actions)
        prev_rec = None
        boundary, internal = [], []
        for rec in ticks:
            a = rec.get("action")
            if a is not None and prev_rec is not None and prev_rec.get("action") is not None:
                d = float(np.linalg.norm(np.array(a) - np.array(prev_rec["action"])))
                if rec.get(id_key) != prev_rec.get(id_key):
                    boundary.append(d)
                elif not rec.get("freeze"):
                    internal.append(d)
            prev_rec = rec
        if boundary:
            s["boundary_jump_median"] = float(np.median(boundary))
            s["boundary_jump_max"] = float(max(boundary))
        if internal:
            s["internal_step_median"] = float(np.median(internal))

        if arm == "sync":
            freeze_ticks = sum(1 for r in ticks if r.get("freeze"))
            s["freeze_frac"] = freeze_ticks / max(len(ticks), 1)
            starts = [c["start_tick"] for c in infs]
            s["period_ticks_median"] = float(np.median(np.diff(starts))) if len(starts) > 1 else None
            s["freeze_ticks_per_cycle_median"] = float(
                np.median([c["end_tick"] - c["start_tick"] + 1 for c in infs])
            ) if infs else None
        else:
            ends = [c["end_tick"] for c in infs]
            s["inference_interval_ticks_median"] = float(np.median(np.diff(ends))) if len(ends) > 1 else None
            if arm == "rtc":
                s["delay_used"] = [c["delay_used"] for c in infs]
                s["real_delay"] = [c["real_delay"] for c in infs]
        summary[key] = s
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sanity", "overhead", "run", "analyze"])
    p.add_argument("--arm", choices=["sync", "async", "rtc"], default=None)
    p.add_argument("--run-id", default="0")
    args = p.parse_args()

    os.makedirs(f"{OUT}/logs", exist_ok=True)
    log_dir = f"{OUT}/logs"

    if args.mode == "analyze":
        print(json.dumps(analyze(log_dir), indent=2))
        return

    assert args.arm is not None or args.mode in ("sanity", "overhead")
    world = World()
    policy, preprocessor = load()

    if args.mode == "sanity":
        mode_sanity(world, policy, preprocessor)
    elif args.mode == "overhead":
        mode_overhead(world, policy, preprocessor)
    elif args.arm == "sync":
        # warmup compile on the first frame before the clock starts
        print("[sync] warmup...", flush=True)
        t0 = time.perf_counter()
        run_one_inference(policy, preprocessor, world.obs_at_tick(0), make_noise(policy, 0))
        run_one_inference(policy, preprocessor, world.obs_at_tick(0), make_noise(policy, 1))
        print(f"[sync] warmup done {time.perf_counter()-t0:.1f}s", flush=True)
        run_sync(world, policy, preprocessor, log_dir, args.run_id)
        print(json.dumps({f"sync_run{args.run_id}": analyze(log_dir)[f"sync_run{args.run_id}"]}, indent=2))
    else:
        use_rtc = args.arm == "rtc"
        print(f"[{args.arm}] warmup...", flush=True)
        t0 = time.perf_counter()
        run_one_inference(policy, preprocessor, world.obs_at_tick(0), make_noise(policy, 0))
        run_one_inference(policy, preprocessor, world.obs_at_tick(0), make_noise(policy, 1))
        if use_rtc:
            # sanity: bitwise identity of the RTC harness vs sync path
            obs = world.obs_at_tick(0)
            batch = preprocessor(obs)
            inputs = policy._extract_model_inputs(batch)
            cfg = RTCConfig(enabled=True, execution_horizon=EXEC_HORIZON)
            rtc = RTCProcessor(cfg)
            images, img_masks, lang_tokens, lang_masks, state, grid = inputs
            with torch.no_grad():
                ref = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                                  noise=make_noise(policy, 42), image_grid_thw=grid)
            out = rtc_sample(policy.model, rtc, inputs, make_noise(policy, 42), None, 0, EXEC_HORIZON)
            eq = torch.equal(ref, out)
            print(f"[rtc] sanity torch.equal={eq} max_abs_diff={(ref-out).abs().max().item():.3e}", flush=True)
            if not eq:
                raise RuntimeError("sanity failed")
            # compile the grad path + warm the transient before the clock starts
            t1 = time.perf_counter()
            for w in range(6):
                run_one_inference(policy, preprocessor, world.obs_at_tick(w),
                                  make_noise(policy, 200 + w),
                                  rtc_processor=rtc, prev=ref[0, :EXEC_HORIZON].clone(), delay=8)
            torch.cuda.synchronize()
            print(f"[rtc] grad-path warmup done {time.perf_counter()-t1:.1f}s", flush=True)
        print(f"[{args.arm}] warmup done {time.perf_counter()-t0:.1f}s", flush=True)
        run_async(world, policy, preprocessor, log_dir, args.run_id, use_rtc)
        res = analyze(log_dir)
        key = f"{args.arm}_run{args.run_id}"
        print(json.dumps({key: res[key]}, indent=2))


if __name__ == "__main__":
    main()
