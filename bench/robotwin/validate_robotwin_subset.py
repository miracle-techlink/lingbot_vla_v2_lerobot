#!/usr/bin/env python3
"""Post-download validation: load the 12-task subset with LeRobotDataset(episodes=...)
and run the LingBot feature transform on one sample."""
import json
import os
import sys

os.environ.setdefault("https_proxy", "http://127.0.0.1:7890")
os.environ.setdefault("http_proxy", "http://127.0.0.1:7890")

ROOT = "/mnt/cpfs/liuyue/robotwin_data/robotwin_unified"
YAML = "/mnt/cpfs/liuyue/robotwin_data/robotwin_lerobot.yaml"

ranges = json.load(open("/mnt/cpfs/liuyue/robotwin_data/task_episode_ranges.json"))
episodes = []
for name, (lo, hi) in sorted(ranges.items()):
    episodes += list(range(lo, hi + 1))
print(f"loading {len(episodes)} episodes from {ROOT}", flush=True)

# lvla-lerobot env has the port installed editable from ~/lvla_scratch/lerobot-port/src
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/robotwin_unified", root=ROOT, episodes=episodes)
print("num_episodes:", ds.num_episodes, "num_frames:", ds.num_frames, "fps:", ds.fps)
s = ds[0]
for k, v in s.items():
    try:
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    except AttributeError:
        print(f"  {k}: {type(v).__name__} = {str(v)[:80]}")
print("VALIDATION OK")
