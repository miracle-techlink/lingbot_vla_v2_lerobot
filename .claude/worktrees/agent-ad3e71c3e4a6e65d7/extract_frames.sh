#!/bin/bash
set -e
cd /home/nvidia/reports/robotwin_eval_report
mkdir -p figs
for pct in 0 33 66 95; do
  t=$((50 * pct / 100))
  ffmpeg -y -v error -ss "$t" -i data/ep_r6.mp4 -frames:v 1 -vf scale=480:-1 "figs/rollout_f${pct}.png"
done
ls -la figs/
