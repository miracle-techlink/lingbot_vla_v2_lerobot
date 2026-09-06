#!/bin/bash
set -e
ls -la /home/nvidia/reports/robotwin_eval_report/data/*.mp4
for f in /home/nvidia/reports/robotwin_eval_report/data/*.mp4; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>&1 | head -1)
  echo "$f dur=$d"
done
