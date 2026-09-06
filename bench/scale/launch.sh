#!/bin/bash
# launch.sh <img_size> <port> <gpu> <compile:true|false> <decomp:0|1>
SZ=$1; PORT=$2; GPU=$3; COMP=${4:-true}; DECOMP=${5:-0}
S=/root/d4090_official/scale
pkill -f "probe_scale_serve[r]" 2>/dev/null
sleep 1
rm -f $S/server_s${SZ}_${PORT}.log
cd /root
CUDA_VISIBLE_DEVICES=$GPU QWEN3VL_PATH=/root/ckpts/Qwen3-VL-4B-Instruct \
  PROBE_CWD=$S/cwd PROBE_DECOMP=$DECOMP PYTHONUNBUFFERED=1 \
  nohup /usr/local/miniconda3/envs/official/bin/python $S/probe_scale_server.py \
  --model_path $S/runs/s$SZ/checkpoints/global_step_1/hf_ckpt \
  --use_length 25 --port $PORT --use_compile $COMP \
  > $S/server_s${SZ}_${PORT}.log 2>&1 &
echo "launched sz=$SZ port=$PORT gpu=$GPU compile=$COMP decomp=$DECOMP pid=$!"
