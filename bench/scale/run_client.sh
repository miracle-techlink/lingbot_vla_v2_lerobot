#!/bin/bash
# run_client.sh <port> <robo> <ncam> <ncalls> <tag>
S=/root/d4090_official/scale
nohup /usr/local/miniconda3/envs/official/bin/python $S/scale_client.py $1 $2 $3 $4 $5 > $S/client_$5.log 2>&1 &
echo "client $5 pid=$!"
