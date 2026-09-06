#!/bin/bash
# pod scp helper: ./podscp.sh <remote> <local>
export SSH_ASKPASS=/home/nvidia/.cache/.d4090_askpass
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0
exec scp -P 26079 -o StrictHostKeyChecking=no "root@cpod-1u4t7pbznvyq.podtcp.compshare.cn:$1" "$2"
