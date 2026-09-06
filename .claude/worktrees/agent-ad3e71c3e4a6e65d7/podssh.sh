#!/bin/bash
# pod ssh helper: ./podssh.sh '<remote command>'
export SSH_ASKPASS=/home/nvidia/.cache/.d4090_askpass
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=:0
exec ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 26079 root@cpod-1u4t7pbznvyq.podtcp.compshare.cn "$@"
