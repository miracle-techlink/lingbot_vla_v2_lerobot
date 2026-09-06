#!/bin/bash
set -e
S=/root/d4090_official/scale
mkdir -p $S/cwd/configs/robot_configs $S/runs
ln -sfn /root/lingbot-vla-v2-official/assets $S/cwd/assets

# robot configs: 3cam (verbatim copy), 2cam, 1cam
cp /root/lingbot-vla-v2-official/configs/robot_configs/robotwin.yaml $S/cwd/configs/robot_configs/robotwin.yaml
/usr/local/miniconda3/envs/official/bin/python - <<'PY'
import yaml
base = yaml.safe_load(open('/root/lingbot-vla-v2-official/configs/robot_configs/robotwin.yaml'))
for name, keep in [('robotwin_2cam', 2), ('robotwin_1cam', 1)]:
    rc = dict(base)
    rc['images'] = base['images'][:keep]
    p = f'/root/d4090_official/scale/cwd/configs/robot_configs/{name}.yaml'
    yaml.safe_dump(rc, open(p, 'w'), sort_keys=False)
    print('wrote', p, [list(i.keys())[0] for i in rc['images']])
PY

# per-img_size run dirs: yaml + checkpoint layout symlink
for SZ in 256 512; do
  R=$S/runs/s$SZ
  mkdir -p $R/checkpoints/global_step_1
  ln -sfn /root/ckpts/lingbot-vla-v2-6b $R/checkpoints/global_step_1/hf_ckpt
  sed "s/^  img_size: .*/  img_size: $SZ/" /root/official-run/lingbotvla_cli.yaml > $R/lingbotvla_cli.yaml
  grep -n 'img_size' $R/lingbotvla_cli.yaml
done
echo SETUP_DONE
