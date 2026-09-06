# PiPER 数据集 — 从训练到 4090D/5880 推理(RTC)完整命令文档

两个数据集:「给桌宠加餐」「机械臂vibe 键盘」

仓库与安装(训练机与推理机同源):

```bash
git clone git@github.com:miracle-techlink/lingbot_vla_v2_lerobot.git && cd lingbot_vla_v2_lerobot
uv sync --extra lingbot_vla2                # 推理机若接机器人硬件再加 --extra hardware
# 或 pip install -e ".[lingbot_vla2,hardware]"
```

- 训练机:A100 ×2,仓库 `lingbot_vla_v2_lerobot`,分支 `feat/lingbot-vla-v2-4090-accel`(推理加速 bake 代码)
- 推理机:RTX 4090D(或 RTX 5880 / sm89,同架构直接复用),直连 PiPER CAN
- 数据集:2 个 PiPER 数据集(已采完)
- 本地 base model:`/home/nvidia/algotithm/models/lingbot-vla-v2-6b-lerobot`(12GB)

## 第 1 步:本地生成 norm_stats 和 piper.yaml

### 1.1 piper.yaml
PiPER 是 6 关节 + 夹爪(7 维),与 reBot 维度相同:

```yaml
# piper.yaml — 松灵 PiPER 单臂(6+1=7维)
states:
  observation.state.arm.position:
    origin_keys:
      - observation.state: { start: 0, end: 6 }
  observation.state.effector.position:
    origin_keys:
      - observation.state: { start: 6, end: 7 }
actions:
  action.arm.position:
    origin_keys:
      - action: { start: 0, end: 6 }
    subtract_state: false
  action.effector.position:
    origin_keys:
      - action: { start: 6, end: 7 }
    subtract_state: false
images:
  observation.images.camera_top:
    origin_keys: observation.images.front
  observation.images.camera_wrist_left:
    origin_keys: observation.images.wrist
norm_stats: piper_norm_stats.json
```

### 1.2 生成 norm_stats
```bash
cd ~/algotithm/lingbot-rebot
# 数据集 A:给桌宠加餐
python gen_rebot_norm_stats.py --dataset-root /path/to/piper_feeding --out piper_feeding_norm_stats.json
# 数据集 B:机械臂vibe 键盘
python gen_rebot_norm_stats.py --dataset-root /path/to/piper_keyboard --out piper_keyboard_norm_stats.json
```

### 1.3 生成 overlay
```bash
# 数据集 A
python make_rebot_ckpt_overlay.py \
  --robot-config piper.yaml \
  --norm-stats piper_feeding_norm_stats.json \
  --out /home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-feeding \
  --dataset-root /path/to/piper_feeding
# 数据集 B
python make_rebot_ckpt_overlay.py \
  --robot-config piper.yaml \
  --norm-stats piper_keyboard_norm_stats.json \
  --out /home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-keyboard \
  --dataset-root /path/to/piper_keyboard
```

### 1.4 scp 到 A100
```bash
scp piper.yaml A100:/home/nvidia/lingbot-rebot/
scp -r /home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-* A100:/home/nvidia/models/
```

## 第 2 步:A100 训练

### 2.1 环境
```bash
conda activate pr3967
cd /home/nvidia/platform/20-training/upstream/lerobot-pr3967   # 与 lingbot_vla_v2_lerobot 同源
```

### 2.2 训练命令(各跑一次,30k steps ≈ 6.5h)

数据集 A:给桌宠加餐

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes=2 --mixed_precision=bf16 \
  lerobot-train \
  --dataset.repo_id=local/piper_feeding \
  --dataset.root=/path/to/piper_feeding \
  --dataset.video_backend=pyav \
  --policy.path=/home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-feeding \
  --policy.dtype=bfloat16 --policy.loss_type=L1_fm \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=1e-5 \
  --policy.gradient_checkpointing=false \
  --policy.optimizer_fused=true \
  --peft.r=32 --peft.lora_alpha=64 \
  --batch_size=4 --steps=30000 --save_freq=2000 \
  --num_workers=6 \
  --output_dir=/home/nvidia/output_piper/feeding
```

数据集 B:机械臂vibe 键盘(仅 `--policy.path`/`--dataset.*`/`--output_dir` 不同)

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes=2 --mixed_precision=bf16 \
  lerobot-train \
  --dataset.repo_id=local/piper_keyboard \
  --dataset.root=/path/to/piper_keyboard \
  --dataset.video_backend=pyav \
  --policy.path=/home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-keyboard \
  --policy.dtype=bfloat16 --policy.loss_type=L1_fm \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps=30000 \
  --policy.scheduler_decay_lr=1e-5 \
  --policy.gradient_checkpointing=false \
  --policy.optimizer_fused=true \
  --peft.r=32 --peft.lora_alpha=64 \
  --batch_size=4 --steps=30000 --save_freq=2000 \
  --num_workers=6 \
  --output_dir=/home/nvidia/output_piper/keyboard
```

### 2.3 打包 checkpoint
```bash
tar czf piper_feeding_ckpt.tar.gz -C /home/nvidia/output_piper/feeding/checkpoints/last/pretrained_model .
tar czf piper_keyboard_ckpt.tar.gz -C /home/nvidia/output_piper/keyboard/checkpoints/last/pretrained_model .
```

## 第 3 步:4090D/5880 推理机部署

### 3.1 最小环境
```bash
conda create -n lerobot_infer python=3.12 -y
conda activate lerobot_infer
git clone git@github.com:miracle-techlink/lingbot_vla_v2_lerobot ~/lerobot-infer
cd ~/lerobot-infer
pip install -e ".[lingbot_vla2,hardware]"
```

### 3.2 拷 checkpoint
```bash
scp A100:~/piper_feeding_ckpt.tar.gz ~/checkpoints/
mkdir -p ~/checkpoints/piper_feeding
tar xzf ~/checkpoints/piper_feeding_ckpt.tar.gz -C ~/checkpoints/piper_feeding
```

### 3.3 验证
```bash
python -c "from lerobot.policies.lingbot_vla_v2 import LingbotVLAV2Policy; p = LingbotVLAV2Policy.from_pretrained('~/checkpoints/piper_feeding'); p.to('cuda').eval(); print('OK')"
```

## 第 4 步:推理运行

加速配置(已 baked 进 checkpoint,加载即生效):

| 配置 | 值 | 说明 |
|---|---|---|
| attention_implementation | eager | 4090 上比 sdpa 快 |
| dtype | bfloat16 | |
| compile_predict_velocity | true | max-autotune-no-cudagraphs |
| compile_prefix | true | 编译 vision + prefix KV |
| use_cudagraph_denoise | true | denoise 循环捕获为 CUDA graph |
| use_cudagraph_prefix | true | prefix KV fill 为 CUDA graph |
| num_steps | 7 | 7 步 denoise |

4090D 实测:7 步 ~170ms + RTC guidance,等效延迟满足 30fps 真机闭环。

### 4.1 基准测试(不接机器人)
```bash
cd ~/lerobot-infer
python bench/bench_lingbot_v2.py infer --ckpt ~/checkpoints/piper_feeding --iters 50 --num-steps 7
```

### 4.2 真机推理(RTC 模式,推荐)

数据集 A:给桌宠加餐

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=~/checkpoints/piper_feeding \
  --policy.n_action_steps=25 \
  --device=cuda \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=ros2_piper_follower \
  --robot.port=can0 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/video_wrist, fps: 30, width: 640, height: 480}, front: {type: opencv, index_or_path: /dev/video_front, fps: 30, width: 640, height: 480} }" \
  --task="给桌宠加餐" \
  --fps=30 --display_data=true \
  --dataset.repo_id=local/eval_piper_feeding \
  --dataset.single_task="给桌宠加餐" \
  --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
  --dataset.num_episodes=10 --dataset.push_to_hub=false
```

数据集 B:机械臂vibe 键盘(只换 3 项,其余不变:policy.path/task/single_task)

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=~/checkpoints/piper_keyboard \
  --policy.n_action_steps=25 \
  --device=cuda \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 \
  --robot.type=ros2_piper_follower \
  --robot.port=can0 \
  --robot.cameras="同上(wrist/front)" \
  --task="机械臂vibe 键盘" \
  --fps=30 --display_data=true \
  --dataset.repo_id=local/eval_piper_keyboard \
  --dataset.single_task="机械臂vibe 键盘" \
  --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
  --dataset.num_episodes=10 --dataset.push_to_hub=false
```

### 4.3 首次推理注意
- 第 1 帧:几分钟(compile + CUDA graph capture,正常)
- 第 2 帧:~1-2s(cache warm)
- 第 3 帧起:~170ms(稳定)
- `--episode_time_s` 设 ≥60s,让 warmup 有足够时间。

## 附录:文件清单

| 文件 | 用途 |
|---|---|
| piper.yaml | robot_config:7 维 PiPER → 55 维 canonical 映射 |
| piper_feeding_norm_stats.json | 数据集 A 的归一化统计 |
| piper_keyboard_norm_stats.json | 数据集 B 的归一化统计 |
| lingbot-vla-v2-6b-lerobot-piper-feeding/ | 数据集 A overlay(28KB) |
| lingbot-vla-v2-6b-lerobot-piper-keyboard/ | 数据集 B overlay(28KB) |

参考命令差异:数据集 A `--policy.path=~/checkpoints/piper_feeding --task="给桌宠加餐"`;数据集 B `--policy.path=~/checkpoints/piper_keyboard --task="机械臂vibe 键盘"`。
