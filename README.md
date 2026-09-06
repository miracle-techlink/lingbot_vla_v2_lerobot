<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![Tests](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1, reBot B601.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, Vision-Language-Action (VLA) models, World Models, and Reward Models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md)                                                                                                                                                                                                                         |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                                                                                                                                                                                                                                                     |
| **VLAs Models**            | [Pi0](./docs/source/pi0.mdx), [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.7](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx), [EO-1](./docs/source/eo1.mdx), [MolmoAct2](./docs/source/molmoact2.mdx), [LingBot-VLA 2.0](./docs/source/lingbot_vla_v2.mdx), [WALL-OSS](./docs/source/walloss.mdx), [EVO1](./docs/source/evo1.mdx) |
| **World Models**           | [VLA-JEPA](./docs/source/vla_jepa.mdx), [LingBot-VA](./docs/source/lingbot_va.mdx), [FastWAM](./docs/source/fastwam.mdx)                                                                                                                                                                                                                                                                                                                        |
| **Reward Models**          | [SARM](./docs/source/sarm.mdx), [TOPReward](./docs/source/topreward.mdx), [Robometer](./docs/source/robometer.mdx)                                                                                                                                                                                                                                                                                                                              |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub.

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies). For GPU/RAM requirements and expected training time per policy, see the [Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub).

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.
- **[T-Shirt Folding Experiment](https://huggingface.co/spaces/lerobot/robot-folding):** An end-to-end demonstration of folding t-shirts with LeRobot.
- **[LeLab](https://github.com/huggingface/leLab):** A web interface for LeRobot — teleoperate, calibrate, record datasets, replay, and train your SO arm from the browser, no CLI required.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>

<details>
<summary><b>📖 PiPER 训推指南</b>(两数据集:「给桌宠加餐」「机械臂vibe 键盘」;训练→4090D/5880 推理 RTC 完整命令;全文另见 docs/accel/piper_train2infer.md)</summary>

# PiPER 数据集 — 从训练到 4090D/5880 推理(RTC)完整命令文档

两个数据集:「给桌宠加餐」「机械臂vibe 键盘」

仓库与安装(训练机与推理机同源):

```bash
git clone git@github.com:miracle-techlink/lingbot_vla_v2_lerobot.git && cd lingbot_vla_v2_lerobot
uv sync --extra lingbot_vla2                # 推理机若接机器人硬件再加 --extra hardware
##### 或 pip install -e ".[lingbot_vla2,hardware]"
```

- 训练机:A100 ×2,仓库 `lingbot_vla_v2_lerobot`,分支 `feat/lingbot-vla-v2-4090-accel`(推理加速 bake 代码)
- 推理机:RTX 4090D(或 RTX 5880 / sm89,同架构直接复用),直连 PiPER CAN
- 数据集:2 个 PiPER 数据集(已采完)
- 本地 base model:`/home/nvidia/algotithm/models/lingbot-vla-v2-6b-lerobot`(12GB)

#### 第 1 步:本地生成 norm_stats 和 piper.yaml

##### 1.1 piper.yaml
PiPER 是 6 关节 + 夹爪(7 维),与 reBot 维度相同:

```yaml
##### piper.yaml — 松灵 PiPER 单臂(6+1=7维)
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

##### 1.2 生成 norm_stats
```bash
cd ~/algotithm/lingbot-rebot
##### 数据集 A:给桌宠加餐
python gen_rebot_norm_stats.py --dataset-root /path/to/piper_feeding --out piper_feeding_norm_stats.json
##### 数据集 B:机械臂vibe 键盘
python gen_rebot_norm_stats.py --dataset-root /path/to/piper_keyboard --out piper_keyboard_norm_stats.json
```

##### 1.3 生成 overlay
```bash
##### 数据集 A
python make_rebot_ckpt_overlay.py \
  --robot-config piper.yaml \
  --norm-stats piper_feeding_norm_stats.json \
  --out /home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-feeding \
  --dataset-root /path/to/piper_feeding
##### 数据集 B
python make_rebot_ckpt_overlay.py \
  --robot-config piper.yaml \
  --norm-stats piper_keyboard_norm_stats.json \
  --out /home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-keyboard \
  --dataset-root /path/to/piper_keyboard
```

##### 1.4 scp 到 A100
```bash
scp piper.yaml A100:/home/nvidia/lingbot-rebot/
scp -r /home/nvidia/models/lingbot-vla-v2-6b-lerobot-piper-* A100:/home/nvidia/models/
```

#### 第 2 步:A100 训练

##### 2.1 环境
```bash
conda activate pr3967
cd /home/nvidia/platform/20-training/upstream/lerobot-pr3967   # 与 lingbot_vla_v2_lerobot 同源
```

##### 2.2 训练命令(各跑一次,30k steps ≈ 6.5h)

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

##### 2.3 打包 checkpoint
```bash
tar czf piper_feeding_ckpt.tar.gz -C /home/nvidia/output_piper/feeding/checkpoints/last/pretrained_model .
tar czf piper_keyboard_ckpt.tar.gz -C /home/nvidia/output_piper/keyboard/checkpoints/last/pretrained_model .
```

#### 第 3 步:4090D/5880 推理机部署

##### 3.1 最小环境
```bash
conda create -n lerobot_infer python=3.12 -y
conda activate lerobot_infer
git clone git@github.com:miracle-techlink/lingbot_vla_v2_lerobot ~/lerobot-infer
cd ~/lerobot-infer
pip install -e ".[lingbot_vla2,hardware]"
```

##### 3.2 拷 checkpoint
```bash
scp A100:~/piper_feeding_ckpt.tar.gz ~/checkpoints/
mkdir -p ~/checkpoints/piper_feeding
tar xzf ~/checkpoints/piper_feeding_ckpt.tar.gz -C ~/checkpoints/piper_feeding
```

##### 3.3 验证
```bash
python -c "from lerobot.policies.lingbot_vla_v2 import LingbotVLAV2Policy; p = LingbotVLAV2Policy.from_pretrained('~/checkpoints/piper_feeding'); p.to('cuda').eval(); print('OK')"
```

#### 第 4 步:推理运行

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

##### 4.1 基准测试(不接机器人)
```bash
cd ~/lerobot-infer
python bench/bench_lingbot_v2.py infer --ckpt ~/checkpoints/piper_feeding --iters 50 --num-steps 7
```

##### 4.2 真机推理(RTC 模式,推荐)

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

##### 4.3 首次推理注意
- 第 1 帧:几分钟(compile + CUDA graph capture,正常)
- 第 2 帧:~1-2s(cache warm)
- 第 3 帧起:~170ms(稳定)
- `--episode_time_s` 设 ≥60s,让 warmup 有足够时间。

#### 附录:文件清单

| 文件 | 用途 |
|---|---|
| piper.yaml | robot_config:7 维 PiPER → 55 维 canonical 映射 |
| piper_feeding_norm_stats.json | 数据集 A 的归一化统计 |
| piper_keyboard_norm_stats.json | 数据集 B 的归一化统计 |
| lingbot-vla-v2-6b-lerobot-piper-feeding/ | 数据集 A overlay(28KB) |
| lingbot-vla-v2-6b-lerobot-piper-keyboard/ | 数据集 B overlay(28KB) |

参考命令差异:数据集 A `--policy.path=~/checkpoints/piper_feeding --task="给桌宠加餐"`;数据集 B `--policy.path=~/checkpoints/piper_keyboard --task="机械臂vibe 键盘"`。


</details>
