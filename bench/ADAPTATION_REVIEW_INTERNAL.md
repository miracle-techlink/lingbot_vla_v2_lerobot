# LingBot-VLA 2.0 × lerobot 适配 — 内部审阅手册

> **INTERNAL — 本文件不进 PR。** 含内网地址、真实路径与机器入口;社区版文档见
> `docs/source/policy_lingbot_vla_v2_README.md`(全占位符)。本文件刻意保持 untracked,
> 推 PR 前勿 `git add`。
>
> 日期: 2026-08-17 · 分支: `feat/lingbot-vla-v2-4090-accel` · 基线: lerobot PR #3967

---

## 1. 适配全景(六层)

### L1 模型接入
上游 LingBot-VLA 2.0(Qwen3-VL-4B 主干 + 36 层稀疏 MoE 动作专家,flow matching,~6B)
→ `src/lerobot/policies/lingbot_vla_v2/`:

| 文件 | 职责 |
|---|---|
| `convert_upstream_checkpoint.py` | 上游权重 → 自包含 lerobot ckpt(内嵌 robot config/norm stats/前后处理) |
| `configuration_*.py` | 全部开关:attention 后端、dtype、compile、cudagraph、dense MoE 阈值、lr 调度、LoRA 字段 |
| `modeling_lingbot_vla_v2_base.py` | 双流前向(主干+专家)、去噪环、KV 前缀、CUDA Graph 捕获 |
| `feature_transform.py` / `data_transform.py` | 55 维 canonical 动作空间的打包/归一化(非 lerobot 通用 normalizer) |
| `ee_pose_transform.py` | 末端位姿 slot |
| `flex_attention.py` | flex 后端(可选) |
| `lora.py` | 自研 LoRA 注入:regex 可达 MoE 路由专家(社区 PEFT 只到共享专家) |
| `export_merged.py` | adapter → 合并 ckpt(fp32 计算回存,差值=bf16 舍入);合并版才能走 CUDA Graph 热路径 |

### L2 训练接入
- `lerobot-train` 全参数/LoRA 均走社区入口;多卡 = accelerate launch。
- 验收三红线:同种子 loss 逐位一致(1e-6)、300 步真实训练曲线等价、LIBERO 偏移 ≤2%。
- 否决存档:cuDNN SDPA(反向 96% 张量 NaN)、fp16(偏移 3.33%)、bench 短测速记(被真实训练推翻)。

### L3 推理接入(RTC)
`predict_action_chunk` 扩展推理延迟与残留前缀参数 → `policies/rtc/` 引导式去噪。
三件关键适配:① 引导目标保持**归一化采样空间**(否则前后处理破坏引导谱系);
② 引导路径首用时**冻结参数**(避免全网络反传 OOM);③ 变观测形状自动弃用旧 graph 重捕。

### L4 加速
- 训练:sdpa+bf16+dense MoE(≤768 token)+去同步 2378→1202 ms;fused AdamW →**1137.3 ms(2.09×)**。
- 推理:compile 去噪环+前缀、GPU 预处理(290→36 ms)、CUDA Graph 去噪环、4 步去噪 →
  **108.5 ms/chunk**(4090D,10 步 153.0 / 7 步 130.9);`precompute_grid_thw` 每块再省 6–7 ms。
- 全部开关已烘焙进 ckpt `config.json`,加载即生效,无 CLI 负担。

### L5 硬件
- rebot B601 follower(socketcan can0,1 Mbps)。
- OpenCV 相机:`CAP_PROP_FRAME_WIDTH` set() 返回 False 但实际生效 —— 改为按实际值判死刑,V4L2 怪癖。
- Orbbec 相机插件移植自采集栈(pyorbbecsdk2 后端,serial 打开,楔死自愈);
  注册必须在 `cameras/__init__.py` import(draccus ChoiceRegistry 时机)。
  USB2 必须 `color_format: mjpg`;固件"一次性会话":任何会话结束(含正常断开)后 setXu 失败,
  SOP = 每次启动前独立进程 USB 复位(`usbreset_orbbec.py`)。

### L6 产线状态(2026-08-17)
- A100 rebot LoRA 28k → merged_28000 已交付,开环 L1 首步 **-61%**,GB10 复核对齐(Δ5.9%)。
- pod robotwin LoRA 30k 在训(~24k/30k),auto_eval 等 030000 自动粗扫 4 ckpt;
  另挂 20k 提前评测接力(训练退出后 GPU1 并行)。

---

## 2. 原生启动脚本全集(原文)

### 2.1 A100 — rebot 真机数据 LoRA(已完成,交付 merged_28000)

入口 `launch_rebot30k.sh`:

```bash
#!/bin/bash
# rebot 30k on 2xA100(6,7): bs4/卡nogc flash mlp regex — 用户拍板 2026-08-17
# 预期: updt ~0.70s, 30k ≈ 6.5-7h, mem ~22GB/80GB, 有效 batch 8
cd /home/liuyue/accel-bench/rebot
GC=false BS=4 STEPS=30000 MAIN_PORT=29527 \
OUT=/home/liuyue/accel-bench/rebot/train_lora_rebot_30k \
bash /home/liuyue/accel-bench/rebot/train_lora_rebot_a100.sh \
  2>&1 | tee /home/liuyue/accel-bench/rebot/rebot_train.log
```

主体 `train_lora_rebot_a100.sh`(环境变量 GPUS/STEPS/BS/GC/OUT/POLICY_PATH 可覆盖;节选头部,
加速/LoRA 参数全文见 §2.2 pod 版,两者同构):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pr3967
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd ~/accel-bench/speed_probe_0816/lerobot-pr3967
GPUS=${GPUS:-6,7}; STEPS=${STEPS:-20000}; BS=${BS:-8}; GC=${GC:-true}
DATA=/home/liuyue/rebot_port/rebot_rgb1_20260813_182056
export CUDA_VISIBLE_DEVICES=$GPUS; NPROC=$(awk -F, '{print NF}' <<<"$GPUS")

accelerate launch --num_processes=$NPROC --main_process_port=${MAIN_PORT:-29527} \
  --mixed_precision=bf16 $(which lerobot-train) \
  --dataset.repo_id=local/rebot_rgb1_20260813_182056 --dataset.root="$DATA" \
  --rename_map='{"observation.images.front": "observation.images.cam_high",
                 "observation.images.wrist": "observation.images.cam_left_wrist"}' \
  --dataset.video_backend=pyav \
  --policy.path=$HOME/lvla_scratch/robotwin-6b-lerobot \
  --policy.robot_config_path=/home/liuyue/rebot_port/rebot_lerobot.yaml \
  --policy.norm_stats_path=/home/liuyue/rebot_port/rebot_norm_stats.json \
  --policy.tokenizer_path=$HOME/lvla_scratch/Qwen3-VL-4B-Instruct \
  --policy.processor_path=$HOME/lvla_scratch/Qwen3-VL-4B-Instruct \
  --policy.device=cuda --policy.push_to_hub=false --policy.dtype=bfloat16 \
  --policy.attention_implementation=sdpa --policy.vit_attn_implementation=sdpa \
  --policy.loss_type=L1_fm --policy.attn_split_prefix_suffix=true \
  --policy.attn_split_prefix_backend=flash \
  --policy.gradient_checkpointing=$GC --policy.optimizer_fused=true \
  --policy.optimizer_lr=1e-4 --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps=$STEPS --policy.scheduler_decay_lr=1e-5 \
  --policy.moe_dense_max_tokens=768 \
  --peft.r=32 --peft.lora_alpha=64 \
  --peft.target_modules='.*qwen_expert.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)' \
  --batch_size=$BS --steps=$STEPS --save_freq=1000 --log_freq=20 \
  --num_workers=6 --wandb.enable=false --output_dir=$OUT
```

### 2.2 pod(4×4090D)— robotwin 出分线(在训)

训练 `bench/train_lora_4090_20k.sh`(watchdog 以 `GPUS=0,1,2,3 STEPS=30000 BS=4 RESUME=true
POLICY_PATH=$LAST` 调用)。与 A100 版差异:GC=true(4090 24GB 需要)、数据集
`lerobot/robotwin_unified` 按 `task_episode_ranges.json` 展开全 episode 列表、
A/B 判决(2026-08-16):B 赢 —— regex 只挂 action expert 注意力,4.68 vs 5.44 s/step。
其余参数同 §2.1。

watchdog `/root/rtw_watchdog.sh`(dataloader 自动复活):

```bash
CKPT=/root/outputs/train_lora_r32_20k/checkpoints/030000/pretrained_model/config.json
LAST=/root/outputs/train_lora_r32_20k/checkpoints/last/pretrained_model
LOG=/root/logs/train_30k_auto.log
# 规则: 030000 出现即退出; 进程消失两次→拉起; data_s>1.5 连续两次→TERM+从 last 恢复
# ⚠ 已知坑: last 指针曾回退到 005000 → 整段重饭 5.5h(2026-08-17 观察)
```

自动评测 `/root/auto_eval_rtw.sh`:

```bash
# 等 030000 → 等训练退出 → 粗扫 4 ckpt × 6 任务 × 3 trials(≈72 episodes)
for STEP in 030000 024000 016000 008000; do
  bash /root/lerobot-port/bench/eval_ckpt_pod.sh $STEP "" 3 0
done
```

单 ckpt 评测 `/root/lerobot-port/bench/eval_ckpt_pod.sh <STEP> [tasks] [trials] [seed]`:

```bash
# 1) export_merged(训练占卡时走 CPU,fp32,RAM 192G 够)
python -m lerobot.policies.lingbot_vla_v2.export_merged \
  --adapter $CKPT --output $MERGED --tokenizer-path /root/ckpts/Qwen3-VL-4B-Instruct
# 2) 闭环: RoboTwin env,llvmpipe 软渲染(mesa lvp, 不占 GPU)
export MESA_LIB=/root/xpl_ws/envs/mesa/lib; export VK_ICD_FILENAMES=$MESA_LIB/lvp_icd.x86_64.json
export LP_NUM_THREADS=8; export LINGBOT_VLA_V2_CKPT=$MERGED
bash $XPL_POLICY/eval.sh RoboTwin "$task" merged_${STEP} aloha_agilex joint $seed 0 0 py312 xpl
```

20k 提前评测接力 `/root/eval_20k_watch.sh`(2026-08-17 部署,PID 见 `pgrep -f eval_20k_watch`):

```bash
while [ ! -f $CKPT/config.json ]; do sleep 120; done   # 020000
while pgrep -f "lerobot-trai[n]" >/dev/null; do sleep 120; done
sleep 600   # 让 auto_eval 的 030000 先占 GPU0
CUDA_VISIBLE_DEVICES=1 bash /root/lerobot-port/bench/eval_ckpt_pod.sh 020000 "" 3 0
```

### 2.3 GB10 — 真机 RTC rollout(部署链)

`bench/rebot/rollout_rtc_local.sh`(环境变量 POLICY/TASK/FPS/DURATION/HORIZON=12/GUID_W=10.0/DISPLAY_DATA):

```bash
# 每次(含每次)启动前: Orbbec 固件一次性会话 SOP
/home/nvidia/envs/pr3967b/bin/python \
  /home/nvidia/lingbot-ttt/platforms/rebot_datacollect/scripts/usbreset_orbbec.py \
  || echo "[warn] Orbbec USB 复位失败"
export PATH="/home/nvidia/envs/pr3967b/bin:$PATH"   # rr.spawn() 按 PATH 找 rerun viewer

exec /home/nvidia/envs/pr3967b/bin/lerobot-rollout \
  --config_path=/home/nvidia/lerobot-pr3967/bench/rebot/rebot_rtc_robot.yaml \
  --strategy.type=base \
  --policy.path=${POLICY:-/home/nvidia/models/rebot30k/merged_28000} \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=12 \
  --inference.rtc.max_guidance_weight=10.0 \
  --task="shake hands" --fps=30 --duration=${DURATION:-60} \
  --display_data=true
```

`bench/rebot/rebot_rtc_robot.yaml`(内嵌相机必须走 YAML,CLI 内联字典 draccus 解析不了):

```yaml
robot:
  type: rebot_b601_follower
  port: can0
  can_adapter: socketcan
  id: follower1
  cameras:
    front: {type: opencv, index_or_path: /dev/video0, fps: 30,
            width: 640, height: 480, fourcc: MJPG}
    wrist:  {type: orbbec, serial_number_or_name: CV2856D0006R, fps: 30,
            width: 640, height: 480, color_format: mjpg}
```

开环评测(预测-真值逐帧):

```bash
python bench/open_loop_curves.py --ckpt <merged_ckpt> --stride 25   # → open_loop_curves_*.json/png
```

---

## 3. 机器入口

| 机器 | 入口 | 说明 |
|---|---|---|
| A100 8×80G(liuyue 共享) | `bench/a100.py exec/put/get`(8.130.97.174:1014, askpass) | **只用 GPU 6,7**;密码在 `~/.cache/.a100_pass`(勿外泄);出网走 clash:7890 |
| 4090 pod | `bench/d4090.py exec/put/get`(compshare pod, askpass) | 全卡可用;密码 `~/.cache/.d4090_pass` |
| GB10(本地) | venv `/home/nvidia/envs/pr3967b`(py3.12 + torch cu130 + 本 repo -e) | sm_121 边缘支持,eager 极慢,走 CUDA Graph 路径 |

---

## 4. 坑位速查(踩过的,一句话版)

| 坑 | 解法 |
|---|---|
| Orbbec 任何会话结束即 setXu failed | 启动前独立进程 USB 复位;mjpg;拔线 15s 兜底 |
| Orbbec/自定义相机 `Couldn't find a choice class` | `cameras/__init__.py` 顶层 import 注册(draccus 时机) |
| OpenCV 相机 width set() 返回 False | V4L2 怪癖,按实际读回值判断,勿按返回值 |
| `--display_data=1` 解析失败 | 必须 true/false 字面量 |
| rerun viewer spawn 失败 | venv/bin 加进 PATH |
| can0 掉 DOWN(USB 插拔风暴后) | `sudo ip link set can0 up type can bitrate 1000000` |
| rollout 无 dry-run | 启动=上电=执行;DURATION=10 短跑 + 断夹爪气路起步 |
| pod dataloader 饥饿(data_s>1.5) | watchdog 监控两振出局 TERM 重拉;last 指针回退已观察到一次 |
| uv sync 改写 uv.lock 镜像源 | 提交前 `git checkout uv.lock` |
| FSDP2 2×24GB 四条件缺一即炸 | auto_wrap 三层类串 + offload + 本分支 policy() 修复 |

---

## 5. 数字总账

| 项 | 数值 |
|---|---|
| 训练单步(A100,2×卡,B=4/卡) | 2378.1 → **1137.3 ms**(2.09×),loss 逐位一致 1e-6 |
| LoRA 显存 | 全参固定态 65.7GB → **22GB/卡** |
| 30k 步产线 | 2×A100 ≈ 6.5h;8×A100 外推 ≈ 9h |
| 推理 chunk 延迟(4090D) | 1039.5 → **108.5 ms**(4 步+graph;7 步 130.9,10 步 153.0) |
| 官方栈对照(A100 三相机 e2e) | 官方 eager 1705 / compile 1113 vs 本文 **436.6 ms**(3.9×/2.55×) |
| 开环(merged_28000) | L1 均值 -53%,首步 **-61%**;MAE 2.489°,关节 1.1–2.7° |
| 跨平台 | A100 3.240 vs GB10 3.049(Δ5.9%,对齐) |a
