# LingBot-VLA 2.0 标准化训练/推理命令(lerobot 社区接口)

> 分支 `pr3967-combined`,2026-08-13。所有 flag 均已对照本仓库源码核实存在。
> 入口:训练 `lerobot-train`(多卡 `accelerate launch`),推理 `lerobot-rollout`。
> 以下以 gpudad SO101 数据集 + 转换后的 robotwin 6B ckpt 为例;路径按实际环境替换。

---

## 1. 训练:单卡(80GB)

```bash
lerobot-train \
  --dataset.repo_id=gpudad/so101_pick_cube_v2 \
  --dataset.root=~/lvla_scratch/datasets/gpudad_so101_pick_cube_v2/gpudad_so101_pick_cube_v2 \
  --dataset.streaming=false \
  --policy.type=lingbot_vla_v2 \
  --policy.pretrained_path=~/lvla_scratch/robotwin-6b-lerobot \
  --policy.robot_config_path=~/lvla_scratch/configs/gpudad_so101_robot_config.yaml \
  --policy.norm_stats_path=~/lvla_scratch/configs/gpudad_so101_norm_stats.json \
  --policy.dtype=bfloat16 \
  --policy.optimizer_fused=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=3 \
  --steps=30000 \
  --log_freq=50 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --output_dir=~/output_lerobot_train/gpudad/lingbot_v2_A \
  --job_name=gpudad_lingbot_v2_A \
  --wandb.enable=false
```

要点:
- **`--policy.robot_config_path` / `--policy.norm_stats_path` 必须显式传**:robotwin 转换
  ckpt 内嵌的 robot_config 是 robotwin 相机名(`cam_high`/`cam_left_wrist`),与 gpudad 数据集的
  `front`/`wrist`/`overhead` 不匹配,不传会在 FeatureTransform 断言失败。显式路径优先于 ckpt 内嵌。
- `--batch_size=3` 是单卡 80GB 实测上限(e2e 峰值 72GB);B=4 单卡 OOM。
- `optimizer_fused=true`:fused AdamW,实测 -5.4% step 时间,数值与默认等价(loss 差 1e-6)。
- dense MoE(`moe_dense_max_tokens=512`)、sdpa+bf16 attention 默认已开,无需显式传。
- **不要**设 `--policy.sdpa_backend=CUDNN_ATTENTION`:torch 2.8 cuDNN 后端在
  bf16+bool mask+GQA 下 backward 产 NaN(已实测否决)。
- **不要**开 `--policy.gradient_checkpointing=true`(B≤4 时 +38% 慢,显存够就别开)。
- **不要**改 `--policy.attention_implementation=eager`:300 步真实数据 A/B(2026-08-14)
  loss 曲线等价但 eager 稳态 +2.2% 慢、+2.9GB 显存、B=4 OOM(合成 bench 上的 eager 优势是
  相同样本平铺的假象,真实变长序列下 mem-efficient sdpa 更优)。
- 实测吞吐:1.99 samples/s(B=3),30k 步 ≈ 12.6h(单卡,90k 样本)。

## 2. 训练:多卡(社区标准 `accelerate launch`)

```bash
accelerate launch --num_processes=4 $(which lerobot-train) \
  --dataset.repo_id=gpudad/so101_pick_cube_v2 \
  --dataset.root=~/lvla_scratch/datasets/gpudad_so101_pick_cube_v2/gpudad_so101_pick_cube_v2 \
  --policy.type=lingbot_vla_v2 \
  --policy.pretrained_path=~/lvla_scratch/robotwin-6b-lerobot \
  --policy.robot_config_path=~/lvla_scratch/configs/gpudad_so101_robot_config.yaml \
  --policy.norm_stats_path=~/lvla_scratch/configs/gpudad_so101_norm_stats.json \
  --policy.dtype=bfloat16 \
  --policy.optimizer_fused=true \
  --policy.push_to_hub=false \
  --batch_size=2 \
  --steps=30000 \
  --log_freq=50 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --output_dir=~/output_lerobot_train/gpudad/lingbot_v2_4gpu \
  --job_name=gpudad_lingbot_v2_4gpu \
  --wandb.enable=false
```

要点:
- **`batch_size` 是每卡语义**,global batch = `num_processes × batch_size`(脚本启动日志会打印
  "effective batch size")。
- **DDP 下每卡上限 B=2**(allreduce grad buckets 额外 ~11GB;B=3/卡 OOM,实测)。
- 实测:2 卡 4.0 samples/s,4 卡 8.0 samples/s(2→4 完美线性;allreduce ~0.15s/步)。
- 30k 步:4 卡 ≈8.3h(稳态;首轮冷读视频保守 ≤13h);8 卡推断 ≈8.6h 但样本量翻倍。
- 同一样本预算(480k)下单卡需 ~51h,4 卡 ≈17h,8 卡 ≈8.6h。

## 3. 推理:`lerobot-rollout`

### 3a. 一次性把优化开关写进 ckpt(推荐)

训练出的 ckpt 默认不带推理优化开关,先固化(之后 rollout 只需 `--policy.path`):

```bash
python3 - << 'EOF'
import json, pathlib
p = pathlib.Path.home() / "output_lerobot_train/gpudad/lingbot_v2_A/checkpoints/last/pretrained_model/config.json"
c = json.loads(p.read_text())
c.update({
    "compile_predict_velocity": True,
    "compile_predict_velocity_mode": "max-autotune-no-cudagraphs",
    "compile_prefix": True,
    "preprocess_device": "cuda",
    "num_steps": 10,
})
p.write_text(json.dumps(c, indent=2))
print("baked:", p)
EOF
```

效果(A100,robotwin 3 相机 workload):e2e 1705ms(官方 eager)→ **436ms**;
SO101 单相机 ~300ms。首次推理有一次性编译(数分钟,warmup ≥3 次后稳定)。

### 3b. rollout 命令(SO101 三相机示例)

```bash
lerobot-rollout \
  --policy.path=~/output_lerobot_train/gpudad/lingbot_v2_A/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=my_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, overhead: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --strategy.type=base \
  --inference.type=sync \
  --task="pick up the cube" \
  --fps=30 \
  --duration=60 \
  --display_data=false \
  --play_sounds=false
```

要同时录制 rollout 数据,加:

```bash
  --dataset.repo_id=<user>/eval_gpudad_lingbot_v2 \
  --dataset.root=~/eval_gpudad_lingbot_v2 \
  --dataset.single_task="pick up the cube" \
  --dataset.push_to_hub=false
```

要点:
- **相机 key(`front`/`wrist`/`overhead`)必须和训练用 robot_config 的 origin_keys 一致**
  (observation 按 `observation.images.<key>` 进 batch,靠 gpudad robot_config 映射到 canonical
  camera_top/camera_wrist_left/camera_wrist_right)。
- 相机 key 与训练不同时用 `--rename_map` 补救,但建议直接对齐。
- reBot 臂:本分支已注册 `--robot.type=rebot_b601_follower`(双臂 `bi_rebot_b601_follower`),
  端口/相机按 reBot 采集环境(udev 软链 ttyUSB0 主臂)替换即可。
- 推理优化的 CLI 覆盖(--policy.compile_predict_velocity=true 等)依赖 draccus 对
  ckpt 子类型的解析,**优先用 3a 的固化方式**,更不易踩坑。

## 4. 本分支新增 flag 速查

| flag | 默认 | 说明 |
|---|---|---|
| `--policy.optimizer_fused` | false | fused AdamW 单 kernel step;训练 -5.4%,数值等价 |
| `--policy.sdpa_backend` | null | 强制 SDPA 后端;**勿设 CUDNN_ATTENTION(训练 NaN)** |
| `--policy.compile_predict_velocity` | false | compile 去噪步(推理,需配 mode) |
| `--policy.compile_predict_velocity_mode` | default | 推荐 `max-autotune-no-cudagraphs` |
| `--policy.compile_prefix` | false | compile vision+prefix(需上面先开) |
| `--policy.preprocess_device` | null | `cuda` = GPU 图像预处理(仅推理;训练自动留 CPU) |
| `--policy.moe_dense_max_tokens` | 512 | dense 双 GEMM MoE 阈值(默认开;0 关闭) |
| `--policy.gradient_checkpointing` | false | B≤4 别开(+38%);大 batch 显存不够时才用 |
| `--policy.num_steps` | 10 | 去噪步数(已锁定 10) |

## 5. 实测性能锚点(A100-80GB,2026-08-13,宿主机有争抢)

| 场景 | 数字 |
|---|---|
| 推理 e2e(robotwin 3 相机,优化栈固化后) | 436.6ms(官方 eager 1705ms,官方 compile 1113ms) |
| 推理 e2e(SO101 单相机) | ~300-315ms |
| 训练单卡 B=3(lerobot-train,真实数据集) | updt 0.94s/step,1.99 samples/s |
| 训练 4 卡 B=2/卡 | 1.00s/step,8.0 samples/s,30k 步 ≈8.3h |
| 训练 8 卡 B=2/卡(推断) | ~1.05s/step,~15 samples/s,30k 步 ≈8.6h(样本量×2) |
