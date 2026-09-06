# LingBot-VLA 2.0 × reBot B601 实操手册(采集 → 训练 → 上机 rollout)

> 分支 `pr3967-combined`,2026-08-13。配套采集仓:
> https://github.com/miracle-techlink/rebot_datacollect
> 通用版命令手册见 `lingbot_vla_train_rollout_commands.md`;本文是 reBot 臂的端到端落地版。

---

## 0. 一句话流程

```
rebot_datacollect 采集(闸门式) → 生成 robot_config.yaml + norm_stats.json
→ lerobot-train(本分支) → bake 推理开关进 ckpt → lerobot-rollout 上真机
```

**两套环境不要混:**

| 环境 | 用途 | 内容 |
|---|---|---|
| conda `data_collect` | 采集(跟随 rebot_datacollect) | lerobot v0.6.1 + 三件套插件 |
| 本分支环境 | 训练 + rollout | pr3967-combined(含 lingbot_vla_v2 policy 和 rebot_b601_follower) |

## ⚠️ 0.1 先确认你的臂是 -DM 还是 -RS

- **采集仓 `rebot_datacollect` 的 `rebot_follower` 驱动:B601-RS(RobStride 电机,SocketCAN/PCAN)**
- **本分支注册的 `rebot_b601_follower`:B601-DM(达妙电机,motorbridge 包)**

采集用什么驱动都行(数据集格式与驱动无关),但 **rollout 必须在本分支环境跑**(只有这里有
lingbot_vla_v2 policy),所以上机时臂的驱动必须在本分支可用:

- 你的臂是 **DM** → 直接用 `--robot.type=rebot_b601_follower`,本文命令开箱即用。
- 你的臂是 **RS** → 需要先把采集仓的 RobStride 驱动移植为本分支的 robot 插件
  (参照 `src/lerobot/robots/rebot_b601_follower/` 的结构),否则 rollout 起不来。
  训练不受影响,先训,移植后上机。

## 1. 采集(rebot_datacollect,data_collect 环境)

按采集仓 README 的标准流程(详见该仓 scripts/):

```bash
# 一次性:装环境 + 插件
bash scripts/setup_env.sh
# CAN 起来 + 校准(仅首次)
bash scripts/setup_rebot_can.sh
lerobot-calibrate --robot.type=rebot_follower --robot.id=follower1 \
  --robot.port=can0 --robot.can_adapter=socketcan
# 遥操作 dry-run(rerun 可视化确认映射正常)
bash scripts/teleop_rebot.sh
# 闸门式录制(每段固定时长,当场决定保留/丢弃)
PY=~/miniconda3/envs/data_collect/bin/python CAN=can0 \
  FRONT_CAM=/dev/v4l/by-id/<front-cam-id> \
  REPO_ID=<user>/rebot_pick_place TASK="pick up the black object and place it in the box" \
  EPISODES=50 EP_TIME=15 PUSH=false \
  bash scripts/record_rebot_gated.sh
```

采集仓用命换来的坑(照做):

- **Orbbec 腕部相机必须 USB3**,且固件每次上电只允许一个干净会话(脚本已带自动 USB reset);
  掉成 USB2 会直接挂。
- 相机一律用 `/dev/v4l/by-id/` 路径,不用 `/dev/videoN`(插拔会变号)。
- `NONBLOCK=1` 默认开(循环 ~77Hz vs 阻塞 ~30Hz)。
- `STREAM_ENCODE` 默认关——开过一次 SIGINT 死锁丢 27 段,别碰。
- 数据集落在 `~/.cache/huggingface/lerobot/<REPO_ID>_<时间戳>/`,`RESUME=1` 可续录。
- Jetson 上先 `bash scripts/maxn_lock.sh` 锁频。

采完核对数据集 keys(下一步要对照):

```
observation.state            # 7 维(6 关节 + 夹爪)
action                       # 7 维,与 state 同坐标系
observation.images.front     # UVC 前视
observation.images.wrist     # Orbbec 腕部 RGB
observation.images.wrist_depth  # Orbbec 深度 uint16 mm(本手册暂不使用,见 §7)
```

## 2. 生成 robot_config.yaml(reBot 版)

lingbot_vla_v2 的 ckpt 用的是统一 canonical 布局:关节 `arm.position`(14 维槽位)、
相机 `camera_top / camera_wrist_left / camera_wrist_right`。robot_config 负责把
reBot 数据集的原始 key 映射上去,**缺位的 canonical 相机会自动 mask(设计与推理一致),
所以两路物理相机只需要映射两条**:

存为 `~/rebot_lingbot/rebot_b601_robot_config.yaml`:

```yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state:
            start: 0
            end: 7          # reBot 7 维(6 关节 + 夹爪),14 维槽位其余补零

actions:
  - action.arm.position:
      origin_keys:
        - action:
            start: 0
            end: 7
      subtract_state: false

images:
  - observation.images.camera_top:
      origin_keys: observation.images.front
  - observation.images.camera_wrist_left:
      origin_keys: observation.images.wrist
  # camera_wrist_right 不映射 → 自动 mask;若你装了第三路(俯视/对侧)相机,加:
  # - observation.images.camera_wrist_right:
  #     origin_keys: observation.images.overhead

norm_stats: /home/<you>/rebot_lingbot/rebot_b601_norm_stats.json
```

> 训练日志里出现 `canonical slot 'observation.images.camera_wrist_right' left unfilled`
> 的 warning 是**预期行为**(缺位 mask),不是错误。

## 3. 生成 norm_stats.json

对本数据集全量帧统计 state/action 的 mean/std(脚本在本分支环境跑,
把 root 换成你实际的数据集路径):

```bash
python3 - << 'EOF'
import json
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "<user>/rebot_pick_place",
    root="~/.cache/huggingface/lerobot/<user>_rebot_pick_place_<时间戳>",
)
s = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float64)
a = np.asarray(ds.hf_dataset["action"], dtype=np.float64)
assert s.shape[1] == 7 and a.shape[1] == 7, (s.shape, a.shape)

def stats(x):
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": (x.std(0) + 1e-8).tolist(),   # 防夹爪常量化导致除零
    }

out = {"norm_stats": {
    "observation.state.arm.position": stats(s),
    "action.arm.position": stats(a),
}}
with open("rebot_b601_norm_stats.json", "w") as f:
    json.dump(out, f, indent=2)
print("written:", s.shape[0], "frames")
EOF
```

## 4. 训练(本分支环境)

单卡 80GB(B=3 是实测单卡上限;显存更小就降 batch_size):

```bash
lerobot-train \
  --dataset.repo_id=<user>/rebot_pick_place \
  --dataset.root=~/.cache/huggingface/lerobot/<user>_rebot_pick_place_<时间戳> \
  --dataset.streaming=false \
  --policy.type=lingbot_vla_v2 \
  --policy.pretrained_path=~/lvla_scratch/robotwin-6b-lerobot \
  --policy.robot_config_path=~/rebot_lingbot/rebot_b601_robot_config.yaml \
  --policy.norm_stats_path=~/rebot_lingbot/rebot_b601_norm_stats.json \
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
  --output_dir=~/output_lerobot_train/rebot/lingbot_v2_A \
  --job_name=rebot_lingbot_v2_A \
  --wandb.enable=false
```

多卡:`accelerate launch --num_processes=4 $(which lerobot-train) ... --batch_size=2`
(**batch_size 是每卡语义**;DDP 每卡上限 2,3 会 OOM)。

纪律(都来自实测,别踩):

- `robot_config_path` / `norm_stats_path` **必须显式传**(robotwin ckpt 内嵌的是 robotwin 相机名)。
- **不要** `--policy.sdpa_backend=CUDNN_ATTENTION`(torch 2.8 下 backward 产 NaN,已否决)。
- B≤4 **不要**开 `--policy.gradient_checkpointing=true`(+38% 慢)。
- 吞吐锚点:单卡 B=3 ≈ 1.99 samples/s,30k 步 ≈ 12.6h;4 卡 B=2/卡 ≈ 8.0 samples/s ≈ 8.3h。

## 5. bake 推理开关 + rollout 上机

### 5a. 一次性固化(训练完做一次)

```bash
python3 - << 'EOF'
import json, pathlib
p = pathlib.Path.home() / "output_lerobot_train/rebot/lingbot_v2_A/checkpoints/last/pretrained_model/config.json"
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

固化后 e2e 时延:robotwin 3 相机 436ms;相机更少会更低。首次推理有一次性编译(几分钟),
之后稳定。**注意固化是在训练机上做的;ckpt 拷到随臂机器(Jetson/工作站)后配置随之生效,
但 compile 缓存会在新机器上重新编一次。**

### 5b. rollout(reBot B601-DM)

```bash
lerobot-rollout \
  --policy.path=~/output_lerobot_train/rebot/lingbot_v2_A/checkpoints/last/pretrained_model \
  --robot.type=rebot_b601_follower \
  --robot.port=can0 \
  --robot.can_adapter=socketcan \
  --robot.id=follower1 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/v4l/by-id/<front-cam-id>, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/v4l/by-id/<orbbec-id>, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --strategy.type=base \
  --inference.type=sync \
  --task="pick up the black object and place it in the box" \
  --fps=30 \
  --duration=60 \
  --display_data=false \
  --play_sounds=false
```

要同时录 eval 数据,追加:

```bash
  --dataset.repo_id=<user>/eval_rebot_lingbot_v2 \
  --dataset.single_task="pick up the black object and place it in the box" \
  --dataset.push_to_hub=false
```

要点:

- **相机 key 必须叫 `front` / `wrist`**,和 robot_config 的 origin_keys 一字不差;
  模型内部再映射到 canonical 槽位。
- 达妙串口桥则 `--robot.port=/dev/ttyACM0`(默认 `can_adapter=damiao`,可省掉
  `--robot.can_adapter`);SocketCAN/PCAN 就用 `can0` + `--robot.can_adapter=socketcan`。
- **task 文本必须和采集时的 TASK 一致**(语言条件进模型)。
- 第一次上机先 `--duration=10` 短跑确认动作方向/归一化正确,再上长任务。
- Orbbec 在 rollout 侧当普通 UVC 相机用(只要 RGB);深度目前不进策略(见 §7)。

## 6. 采集仓 ↔ 本分支对应表

| 项 | rebot_datacollect(采集) | pr3967-combined(训练/rollout) |
|---|---|---|
| robot type | `rebot_follower`(B601-RS 驱动) | `rebot_b601_follower`(B601-DM 驱动) |
| teleop | `starai_to_rebot_leader` | rollout 不需要 teleop |
| 数据集格式 | LeRobot v2.1,可直接被本分支 `LeRobotDataset` 读 | 同左 |
| 状态/动作维度 | 7(6 关节 + 夹爪) | 映射进 14 维 `arm.position` 槽位 [0:7] |
| 相机 | front / wrist / wrist_depth | front + wrist 进策略;depth 暂不用 |

## 7. 已知边界 / 后续项

- **wrist_depth 暂未进策略**:feature_transform 有 `use_depth_align` 通路但本分支未在
  reBot 数据上验证过;v1 先只用 RGB,深度留着做后续实验。
- 两路相机训出来的模型,`camera_wrist_right` 全程 mask,推理同样 mask,行为一致;
  若之后加第三路相机,需重新采集含该视角的数据再训(缺位 mask 不等于模型会用该视角)。
- B601-RS(RobStride)臂的 rollout 驱动移植是独立工作项,见 §0.1。
- 30k 步 ≈ 3.6 个 epoch(global batch 16、50 段 × 15s × 30fps ≈ 22.5k 帧的数据集规模下
  更多 epoch;样本量小可以降 steps,5k 一存看 eval 曲线再定)。
