# LingBot-VLA 2.0 优化开关表 + 逐项启动验证

> **一句话定位(2026-08-14 锁定,sdpa 全程保留,从未被否决):**
> **训练** = sdpa(自动后端)+ bf16 + 稠密双 GEMM MoE(默认 512)+ fused AdamW(`optimizer_fused=true`)+ 不开 grad-ckpt;单卡 B=3 / DDP 每卡 B=2。
> **推理** = sdpa + bf16 + `compile_predict_velocity`(max-autotune-no-cudagraphs)+ `compile_prefix` + `preprocess_device=cuda` + `num_steps=10`,bake 进 ckpt。
> 被否决的只是:显式设 `sdpa_backend`(CUDNN 训练 NaN / EFFICIENT 无 kernel)、eager 训练(真实 A/B:+2.2% 慢、+2.9GB、B=4 OOM)、`attention_fp32`(+8~14% 慢)。

> 分支 `pr3967-combined`,2026-08-13。
> 验证环境:A100-80GB ×8(liuyue 机),conda `lvla-lerobot`,robotwin 6B 转换 ckpt +
> gpudad SO101 数据集(993 段 / 132k 帧 / 3 相机)。
> 验证方式:训练开关走 **`lerobot-train` 真启动**(steps=3, batch_size=1, 真实数据集);
> 推理开关走 **ckpt 加载 + `predict_action_chunk`**(rollout 的策略路径,减去硬件);
> I6 额外完整复刻 rollout 加载方式(开关固化进 config.json 后 `from_pretrained`)。
> 驱动脚本:`bench/run_switch_smoke.sh`(A100: `~/accel-bench/switch_smoke/`)。

## 1. 训练开关(`lerobot-train --policy.*`)

| 开关 | 默认 | 作用 | 启动验证 | 数值/性能结论 |
|---|---|---|---|---|
| `optimizer_fused` | false | fused AdamW 单 kernel step | ✅ exit 0,3 步 loss 有限 | loss 0.429/0.446/2.445 ≈ 对照组(-5.4% step 时间,此前 e2e 实测) |
| `gradient_checkpointing` | false | 重算激活省显存 | ✅ exit 0,loss 与对照逐位一致 | B≤4 时 +38% 慢;**显存够别开** |
| `moe_dense_max_tokens` | 512 | dense 双 GEMM MoE 阈值,0=关闭回逐专家循环 | ✅ exit 0(loss 0.451/… 路径不同属预期) | 默认开即是优化后状态;0 用于回退对照 |
| `sdpa_backend=EFFICIENT_ATTENTION` | null(自动) | 强制 mem-efficient SDPA 后端 | ❌ **起不来**:`RuntimeError: No available kernel`(该后端不支持本模型 bool mask+GQA+head_dim128 组合) | 不要设;自动选择已最优 |
| `sdpa_backend=CUDNN_ATTENTION` | null(自动) | 强制 cuDNN SDPA 后端 | ⚠️ **能启动但训练即毁**:exit 0、前向正常,但 step 1 梯度 `grdn:nan`,step 2 起 loss=nan(lerobot-train 实测复现;单步梯度对照 1222/1267 NaN) | **❌ 禁用**,torch 2.8 bf16+bool mask+GQA backward 缺陷 |

## 2. 推理开关(固化进 ckpt config.json,`lerobot-rollout` 生效)

| 开关 | 默认 | 作用 | 启动验证 | 结论 |
|---|---|---|---|---|
| `compile_predict_velocity` | false | torch.compile 去噪步 | ✅ exit 0 | 推理主加速项(编译需 ≥3 次 warmup;冒烟 warmup 不足时均值被编译时间污染,看 I6 稳态) |
| `compile_predict_velocity_mode` | default | inductor 模式 | ✅(随 I2/I3/I6) | 锁定 `max-autotune-no-cudagraphs` |
| `compile_prefix` | false | compile vision+prefix 段 | ✅ exit 0(I3 全套栈) | 需先开 compile_predict_velocity |
| `preprocess_device` | null(CPU) | GPU 图像预处理 | ✅ exit 0(预处理 519→198ms) | 仅推理;训练自动留 CPU |
| `num_steps` | 10 | 去噪步数 | ✅ exit 0(I4,steps=5) | 锁定 10;改小线性降时延但掉质量 |
| `moe_dense_max_tokens` | 512 | 推理 MoE 路径 | ✅ exit 0(关掉后 911→1130ms,符合回退预期) | 0 可回退逐专家路径对照 |

## 3. 端到端组合验证(rollout 完整加载路径)

| 验证 | 内容 | 结果 |
|---|---|---|
| I6 baked-config | 把 §2 全套开关固化进 config.json → `from_pretrained` → 3 次 `predict_action_chunk` | ✅ exit 0:iter0 43.7s(一次性编译)→ 稳态 **258ms**,action (1,50,14) 有限 |

## 4. 推荐组合(已被验证矩阵覆盖)

**训练**:全部默认 + `--policy.optimizer_fused=true`,单卡 B=3 / 多卡 B=2每卡。
**推理**:`compile_predict_velocity + max-autotune-no-cudagraphs + compile_prefix +
preprocess_device=cuda + num_steps=10`,e2e 436ms(robotwin 3 相机)/ ~300ms(单相机)。

> 验证日志逐条见 A100 `~/accel-bench/switch_smoke/*.log`。
> 矩阵总账:13 臂,11 ✅ 启动通过;2 个负面结论同样由启动实测支撑——
> `sdpa_backend=EFFICIENT_ATTENTION` 起不来(No available kernel)、
> `sdpa_backend=CUDNN_ATTENTION` 能启动但 step 1 梯度即 NaN。
> 结论:**`sdpa_backend` 整体不要设**(留作调试用),自动后端就是最优且唯一安全选择。
