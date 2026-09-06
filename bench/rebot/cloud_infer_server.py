#!/usr/bin/env python3
"""云端推理服务 (A100): ZeroMQ REP 包裹 lingbot_vla_v2 policy, 供真机端 RemotePolicy 调用.

协议 (pickle over ZMQ REQ/REP):
  请求: {
    "state":    np.float32 [55]         # 原始关节状态 (canonical 55 维)
    "front":    bytes JPEG              # 相机帧 (front/cam_high)
    "wrist":    bytes JPEG              # 相机帧 (wrist/cam_left_wrist)
    "task":     str
    "delay":    int                     # RTC inference_delay
    "prev":     np.float32 [T,55]|None  # RTC 归一化 leftover prefix
    "norm_stats": bool                  # False: state 为原始单位, 服务端跑完整 preprocessor
  }
  响应: {
    "chunk": np.float32 [50, 7]         # 反归一化 action chunk (原始单位)
    "norm":  np.float32 [50, 55]        # 归一化 chunk (RTC guidance 引用)
    "lat_ms": float                     # 服务端纯推理延迟
    "error": str|None
  }

用法 (A100 GPU6):
  CUDA_VISIBLE_DEVICES=6 python bench/rebot/cloud_infer_server.py \
    --ckpt /home/liuyue/accel-bench/rebot/merged_28000 \
    --tokenizer /home/liuyue/lvla_scratch/Qwen3-VL-4B-Instruct \
    --robot-config /home/liuyue/rebot_port/rebot_lerobot.yaml \
    --norm-stats /home/liuyue/rebot_port/rebot_norm_stats.json \
    --port 5557
"""
import argparse
import io
import pickle
import time

import numpy as np
import torch
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--robot-config", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--compile", action="store_true", help="enable torch.compile prefix/velocity")
    ap.add_argument("--cudagraph", action="store_true", help="enable CUDA-graph denoise loop (unguided)")
    ap.add_argument("--cpu-preprocess", action="store_true",
                    help="keep image preprocessing on CPU (default: GPU, C4 prepare_images_on_device)")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override flow-matching denoise steps (ckpt bake 是 4; 7=原 bake 档, 动作更平滑)")
    args = ap.parse_args()

    import lerobot.policies  # noqa: F401  # import side effect: 注册全部 policy config 到 ChoiceRegistry

    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cuda"
    cfg.dtype = "bfloat16"
    cfg.tokenizer_path = args.tokenizer
    cfg.processor_path = args.tokenizer
    cfg.robot_config_path = args.robot_config
    cfg.norm_stats_path = args.norm_stats
    if args.cudagraph:
        cfg.use_cudagraph_denoise = True
    # C4: 图像预处理搬上 GPU (prepare_images_on_device, 复刻 HF 算子但跳过 Python 包装).
    # 130ms 基准的 bake 配置之一; CPU 路径在 4090 上实测多 ~100-150ms.
    cfg.preprocess_device = "cpu" if args.cpu_preprocess else "cuda"
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    # 与 rollout 链路同款加载路径: from_pretrained 不需要 ds_meta/env
    # (lingbot_vla_v2 的输入输出形状由自身 config 的 robot_config_path 决定).
    policy = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg)
    policy.eval()
    overrides = {"device_processor": {"device": "cuda"}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.ckpt, preprocessor_overrides=overrides,
    )
    if args.compile:
        policy.model._use_compile_prefix = True
        policy.model._use_compile_predict_velocity = True

    import cv2

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Serving on :{args.port} (compile={args.compile} cudagraph={args.cudagraph})", flush=True)

    # 预处理管线的图像/状态键 (由 robot_config 决定; rebot 双相机 + 55 维状态)
    from lerobot.utils.feature_utils import build_dataset_frame

    hw_features = None  # built lazily from first request's shapes via preprocessor features

    while True:
        try:
            req = pickle.loads(sock.recv())
            t0 = time.perf_counter()
            if req.get("shutdown"):
                sock.send(pickle.dumps({"error": None, "bye": True}))
                break

            # imdecode 出来是 BGR; 相机侧 obs 张量是 RGB — 转回 RGB 保证与本地
            # rollout 路径喂给 preprocessor 的张量逐位一致 (真机色彩正确性).
            front = cv2.cvtColor(cv2.imdecode(np.frombuffer(req["front"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            wrist = cv2.cvtColor(cv2.imdecode(np.frombuffer(req["wrist"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # 数据集同款源键名: 管线内的 rename 步骤负责 front→cam_high / wrist→cam_left_wrist
            obs = {
                "observation.images.front": torch.from_numpy(front.transpose(2, 0, 1)[None].copy()),
                "observation.images.wrist": torch.from_numpy(wrist.transpose(2, 0, 1)[None].copy()),
                "observation.state": torch.from_numpy(np.asarray(req["state"], np.float32)[None].copy()),
                "task": [req["task"]],
            }

            obs = preprocessor(obs)
            t_pre = time.perf_counter()

            prev = None
            if req.get("prev") is not None:
                prev = torch.as_tensor(req["prev"], dtype=torch.bfloat16, device="cuda").unsqueeze(0)

            # 注意: 不包 torch.inference_mode()/no_grad() — RTC guidance 路径
            # (prev_chunk_left_over != None 时) 需要 autograd, 与本地 RTC 引擎一致.
            chunk = policy.predict_action_chunk(
                obs, inference_delay=int(req.get("delay", 0)), prev_chunk_left_over=prev
            )
            norm = policy.get_last_normalized_chunk()
            t_model = time.perf_counter()
            out_chunk = postprocessor(chunk.reshape(-1, chunk.shape[-1]).unsqueeze(1))
            out_chunk = out_chunk.reshape(chunk.shape).squeeze(0).float().cpu().numpy()
            torch.cuda.synchronize()
            t_post = time.perf_counter()

            sock.send(pickle.dumps({
                "chunk": out_chunk,
                "norm": norm.squeeze(0).float().cpu().numpy(),
                "lat_ms": (t_post - t0) * 1000,
                # 分段 (ms): decode+预处理 / 模型 forward / 后处理+同步
                "seg_ms": {
                    "prep": (t_pre - t0) * 1000,
                    "model": (t_model - t_pre) * 1000,
                    "post": (t_post - t_model) * 1000,
                },
                "error": None,
            }))
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                sock.send(pickle.dumps({"chunk": None, "norm": None, "lat_ms": 0, "error": str(e)}))
            except Exception:
                pass


if __name__ == "__main__":
    main()
