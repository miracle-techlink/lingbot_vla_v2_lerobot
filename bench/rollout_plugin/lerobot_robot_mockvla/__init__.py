"""Headless mock robot plugin for lerobot-rollout smoke tests.

Registers a ``mock_vla`` robot type that needs no hardware: 7 reBot-style
positional joints plus two synthetic RGB cameras (front / wrist) matching the
shakehands LingBot-VLA 2.0 checkpoint's expected visual inputs.

Discovered by ``register_third_party_plugins`` because the distribution name
starts with ``lerobot_robot_``.  Also installs a logging wrapper around
``LingbotVLAV2Policy.predict_action_chunk`` so the action-chunk shape appears
in the rollout log (validation evidence only; pass-through wrapper).
"""

import logging
from dataclasses import dataclass

import numpy as np

from lerobot.robots import Robot, RobotConfig

logger = logging.getLogger(__name__)

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


@RobotConfig.register_subclass("mock_vla")
@dataclass(kw_only=True)
class MockVlaConfig(RobotConfig):
    image_height: int = 224
    image_width: int = 224


class MockVla(Robot):
    """Mock robot producing synthetic observations and accepting any action."""

    config_class = MockVlaConfig
    name = "mock_vla"

    def __init__(self, config: MockVlaConfig):
        super().__init__(config)
        self.config = config
        self.cameras: dict = {}
        self._connected = False
        self._pos = {f"{m}.pos": 0.0 for m in MOTOR_NAMES}
        self._tick = 0
        self._n_actions = 0
        h, w = config.image_height, config.image_width
        yy, xx = np.mgrid[0:h, 0:w]
        # Static gradient base; rolled per observation so frames differ.
        self._base = np.stack(
            [xx % 256, yy % 256, ((xx + yy) // 2) % 256], axis=-1
        ).astype(np.uint8)

    # -- features ------------------------------------------------------
    @property
    def observation_features(self) -> dict:
        fts = {k: float for k in self._pos}
        h, w = self.config.image_height, self.config.image_width
        fts["front"] = (h, w, 3)
        fts["wrist"] = (h, w, 3)
        return fts

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(self._pos, float)

    # -- lifecycle -----------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        self._connected = True
        logger.info("mock_vla connected (no hardware)")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
        logger.info("mock_vla disconnected")

    # -- IO ------------------------------------------------------------
    def get_observation(self) -> dict:
        if not self._connected:
            raise RuntimeError("mock_vla is not connected")
        self._tick += 1
        img = np.roll(self._base, self._tick % self._base.shape[1], axis=1)
        obs = dict(self._pos)
        obs["front"] = img
        obs["wrist"] = np.flipud(img).copy()
        return obs

    def send_action(self, action: dict) -> dict:
        if not self._connected:
            raise RuntimeError("mock_vla is not connected")
        for k, v in action.items():
            if k in self._pos:
                self._pos[k] = float(v)
        self._n_actions += 1
        if self._n_actions == 1 or self._n_actions % 25 == 0:
            logger.info(
                "mock_vla send_action #%d: keys=%d sample=%s",
                self._n_actions,
                len(action),
                {k: round(float(v), 3) for k, v in list(action.items())[:3]},
            )
        return action


def _patch_chunk_shape_logger() -> None:
    """Log LingbotVLAV2Policy.predict_action_chunk output shape (smoke evidence)."""
    try:
        from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
            LingbotVLAV2Policy,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("chunk-shape logger not installed: %s", exc)
        return
    orig = LingbotVLAV2Policy.predict_action_chunk

    # Pass-through wrapper: keep the wrapped signature visible (functools.wraps
    # → __wrapped__) so RTC's inspect.signature support check still binds
    # inference_delay/prev_chunk_left_over, and forward all call shapes.
    import functools

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        logger.info(
            "predict_action_chunk produced chunk shape=%s dtype=%s",
            tuple(out.shape),
            out.dtype,
        )
        return out

    LingbotVLAV2Policy.predict_action_chunk = wrapped
    logger.info("Installed predict_action_chunk shape logger")


_patch_chunk_shape_logger()
