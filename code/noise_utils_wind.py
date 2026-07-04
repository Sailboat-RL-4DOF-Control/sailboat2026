"""Fixed-scale sensor noise for the 16-dimensional wind observation."""

from __future__ import annotations

import numpy as np

from noise_utils_nowind import ActuatorNoise, FixedSensorNoise


class FixedWindSensorNoise(FixedSensorNoise):
    WIND_SPEED_STD_MPS = 0.1
    WIND_DIRECTION_STD_RAD = np.deg2rad(1.0)

    def add_observation_noise(self, state: np.ndarray) -> np.ndarray:
        if np.asarray(state).shape != (16,):
            raise ValueError(f"Wind noise expects shape (16,), got {np.asarray(state).shape}")
        noisy = self._add_base_noise(state)
        # Environment contract: index 14 is wind speed; index 15 is direction.
        noisy[14] = max(0.0, noisy[14] + self.rng.normal(0.0, self.WIND_SPEED_STD_MPS))
        noisy[15] += self.rng.normal(0.0, self.WIND_DIRECTION_STD_RAD)
        noisy[15] = np.arctan2(np.sin(noisy[15]), np.cos(noisy[15]))
        return noisy.astype(np.float32)


class NoiseManager:
    def __init__(
        self,
        actuator_noise_ratio: float = 0.03,
        enable_sensor_noise: bool = True,
        enable_actuator_noise: bool = True,
        seed: int | None = None,
        **_: object,
    ):
        self.enable_sensor_noise = bool(enable_sensor_noise)
        self.enable_actuator_noise = bool(enable_actuator_noise)
        self.sensor_noise = FixedWindSensorNoise(seed=seed)
        self.actuator_noise = ActuatorNoise(
            noise_ratio=actuator_noise_ratio,
            seed=None if seed is None else seed + 1000,
        )

    def add_observation_noise(self, state: np.ndarray) -> np.ndarray:
        return self.sensor_noise.add_observation_noise(state) if self.enable_sensor_noise else state

    def add_action_noise(self, action: np.ndarray, max_action: np.ndarray) -> np.ndarray:
        return self.actuator_noise.apply_action_noise(action, max_action) if self.enable_actuator_noise else action
