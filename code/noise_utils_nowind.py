"""Fixed-scale sensor noise for the 14-dimensional observation.

Unlike proportional noise, each sensor has a value-independent Gaussian
standard deviation. The magnitudes follow the fixed settings used by the
original nowind experiments.
"""

from __future__ import annotations

import numpy as np


class FixedSensorNoise:
    POSITION_STD_M = 1.0
    ANGLE_STD_RAD = np.deg2rad(1.0)
    VELOCITY_STD_MPS = 0.1
    ANGULAR_VELOCITY_STD_RADPS = np.deg2rad(0.1)
    DISTANCE_STD_M = 0.5

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def _add_base_noise(self, state: np.ndarray) -> np.ndarray:
        noisy = np.asarray(state, dtype=np.float32).copy()
        noisy[0:2] += self.rng.normal(0.0, self.POSITION_STD_M, size=2)
        noisy[2] += self.rng.normal(0.0, self.ANGLE_STD_RAD)
        noisy[3] += self.rng.normal(0.0, self.ANGLE_STD_RAD)
        noisy[3] = np.arctan2(np.sin(noisy[3]), np.cos(noisy[3]))
        noisy[4:6] += self.rng.normal(0.0, self.VELOCITY_STD_MPS, size=2)
        noisy[6:8] += self.rng.normal(0.0, self.ANGULAR_VELOCITY_STD_RADPS, size=2)
        # Indices 8:10 are measured actuator angles and remain unchanged.
        noisy[10:12] += self.rng.normal(0.0, self.DISTANCE_STD_M, size=2)
        # Indices 12:14 are the known target position and remain unchanged.
        return noisy

    def add_observation_noise(self, state: np.ndarray) -> np.ndarray:
        if np.asarray(state).shape != (14,):
            raise ValueError(f"Nowind noise expects shape (14,), got {np.asarray(state).shape}")
        return self._add_base_noise(state)


class ActuatorNoise:
    """Dead zone, quantization, response lag, and bounded execution error."""

    def __init__(self, noise_ratio: float = 0.03, seed: int | None = None):
        self.noise_ratio = float(noise_ratio)
        self.rng = np.random.RandomState(seed)
        self.params = (
            {
                "deadzone": np.deg2rad(0.5),
                "resolution": np.deg2rad(0.2),
                "max_error": np.deg2rad(1.5),
                "response_factor": 0.95,
                "large_action_threshold": 0.7,
                "large_action_error_mult": 1.5,
            },
            {
                "deadzone": np.deg2rad(0.3),
                "resolution": np.deg2rad(0.15),
                "max_error": np.deg2rad(1.0),
                "response_factor": 0.97,
                "large_action_threshold": 0.7,
                "large_action_error_mult": 1.3,
            },
        )

    def _one(self, normalized_action: float, limit: float, params: dict) -> float:
        command = float(normalized_action) * float(limit)
        if abs(command) < params["deadzone"]:
            return 0.0
        command = round(command / params["resolution"]) * params["resolution"]
        error = abs(command) * self.noise_ratio * self.rng.randn()
        if abs(command) / float(limit) > params["large_action_threshold"]:
            error *= params["large_action_error_mult"]
        error = np.clip(error, -params["max_error"], params["max_error"])
        response = params["response_factor"] * (1.0 + (self.rng.rand() - 0.5) * 0.1)
        return float(np.clip((command * response + error) / float(limit), -1.0, 1.0))

    def apply_action_noise(self, action: np.ndarray, max_action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        limits = np.asarray(max_action, dtype=np.float32)
        if action.shape != (2,) or limits.shape != (2,):
            raise ValueError("Sail and rudder actions must both have shape (2,)")
        return np.asarray(
            [self._one(action[i], limits[i], self.params[i]) for i in range(2)],
            dtype=np.float32,
        )


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
        self.sensor_noise = FixedSensorNoise(seed=seed)
        self.actuator_noise = ActuatorNoise(
            noise_ratio=actuator_noise_ratio,
            seed=None if seed is None else seed + 1000,
        )

    def add_observation_noise(self, state: np.ndarray) -> np.ndarray:
        return self.sensor_noise.add_observation_noise(state) if self.enable_sensor_noise else state

    def add_action_noise(self, action: np.ndarray, max_action: np.ndarray) -> np.ndarray:
        return self.actuator_noise.apply_action_noise(action, max_action) if self.enable_actuator_noise else action
