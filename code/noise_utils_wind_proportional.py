"""Optional proportional sensor noise for the 16-D wind observation."""

import numpy as np

from noise_utils_nowind import ActuatorNoise


class ProportionalSensorNoise:
    def __init__(self, noise_ratio=0.03, seed=None):
        self.noise_ratio = float(noise_ratio)
        self.rng = np.random.RandomState(seed)
        self.ranges = {
            "position": (0.5, 5.0),
            "angle": (np.deg2rad(0.1), np.deg2rad(2.0)),
            "velocity": (0.02, 0.3),
            "angular_velocity": (np.deg2rad(0.05), np.deg2rad(0.5)),
            "distance": (0.5, 5.0),
            "wind_speed": (0.0, np.inf),
            "wind_direction": (0.0, np.inf),
        }

    def _add(self, value, sensor_type):
        low, high = self.ranges[sensor_type]
        raw = abs(float(value)) * self.noise_ratio * self.rng.randn()
        magnitude = np.clip(abs(raw), low, high)
        noise = np.sign(raw) * magnitude if raw != 0.0 else 0.0
        return float(value) + noise

    def add_observation_noise(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (16,):
            raise ValueError(f"Wind noise expects shape (16,), got {state.shape}")
        noisy = state.copy()
        noisy[0] = self._add(state[0], "position")
        noisy[1] = self._add(state[1], "position")
        noisy[2] = self._add(state[2], "angle")
        noisy[3] = self._add(state[3], "angle")
        noisy[3] = np.arctan2(np.sin(noisy[3]), np.cos(noisy[3]))
        noisy[4] = self._add(state[4], "velocity")
        noisy[5] = self._add(state[5], "velocity")
        noisy[6] = self._add(state[6], "angular_velocity")
        noisy[7] = self._add(state[7], "angular_velocity")
        noisy[10] = self._add(state[10], "distance")
        noisy[11] = self._add(state[11], "distance")
        # Wind environment contract: speed at index 14, direction at index 15.
        noisy[14] = max(0.0, self._add(state[14], "wind_speed"))
        noisy[15] = self._add(state[15], "wind_direction")
        noisy[15] = np.arctan2(np.sin(noisy[15]), np.cos(noisy[15]))
        return noisy.astype(np.float32)


class NoiseManager:
    def __init__(
        self,
        sensor_noise_ratio=0.03,
        actuator_noise_ratio=0.03,
        enable_sensor_noise=True,
        enable_actuator_noise=True,
        seed=None,
    ):
        self.enable_sensor_noise = bool(enable_sensor_noise)
        self.enable_actuator_noise = bool(enable_actuator_noise)
        self.sensor_noise = ProportionalSensorNoise(sensor_noise_ratio, seed)
        self.actuator_noise = ActuatorNoise(
            actuator_noise_ratio,
            None if seed is None else seed + 1000,
        )

    def add_observation_noise(self, state):
        if self.enable_sensor_noise:
            return self.sensor_noise.add_observation_noise(state)
        return state

    def add_action_noise(self, action, max_action):
        if self.enable_actuator_noise:
            return self.actuator_noise.apply_action_noise(action, max_action)
        return action
