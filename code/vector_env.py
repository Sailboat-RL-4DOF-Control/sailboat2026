"""Small helper for optional single- or multi-process training environments."""

import platform

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env_factory(env_class, seed, maximum_steps, env_kwargs=None):
    def _make():
        import globals as sail_globals

        sail_globals.total_time = int(maximum_steps)
        env = env_class(**(env_kwargs or {}))
        env.reset(seed=int(seed))
        return env

    return _make


def build_vector_env(env_class, num_envs, base_seed, maximum_steps, env_kwargs=None):
    factories = [
        make_env_factory(
            env_class,
            int(base_seed) + index,
            maximum_steps,
            env_kwargs,
        )
        for index in range(int(num_envs))
    ]
    if int(num_envs) == 1:
        return DummyVecEnv(factories)
    start_method = "spawn" if platform.system().lower().startswith("win") else "forkserver"
    return SubprocVecEnv(factories, start_method=start_method)
