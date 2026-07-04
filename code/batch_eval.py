"""Batch-test all released policies with automatic dimension matching.

The actor weights determine FC/LSTM structure and the observation dimension:
14 dimensions select the nowind environment, while 16 select the wind
environment. Run with ``python batch_eval.py`` or click Run in an IDE.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from noise_utils_nowind import NoiseManager as NowindNoise
from noise_utils_wind import NoiseManager as WindNoise
from sailboat_s14_nowind import SailboatNowindEnv
from sailboat_s14_wind import SailboatWindEnv
from utils_fc import Actor as FcActor
from utils_lstm import Actor as LstmActor


SCRIPT_DIR = Path(__file__).resolve().parent


def load_weights(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class Policy:
    def __init__(self, actor_path, device="cpu"):
        self.actor_path = Path(actor_path)
        self.device = torch.device(device)
        weights = load_weights(self.actor_path, self.device)

        if "lstm.weight_ih_l0" in weights:
            self.kind = "lstm"
            self.state_dim = int(weights["current_state_fc.0.weight"].shape[1])
            self.action_dim = int(weights["mu_layer.weight"].shape[0])
            hidden = int(weights["lstm.weight_hh_l0"].shape[1])
            final = int(weights["final_fc.0.weight"].shape[0])
            self.actor = LstmActor(self.state_dim, self.action_dim, (hidden, final))
        elif "fc.0.weight" in weights:
            self.kind = "fc"
            self.state_dim = int(weights["fc.0.weight"].shape[1])
            self.action_dim = int(weights["mu_layer.weight"].shape[0])
            keys = sorted(
                [key for key in weights if key.startswith("fc.") and key.endswith(".weight")],
                key=lambda key: int(key.split(".")[1]),
            )
            self.actor = FcActor(
                self.state_dim,
                self.action_dim,
                tuple(int(weights[key].shape[0]) for key in keys),
            )
        else:
            raise ValueError(f"Unsupported actor structure: {self.actor_path}")

        if self.state_dim not in (14, 16):
            raise ValueError(f"No environment is defined for state_dim={self.state_dim}")
        self.actor.load_state_dict(weights)
        self.actor.to(self.device).eval()

        normalizer_path = self.actor_path.with_name("normalizer.pth")
        normalizer = load_weights(normalizer_path, self.device)
        self.mean = normalizer["mean"].to(self.device)
        self.var = normalizer["var"].to(self.device)
        self.previous_state = np.zeros(self.state_dim, dtype=np.float32)
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)

    @property
    def name(self):
        return self.actor_path.parent.name

    def normalize(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return (tensor - self.mean) / (torch.sqrt(self.var) + 1e-6)

    def reset(self, state):
        self.previous_state = np.asarray(state, dtype=np.float32).copy()
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        if hasattr(self.actor, "hidden_states"):
            self.actor.hidden_states.clear()

    def act(self, state):
        state = np.asarray(state, dtype=np.float32)
        with torch.no_grad():
            if self.kind == "fc":
                action, _ = self.actor(
                    current_state=self.normalize(state),
                    deterministic=True,
                    with_logprob=False,
                )
            else:
                action, _ = self.actor(
                    current_state=self.normalize(state),
                    prev_state=self.normalize(self.previous_state),
                    prev_action=torch.as_tensor(
                        self.previous_action,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0),
                    deterministic=True,
                    with_logprob=False,
                    env_ids=[0],
                    single_step_mode=True,
                )
        result = action.cpu().numpy()[0].astype(np.float32)
        self.previous_state = state.copy()
        self.previous_action = result.copy()
        return result


def parse_directions(text):
    if text.strip().lower() == "random":
        return [None]
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def run_episode(policy, seed, direction, max_steps, use_noise):
    if policy.state_dim == 16:
        env = SailboatWindEnv()
        noise = WindNoise(
            enable_sensor_noise=use_noise,
            enable_actuator_noise=use_noise,
            seed=seed,
        )
        environment = "wind"
    else:
        env = SailboatNowindEnv()
        noise = NowindNoise(
            enable_sensor_noise=use_noise,
            enable_actuator_noise=use_noise,
            seed=seed,
        )
        environment = "nowind"

    options = None if direction is None else {"initial_wind_direction_deg": direction}
    raw_state, info = env.reset(seed=seed, options=options)
    state = noise.add_observation_noise(raw_state)
    policy.reset(state)
    positions = [np.asarray(raw_state[:2], dtype=float)]
    total_reward = 0.0
    terminated = truncated = False
    steps = 0
    try:
        for steps in range(1, max_steps + 1):
            action = policy.act(state)
            action = noise.add_action_noise(action, env.action_space.high)
            raw_state, reward, terminated, truncated, _ = env.step(
                action * env.action_space.high
            )
            positions.append(np.asarray(raw_state[:2], dtype=float))
            total_reward += float(reward)
            state = noise.add_observation_noise(raw_state)
            if terminated or truncated:
                break
    finally:
        env.close()

    positions = np.asarray(positions)
    return {
        "model": policy.name,
        "architecture": policy.kind,
        "state_dim": policy.state_dim,
        "environment": environment,
        "seed": seed,
        "requested_wind_deg": "random" if direction is None else direction,
        "actual_initial_wind_deg": info.get("initial_wind_deg", ""),
        "success": bool(terminated),
        "steps": steps,
        "path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "final_distance_m": float(np.linalg.norm(positions[-1] - np.array([500.0, 0.0]))),
        "total_reward": total_reward,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    actor_paths = sorted(Path(args.model_dir).glob("*/actor.pth"))
    if not actor_paths:
        raise FileNotFoundError(f"No actor.pth files found in {args.model_dir}")
    policies = [Policy(path, args.device) for path in actor_paths]
    for policy in policies:
        print(
            f"{policy.name}: {policy.kind}, state_dim={policy.state_dim}, "
            f"environment={'wind' if policy.state_dim == 16 else 'nowind'}"
        )

    rows = []
    directions = parse_directions(args.wind_directions)
    for model_index, policy in enumerate(policies):
        for direction_index, direction in enumerate(directions):
            for episode in range(args.episodes):
                seed = args.base_seed + model_index * 100000 + direction_index * 10000 + episode
                row = run_episode(policy, seed, direction, args.max_steps, args.noise)
                rows.append(row)
                print(
                    f"{row['model']} seed={seed} wind={row['requested_wind_deg']} "
                    f"success={row['success']} steps={row['steps']}"
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "episodes.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["requested_wind_deg"])].append(row)
    summary = []
    for (model, direction), group in grouped.items():
        summary.append(
            {
                "model": model,
                "wind_deg": direction,
                "episodes": len(group),
                "successes": sum(row["success"] for row in group),
                "success_rate": sum(row["success"] for row in group) / len(group),
                "mean_final_distance_m": float(np.mean([row["final_distance_m"] for row in group])),
            }
        )
    write_csv(output_dir / "summary.csv", summary)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(SCRIPT_DIR / "model"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--wind-directions", default="random")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=4000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "test_results"))
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
