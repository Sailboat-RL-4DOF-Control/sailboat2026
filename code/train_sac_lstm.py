"""Train SAC-LSTM in the 14-dimensional environment.

Reward ablations use the same model. Set ``--reward-ablation`` to
``no_progress``, ``no_action``, ``no_checkpoint``, or ``no_arrival``. This
file can also be launched directly from an IDE.
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from SAC_lstm import SAC_lstm_countinuous
from noise_utils_nowind import NoiseManager
from sailboat_s14_nowind import SailboatNowindEnv
from vector_env import build_vector_env


SCRIPT_DIR = Path(__file__).resolve().parent


def save_checkpoint(agent, output_dir, step):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), output_dir / f"actor_{step}.pth")
    torch.save(agent.q_critic.state_dict(), output_dir / f"critic_{step}.pth")
    torch.save(agent.state_normalizer.state_dict(), output_dir / f"normalizer_{step}.pth")


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    env = build_vector_env(
        SailboatNowindEnv,
        args.num_parallel_envs,
        args.seed,
        args.maximum_steps_per_episode,
        {"reward_ablation": args.reward_ablation},
    )
    max_action = env.action_space.high.astype(np.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["state_dim"] = 14
    config["action_dim"] = 2
    reward_terms = {
        "progress": True,
        "action": True,
        "checkpoint": True,
        "arrival": True,
        "time": True,
    }
    if args.reward_ablation != "full":
        reward_terms[args.reward_ablation.removeprefix("no_")] = False
    config["reward_terms"] = reward_terms
    config["output_dir"] = str(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    agent = SAC_lstm_countinuous(
        state_dim=14,
        action_dim=2,
        dvc=device,
        net_width=args.recurrent_hidden_size,
        a_lr=args.actor_learning_rate,
        c_lr=args.critic_learning_rate,
        alpha_lr=args.temperature_learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.replay_buffer_capacity,
        seq_len=args.recurrent_training_sequence_length,
        gamma=args.discount_factor,
        tau=args.soft_update_coefficient,
        alpha=args.temperature_coefficient,
        adaptive_alpha=True,
    )
    noise_managers = [
        NoiseManager(
            enable_sensor_noise=args.sensor_noise,
            enable_actuator_noise=args.actuator_noise,
            seed=args.seed + index,
        )
        for index in range(args.num_parallel_envs)
    ]

    raw_states = env.reset()
    states = np.asarray(
        [noise_managers[i].add_observation_noise(raw_states[i]) for i in range(args.num_parallel_envs)]
    )
    zero_action = np.zeros(2, dtype=np.float32)
    sequence_length = args.recurrent_training_sequence_length
    state_histories = [
        deque([states[i].copy()] * sequence_length, maxlen=sequence_length)
        for i in range(args.num_parallel_envs)
    ]
    action_histories = [
        deque([zero_action.copy()] * sequence_length, maxlen=sequence_length)
        for _ in range(args.num_parallel_envs)
    ]
    previous_states = states.copy()
    previous_actions = np.zeros((args.num_parallel_envs, 2), dtype=np.float32)
    total_steps = 0
    next_save = args.save_interval
    last_log = 0

    try:
        while total_steps < args.total_training_steps:
            if total_steps < args.warmup_steps:
                actions = np.random.uniform(
                    -1.0,
                    1.0,
                    (args.num_parallel_envs, 2),
                ).astype(np.float32)
            else:
                actions = np.asarray(
                    [
                        agent.select_action(
                            previous_states[i],
                            previous_actions[i],
                            states[i],
                            deterministic=False,
                            total_steps=total_steps,
                            env_id=i,
                        )
                        for i in range(args.num_parallel_envs)
                    ]
                )

            current_state_sequences = [
                np.asarray(history, dtype=np.float32) for history in state_histories
            ]
            current_action_sequences = [
                np.asarray(history, dtype=np.float32) for history in action_histories
            ]
            env_actions = np.asarray(
                [
                    noise_managers[i].add_action_noise(actions[i], max_action) * max_action
                    for i in range(args.num_parallel_envs)
                ]
            )
            next_raw, rewards, dones, _ = env.step(env_actions)
            next_states = np.asarray(
                [noise_managers[i].add_observation_noise(next_raw[i]) for i in range(args.num_parallel_envs)]
            )

            for i in range(args.num_parallel_envs):
                action_histories[i].append(actions[i].copy())
                state_histories[i].append(next_states[i].copy())
                agent.replay_buffer.add(
                    current_state_sequences[i],
                    current_action_sequences[i],
                    states[i],
                    actions[i],
                    float(rewards[i]),
                    np.asarray(state_histories[i], dtype=np.float32),
                    np.asarray(action_histories[i], dtype=np.float32),
                    next_states[i],
                    bool(dones[i]),
                )
                previous_states[i] = states[i].copy()
                previous_actions[i] = actions[i].copy()
                if dones[i]:
                    agent.actor.reset_hidden_state(env_id=i)
                    state_histories[i] = deque(
                        [next_states[i].copy()] * sequence_length,
                        maxlen=sequence_length,
                    )
                    action_histories[i] = deque(
                        [zero_action.copy()] * sequence_length,
                        maxlen=sequence_length,
                    )
                    previous_states[i] = next_states[i].copy()
                    previous_actions[i] = zero_action.copy()
            states = next_states
            total_steps += args.num_parallel_envs

            if total_steps > args.warmup_steps and agent.replay_buffer.size >= args.batch_size:
                progress = min(total_steps / args.total_training_steps, 1.0)
                agent.replay_buffer.update_beta(progress)
                for _ in range(args.updates_per_step * args.num_parallel_envs):
                    agent.train()

            while total_steps >= next_save and next_save <= args.total_training_steps:
                save_checkpoint(agent, output_dir, next_save)
                next_save += args.save_interval
            if total_steps - last_log >= args.log_interval:
                print(
                    f"steps={min(total_steps, args.total_training_steps)} "
                    f"buffer={agent.replay_buffer.size} mean_reward={float(np.mean(rewards)):.3f}"
                )
                last_log = total_steps
        save_checkpoint(agent, output_dir, args.total_training_steps)
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reward-ablation",
        choices=["full", "no_progress", "no_action", "no_checkpoint", "no_arrival"],
        default="full",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-training-steps", type=int, default=7_500_000)
    parser.add_argument("--maximum-steps-per-episode", type=int, default=1_000)
    parser.add_argument("--num-parallel-envs", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=20_000)
    parser.add_argument("--save-interval", type=int, default=500_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--recurrent-hidden-size", type=int, default=256)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature-coefficient", type=float, default=0.2)
    parser.add_argument("--temperature-learning-rate", type=float, default=3e-5)
    parser.add_argument("--discount-factor", type=float, default=0.98)
    parser.add_argument("--soft-update-coefficient", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-buffer-capacity", type=int, default=1_000_000)
    parser.add_argument("--recurrent-training-sequence-length", type=int, default=10)
    parser.add_argument("--sensor-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actuator-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.output_dir is None:
        suffix = "" if arguments.reward_ablation == "full" else f"_{arguments.reward_ablation}"
        arguments.output_dir = str(
            SCRIPT_DIR
            / "trained_models"
            / f"sac_lstm{suffix}_seed{arguments.seed}"
        )
    train(arguments)
