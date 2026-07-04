"""Run the full 20-step-horizon MPC evaluation.

The defaults match the archived successful runs: N=20, SLSQP with eight
starts, RK4 prediction with a 0.1 s substep, and at most 100 optimizer
iterations. This is intentionally slow and can be launched from a terminal or
directly from an IDE.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from fast_mpc import FastSailboatMPC, parse_action_levels
from sailboat_s14_nowind import SailboatNowindEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def make_controller(env, args):
    return FastSailboatMPC(
        env=env,
        prediction_horizon=args.prediction_horizon,
        control_horizon=1,
        dt=1.0,
        sub_dt=args.rk4_step,
        integrator="rk4",
        n_candidates=34,
        action_levels_deg=parse_action_levels("1,2,5,10"),
        solver="slsqp",
        slsqp_starts=args.slsqp_starts,
        parallel_starts=1,
        refine=False,
        max_iter=args.max_iter,
        ftol=1e-2,
        optimizer_eps=np.deg2rad(1.0),
    )


def run_episode(args, seed, wind_direction):
    env = SailboatNowindEnv()
    controller = make_controller(env, args)
    state, info = env.reset(
        seed=seed,
        options={"initial_wind_direction_deg": wind_direction},
    )
    controller.reset_hidden_state()
    positions = [np.asarray(state[:2], dtype=float)]
    total_reward = 0.0
    solve_times = []
    terminated = truncated = False
    steps = 0
    try:
        for steps in range(1, args.max_steps + 1):
            started = time.perf_counter()
            action = controller.select_action(state, deterministic=True)
            solve_times.append(time.perf_counter() - started)
            state, reward, terminated, truncated, _ = env.step(action)
            positions.append(np.asarray(state[:2], dtype=float))
            total_reward += float(reward)
            if steps == 1 or steps % args.log_interval == 0:
                distance = float(np.linalg.norm(positions[-1] - np.array([500.0, 0.0])))
                print(
                    f"seed={seed} step={steps} distance={distance:.1f} m "
                    f"solve={solve_times[-1]:.2f} s"
                )
            if terminated or truncated:
                break
    finally:
        controller.close()
        env.close()

    positions = np.asarray(positions)
    return {
        "seed": seed,
        "requested_wind_deg": wind_direction,
        "actual_initial_wind_deg": info["initial_wind_deg"],
        "success": bool(terminated),
        "steps": steps,
        "path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "final_distance_m": float(np.linalg.norm(positions[-1] - np.array([500.0, 0.0]))),
        "total_reward": total_reward,
        "mean_solve_time_s": float(np.mean(solve_times)),
    }


def main(args):
    directions = [float(value.strip()) for value in args.wind_directions.split(",")]
    rows = []
    for direction_index, direction in enumerate(directions):
        for episode in range(args.episodes):
            seed = args.base_seed + direction_index * 10000 + episode
            row = run_episode(args, seed, direction)
            rows.append(row)
            print(
                f"finished seed={seed} wind={direction} "
                f"success={row['success']} final_distance={row['final_distance_m']:.2f} m"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-directions", default="0,90,150,180")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--prediction-horizon", type=int, default=20)
    parser.add_argument("--rk4-step", type=float, default=0.1)
    parser.add_argument("--slsqp-starts", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "mpc_results"))
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
