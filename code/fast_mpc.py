"""
Fast MPC runner for the sailboat environment.

This file is intentionally separate from multi_MPC.py.  It keeps the same
experiment workflow and saved artifacts, but uses a lighter MPC solve:

1. cached constant model terms,
2. shorter/default prediction horizon,
3. configurable integration sub-step,
4. warm-started candidates,
5. optional low-iteration local SLSQP refinement.

The default settings target fast offline experiments.  For higher fidelity,
increase --prediction_horizon, decrease --sub_dt, or keep --refine enabled.
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
import multiprocessing as mp

import numpy as np

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import globals
from sailboat_s14_nowind import SailboatNowindEnv


NOWIND_ENV_ID = "SailboatRepro-Nowind-v0"


def register_environments():
    if NOWIND_ENV_ID not in gym.registry:
        gym.register(
            id=NOWIND_ENV_ID,
            entry_point=SailboatNowindEnv,
            max_episode_steps=3000,
        )

warnings.filterwarnings("ignore", category=RuntimeWarning)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def finite_clip(value, max_abs):
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -max_abs, max_abs))


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def parse_action_levels(levels_text):
    if levels_text is None:
        return [1.0, 2.0, 5.0, 10.0]
    if isinstance(levels_text, (list, tuple, np.ndarray)):
        raw_levels = levels_text
    else:
        raw_levels = str(levels_text).replace(";", ",").split(",")

    levels = []
    for item in raw_levels:
        text = str(item).strip()
        if not text:
            continue
        value = abs(float(text))
        if value > 0.0:
            levels.append(value)
    if not levels:
        levels = [1.0, 2.0, 5.0, 10.0]
    return sorted(set(levels))


class FastSailboatMPC:
    MAX_VELOCITY = 50.0
    MAX_ANGULAR_VELOCITY = 10.0
    MAX_FORCE = 1e6
    MAX_POSITION = 2000.0

    def __init__(
        self,
        env,
        prediction_horizon=10,
        control_horizon=1,
        dt=1.0,
        sub_dt=0.1,
        integrator="rk4",
        n_candidates=14,
        action_levels_deg=None,
        solver="slsqp",
        slsqp_starts=2,
        parallel_starts=1,
        refine=False,
        max_iter=8,
        ftol=1e-2,
        optimizer_eps=np.deg2rad(1.0),
    ):
        self.env = env
        self.N = int(prediction_horizon)
        self.M = int(control_horizon)
        self.dt = float(dt)
        self.sub_steps = max(1, int(round(self.dt / float(sub_dt))))
        self.sim_dt = self.dt / self.sub_steps
        self.integrator = integrator.lower()
        if self.integrator not in {"euler", "rk2", "rk4"}:
            raise ValueError("--integrator must be one of: euler, rk2, rk4")

        self.n_candidates = max(1, int(n_candidates))
        self.action_levels_deg = parse_action_levels(action_levels_deg)
        self.solver = solver.lower()
        if self.solver not in {"candidate", "hybrid", "slsqp"}:
            raise ValueError("--solver must be one of: candidate, hybrid, slsqp")
        self.slsqp_starts = max(1, int(slsqp_starts))
        self.parallel_starts = max(1, int(parallel_starts))
        self.refine = bool(refine)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.optimizer_eps = float(optimizer_eps)

        self.max_sail_rate = np.pi / 18.0
        self.max_rudder_rate = np.pi / 18.0
        self.max_sail_angle = np.pi / 2.0
        self.max_rudder_angle = np.pi / 6.0

        self.w_effective_vel = 100.0
        self.w_heading_error = 3.0
        self.w_control_rate_sail = 1.0
        self.w_control_rate_rudder = 2.0
        self.w_no_go_zone = 200.0
        self.no_go_zone_angle = np.pi * 3.0 / 4.0
        self.w_distance = 50.0
        self.w_vmg = 80.0
        self.optimal_tacking_angle = np.pi * 2.0 / 3.0
        self.w_terminal_distance = 0.0

        self.params = dict(env.par)
        self.M_inv = self._compute_mass_inverse()
        self.interp = self._extract_interp_data()
        self.log_wind_ratio = np.log(self.params["h1"] / self.params["h0"])
        self.rudder_induced = self.params["Ar"] / (
            np.pi * 2.0 * self.params["zeta_r"] * self.params["d_r"] ** 2
        )
        self.keel_induced = self.params["Ak"] / (
            np.pi * 2.0 * self.params["zeta_k"] * self.params["d_k"] ** 2
        )

        self.action_dim = self.M * 2
        self.max_delta = self.max_sail_rate * self.dt
        self.action_levels = np.array(
            [
                min(np.deg2rad(level_deg), self.max_delta)
                for level_deg in self.action_levels_deg
            ],
            dtype=float,
        )
        self.bounds = [(-self.max_delta, self.max_delta)] * self.action_dim
        self.lower = np.array([b[0] for b in self.bounds], dtype=float)
        self.upper = np.array([b[1] for b in self.bounds], dtype=float)

        self.last_solution = None
        self.last_action = np.zeros(2, dtype=float)
        self.last_info = {}
        self.cost_evals = 0
        self.pool = None
        self.pool_size = 0

        print("Fast MPC initialized")
        print(f"  prediction horizon N: {self.N}")
        print(f"  control horizon M: {self.M}")
        print(f"  integration: {self.integrator}, sub_steps={self.sub_steps}, sim_dt={self.sim_dt:.3f}s")
        print(
            f"  solver: {self.solver}, slsqp_starts={self.slsqp_starts}, "
            f"parallel_starts={self.parallel_starts}, max_iter={self.max_iter}"
        )
        print(f"  candidates: {self.n_candidates}, refine={self.refine}")
        print(f"  action levels deg: {self.action_levels_deg}")

    def _worker_payload(self):
        return {
            "N": self.N,
            "M": self.M,
            "dt": self.dt,
            "sub_steps": self.sub_steps,
            "sim_dt": self.sim_dt,
            "integrator": self.integrator,
            "n_candidates": self.n_candidates,
            "action_levels_deg": self.action_levels_deg,
            "solver": self.solver,
            "slsqp_starts": self.slsqp_starts,
            "parallel_starts": self.parallel_starts,
            "refine": self.refine,
            "max_iter": self.max_iter,
            "ftol": self.ftol,
            "optimizer_eps": self.optimizer_eps,
            "max_sail_rate": self.max_sail_rate,
            "max_rudder_rate": self.max_rudder_rate,
            "max_sail_angle": self.max_sail_angle,
            "max_rudder_angle": self.max_rudder_angle,
            "w_effective_vel": self.w_effective_vel,
            "w_heading_error": self.w_heading_error,
            "w_control_rate_sail": self.w_control_rate_sail,
            "w_control_rate_rudder": self.w_control_rate_rudder,
            "w_no_go_zone": self.w_no_go_zone,
            "no_go_zone_angle": self.no_go_zone_angle,
            "w_distance": self.w_distance,
            "w_vmg": self.w_vmg,
            "optimal_tacking_angle": self.optimal_tacking_angle,
            "w_terminal_distance": self.w_terminal_distance,
            "params": self.params,
            "M_inv": self.M_inv,
            "interp": self.interp,
            "log_wind_ratio": self.log_wind_ratio,
            "rudder_induced": self.rudder_induced,
            "keel_induced": self.keel_induced,
            "action_dim": self.action_dim,
            "max_delta": self.max_delta,
            "action_levels": self.action_levels,
            "bounds": self.bounds,
            "lower": self.lower,
            "upper": self.upper,
        }

    @classmethod
    def from_worker_payload(cls, payload):
        obj = cls.__new__(cls)
        obj.env = None
        for key, value in payload.items():
            setattr(obj, key, value)
        obj.last_solution = None
        obj.last_action = np.zeros(2, dtype=float)
        obj.last_info = {}
        obj.cost_evals = 0
        obj.pool = None
        obj.pool_size = 0
        return obj

    def _start_pool(self, worker_count):
        worker_count = max(1, int(worker_count))
        if self.pool is not None and self.pool_size == worker_count:
            return
        self._close_pool()
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        self.pool = ctx.Pool(
            processes=worker_count,
            initializer=_parallel_worker_init,
            initargs=(self._worker_payload(),),
        )
        self.pool_size = worker_count

    def _close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self.pool_size = 0

    def _compute_mass_inverse(self):
        if hasattr(self.env, "M_inv"):
            return np.asarray(self.env.M_inv, dtype=float)

        p = self.params
        m_rb = np.array(
            [
                [p["m"], 0.0, 0.0, 0.0],
                [0.0, p["m"], 0.0, 0.0],
                [0.0, 0.0, p["Ixx"], -p["Ixz"]],
                [0.0, 0.0, -p["Ixz"], p["Izz"]],
            ],
            dtype=float,
        )
        m_a = np.array(
            [
                [p["a11"], 0.0, 0.0, 0.0],
                [0.0, p["a22"], p["a24"], p["a26"]],
                [0.0, p["a24"], p["a44"], p["a46"]],
                [0.0, p["a26"], p["a46"], p["a66"]],
            ],
            dtype=float,
        )
        return np.linalg.inv(m_rb + m_a)

    def _extract_interp_data(self):
        angle_range = np.linspace(-180.0, 180.0, 361)
        vel_range = np.linspace(0.0, 15.0, 151)
        env = self.env
        return {
            "sail_cl_x": angle_range,
            "sail_cl_y": np.array([env.sail_cl_interp(a) for a in angle_range], dtype=float),
            "sail_cd_x": angle_range,
            "sail_cd_y": np.array([env.sail_cd_interp(a) for a in angle_range], dtype=float),
            "rudder_cl_x": angle_range,
            "rudder_cl_y": np.array([env.rudder_cl_interp(a) for a in angle_range], dtype=float),
            "rudder_cd_x": angle_range,
            "rudder_cd_y": np.array([env.rudder_cd_interp(a) for a in angle_range], dtype=float),
            "keel_cl_x": angle_range,
            "keel_cl_y": np.array([env.keel_cl_interp(a) for a in angle_range], dtype=float),
            "keel_cd_x": angle_range,
            "keel_cd_y": np.array([env.keel_cd_interp(a) for a in angle_range], dtype=float),
            "hull_x": vel_range,
            "hull_y": np.array([env.hull_resistance_interp(v) for v in vel_range], dtype=float),
        }

    def get_wind_info(self):
        return globals.w_change_v, globals.w_change_d

    def reset_hidden_state(self, env_id=0):
        self.last_solution = None
        self.last_action = np.zeros(2, dtype=float)
        self.last_info = {}

    def close(self):
        self._close_pool()

    def _clip_state(self, state):
        s = np.asarray(state, dtype=float).copy()
        s[0] = np.clip(s[0], -self.MAX_POSITION, self.MAX_POSITION)
        s[1] = np.clip(s[1], -self.MAX_POSITION, self.MAX_POSITION)
        s[2] = np.clip(s[2], -np.pi / 2.0, np.pi / 2.0)
        s[3] = wrap_angle(s[3])
        s[4] = np.clip(s[4], -self.MAX_VELOCITY, self.MAX_VELOCITY)
        s[5] = np.clip(s[5], -self.MAX_VELOCITY, self.MAX_VELOCITY)
        s[6] = np.clip(s[6], -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)
        s[7] = np.clip(s[7], -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)
        return s

    def state_derivative(self, state, sail_angle, rudder_angle, wind_v, wind_d):
        p = self.params
        interp = self.interp

        s = self._clip_state(state)
        x, y, phi, psi, u, v, roll_rate, yaw_rate = s

        sail = np.clip(sail_angle, -self.max_sail_angle, self.max_sail_angle)
        rudder = np.clip(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)

        u = finite_clip(u, self.MAX_VELOCITY)
        v = finite_clip(v, self.MAX_VELOCITY)
        roll_rate = finite_clip(roll_rate, self.MAX_ANGULAR_VELOCITY)
        yaw_rate = finite_clip(yaw_rate, self.MAX_ANGULAR_VELOCITY)
        wind_v = finite_clip(wind_v, self.MAX_VELOCITY)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        if abs(cos_phi) < 0.01:
            cos_phi = 0.01 if cos_phi >= 0.0 else -0.01

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        wind_scale_arg = abs(p["z_s"]) * cos_phi / p["h0"]
        if wind_scale_arg <= 1e-3:
            wind_scale_arg = 1e-3
        wind_scale = np.log(wind_scale_arg) / self.log_wind_ratio
        wind_x = wind_scale * wind_v * np.cos(wind_d)
        wind_y = wind_scale * wind_v * np.sin(wind_d)

        body_wind_x = cos_psi * wind_x + sin_psi * wind_y
        body_wind_y0 = -sin_psi * wind_x + cos_psi * wind_y
        body_wind_y = cos_phi * body_wind_y0

        cross_x = -yaw_rate * p["ys"]
        cross_y = yaw_rate * p["xs"] - roll_rate * p["zs"]
        v_awu = finite_clip(body_wind_x - u - cross_x, self.MAX_VELOCITY)
        v_awv = finite_clip(body_wind_y - v - cross_y, self.MAX_VELOCITY)
        alpha_aw = np.arctan2(v_awv, -v_awu)

        alpha_as_deg = np.rad2deg(wrap_angle(alpha_aw - sail))
        cls = np.interp(alpha_as_deg, interp["sail_cl_x"], interp["sail_cl_y"])
        cds = np.interp(alpha_as_deg, interp["sail_cd_x"], interp["sail_cd_y"])
        v_aw_sq = min(v_awu * v_awu + v_awv * v_awv, self.MAX_VELOCITY ** 2)
        lift_sail = finite_clip(0.5 * p["rho_a"] * v_aw_sq * p["As"] * cls, self.MAX_FORCE)
        drag_sail = finite_clip(0.5 * p["rho_a"] * v_aw_sq * p["As"] * cds, self.MAX_FORCE)
        sail_side = lift_sail * np.cos(alpha_aw) + drag_sail * np.sin(alpha_aw)
        sail_drive = lift_sail * np.sin(alpha_aw) - drag_sail * np.cos(alpha_aw)
        tau_sail = np.array(
            [
                finite_clip(sail_drive, self.MAX_FORCE),
                finite_clip(sail_side, self.MAX_FORCE),
                finite_clip(-sail_side * p["zs"], self.MAX_FORCE),
                finite_clip(-sail_drive * p["Xce"] * np.sin(sail) + sail_side * (p["Xm"] - p["Xce"] * np.cos(sail)), self.MAX_FORCE),
            ],
            dtype=float,
        )

        v_aru = finite_clip(-u + yaw_rate * p["yr"], self.MAX_VELOCITY)
        v_arv = finite_clip(-v - yaw_rate * p["xr"] + roll_rate * p["zr"], self.MAX_VELOCITY)
        alpha_ar = np.arctan2(v_arv, -v_aru)
        alpha_rudder_deg = np.rad2deg(wrap_angle(alpha_ar - rudder))
        clr = np.interp(alpha_rudder_deg, interp["rudder_cl_x"], interp["rudder_cl_y"])
        cdr_base = np.interp(alpha_rudder_deg, interp["rudder_cd_x"], interp["rudder_cd_y"])
        cdr = cdr_base + clr * clr * self.rudder_induced
        v_ar_sq = min(v_aru * v_aru + v_arv * v_arv, self.MAX_VELOCITY ** 2)
        lift_rudder = finite_clip(0.5 * p["rho_w"] * p["Ar"] * v_ar_sq * clr, self.MAX_FORCE)
        drag_rudder = finite_clip(0.5 * p["rho_w"] * p["Ar"] * v_ar_sq * cdr, self.MAX_FORCE)
        rudder_side = lift_rudder * np.cos(alpha_ar) + drag_rudder * np.sin(alpha_ar)
        rudder_drive = lift_rudder * np.sin(alpha_ar) - drag_rudder * np.cos(alpha_ar)
        tau_rudder = np.array(
            [
                finite_clip(rudder_drive, self.MAX_FORCE),
                finite_clip(rudder_side, self.MAX_FORCE),
                finite_clip(-rudder_side * p["zr"], self.MAX_FORCE),
                finite_clip(rudder_side * p["xr"], self.MAX_FORCE),
            ],
            dtype=float,
        )
        tau = tau_sail + tau_rudder

        v_aku = finite_clip(-u + yaw_rate * p["yk"], self.MAX_VELOCITY)
        v_akv = finite_clip(-v - yaw_rate * p["xk"] + roll_rate * p["zk"], self.MAX_VELOCITY)
        alpha_ak = np.arctan2(v_akv, -v_aku)
        alpha_keel_deg = np.rad2deg(wrap_angle(alpha_ak))
        clk = np.interp(alpha_keel_deg, interp["keel_cl_x"], interp["keel_cl_y"])
        cdk_base = np.interp(alpha_keel_deg, interp["keel_cd_x"], interp["keel_cd_y"])
        cdk = cdk_base + clk * clk * self.keel_induced
        v_ak_sq = min(v_aku * v_aku + v_akv * v_akv, self.MAX_VELOCITY ** 2)
        lift_keel = finite_clip(0.5 * p["rho_w"] * p["Ak"] * v_ak_sq * clk, self.MAX_FORCE)
        drag_keel = finite_clip(0.5 * p["rho_w"] * p["Ak"] * v_ak_sq * cdk, self.MAX_FORCE)
        keel_side = lift_keel * np.cos(alpha_ak) + drag_keel * np.sin(alpha_ak)
        d_keel = np.array(
            [
                finite_clip(-lift_keel * np.sin(alpha_ak) + drag_keel * np.cos(alpha_ak), self.MAX_FORCE),
                finite_clip(-keel_side, self.MAX_FORCE),
                finite_clip(keel_side * p["zk"], self.MAX_FORCE),
                finite_clip(-keel_side * p["xk"], self.MAX_FORCE),
            ],
            dtype=float,
        )

        v_ahu = finite_clip(-u + yaw_rate * p["yh"], self.MAX_VELOCITY)
        if abs(cos_phi) > 0.01:
            v_ahv = finite_clip((-v - yaw_rate * p["xh"] + roll_rate * p["zh"]) / cos_phi, self.MAX_VELOCITY)
        else:
            v_ahv = 0.0
        hull_speed = min(np.hypot(v_aku, v_akv), self.MAX_VELOCITY)
        alpha_ah = np.arctan2(v_ahv, -v_ahu)
        hull_resistance = finite_clip(np.interp(hull_speed, interp["hull_x"], interp["hull_y"]), self.MAX_FORCE)
        d_hull = np.array(
            [
                finite_clip(hull_resistance * np.cos(alpha_ah), self.MAX_FORCE),
                finite_clip(-hull_resistance * np.sin(alpha_ah) * cos_phi, self.MAX_FORCE),
                finite_clip(hull_resistance * np.sin(alpha_ah) * cos_phi * p["zh"], self.MAX_FORCE),
                finite_clip(-hull_resistance * np.sin(alpha_ah) * cos_phi * p["xh"], self.MAX_FORCE),
            ],
            dtype=float,
        )

        x_dot = cos_psi * u - sin_psi * cos_phi * v
        y_dot = sin_psi * u + cos_psi * cos_phi * v
        phi_dot = finite_clip(roll_rate, self.MAX_ANGULAR_VELOCITY)
        psi_dot = finite_clip(cos_phi * yaw_rate, self.MAX_ANGULAR_VELOCITY)

        d_heel_yaw = np.array(
            [
                0.0,
                0.0,
                finite_clip(p["c"] * phi_dot * abs(phi_dot), self.MAX_FORCE),
                finite_clip(p["d"] * psi_dot * abs(psi_dot) * cos_phi, self.MAX_FORCE),
            ],
            dtype=float,
        )
        damping = d_keel + d_hull + d_heel_yaw

        phi_deg = finite_clip(np.rad2deg(phi), 90.0)
        restoring = np.array(
            [
                0.0,
                0.0,
                finite_clip(p["a"] * phi_deg * phi_deg + p["b"] * phi_deg, self.MAX_FORCE),
                0.0,
            ],
            dtype=float,
        )

        c_rb_nu = np.array(
            [
                -p["m"] * yaw_rate * v,
                p["m"] * yaw_rate * u,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        a_term = p["a22"] * v + p["a24"] * roll_rate + p["a26"] * yaw_rate
        c_a_nu = np.array(
            [
                -a_term * yaw_rate,
                p["a11"] * u * yaw_rate,
                0.0,
                a_term * u - p["a11"] * u * v,
            ],
            dtype=float,
        )
        rhs = -(c_rb_nu + c_a_nu) - damping - restoring + tau
        nu_dot = self.M_inv @ rhs
        nu_dot = np.clip(np.nan_to_num(nu_dot, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)

        state_dot = np.array(
            [x_dot, y_dot, phi_dot, psi_dot, nu_dot[0], nu_dot[1], nu_dot[2], nu_dot[3]],
            dtype=float,
        )
        if not np.all(np.isfinite(state_dot)):
            return np.zeros(8, dtype=float)
        return state_dot

    def _integrate_one_substep(self, state, sail, rudder, wind_v, wind_d):
        h = self.sim_dt
        if self.integrator == "euler":
            return state + h * self.state_derivative(state, sail, rudder, wind_v, wind_d)
        if self.integrator == "rk2":
            k1 = self.state_derivative(state, sail, rudder, wind_v, wind_d)
            k2 = self.state_derivative(state + 0.5 * h * k1, sail, rudder, wind_v, wind_d)
            return state + h * k2

        k1 = self.state_derivative(state, sail, rudder, wind_v, wind_d)
        k2 = self.state_derivative(state + 0.5 * h * k1, sail, rudder, wind_v, wind_d)
        k3 = self.state_derivative(state + 0.5 * h * k2, sail, rudder, wind_v, wind_d)
        k4 = self.state_derivative(state + h * k3, sail, rudder, wind_v, wind_d)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def predict_trajectory(self, state_8d, action_sequence_flat, initial_sail, initial_rudder, wind_v, wind_d):
        action_sequence = np.asarray(action_sequence_flat, dtype=float).reshape(self.M, 2)
        trajectory = np.empty((self.N + 1, 8), dtype=float)
        current_state = self._clip_state(state_8d)
        trajectory[0] = current_state

        current_sail = float(initial_sail)
        current_rudder = float(initial_rudder)
        target_sail = float(initial_sail)
        target_rudder = float(initial_rudder)
        max_sail_change = self.max_sail_rate * self.dt
        max_rudder_change = self.max_rudder_rate * self.dt

        for k in range(self.N):
            action = action_sequence[k] if k < self.M else action_sequence[-1]
            target_sail = np.clip(target_sail + action[0], -self.max_sail_angle, self.max_sail_angle)
            target_rudder = np.clip(target_rudder + action[1], -self.max_rudder_angle, self.max_rudder_angle)

            start_sail = current_sail
            start_rudder = current_rudder
            delta_sail = np.clip(target_sail - current_sail, -max_sail_change, max_sail_change)
            delta_rudder = np.clip(target_rudder - current_rudder, -max_rudder_change, max_rudder_change)

            for sub_k in range(self.sub_steps):
                progress = (sub_k + 1.0) / self.sub_steps
                sail = np.clip(start_sail + progress * delta_sail, -self.max_sail_angle, self.max_sail_angle)
                rudder = np.clip(start_rudder + progress * delta_rudder, -self.max_rudder_angle, self.max_rudder_angle)
                next_state = self._integrate_one_substep(current_state, sail, rudder, wind_v, wind_d)
                if np.all(np.isfinite(next_state)):
                    current_state = self._clip_state(next_state)

            current_sail = start_sail + delta_sail
            current_rudder = start_rudder + delta_rudder
            trajectory[k + 1] = current_state

        return trajectory

    def compute_cost(self, action_sequence_flat, state_8d, target_pos, initial_sail, initial_rudder, wind_v, wind_d):
        action_sequence_flat = np.clip(np.asarray(action_sequence_flat, dtype=float), self.lower, self.upper)
        self.cost_evals += 1
        try:
            trajectory = self.predict_trajectory(
                state_8d,
                action_sequence_flat,
                initial_sail,
                initial_rudder,
                wind_v,
                wind_d,
            )
        except Exception:
            return 1e10

        if not np.all(np.isfinite(trajectory)):
            return 1e10

        action_sequence = np.asarray(action_sequence_flat, dtype=float).reshape(self.M, 2)
        target_pos = np.asarray(target_pos, dtype=float)
        initial_dist = float(np.linalg.norm(np.asarray(state_8d[:2], dtype=float) - target_pos))
        prev_dist = initial_dist
        total_cost = 0.0

        for k in range(self.N):
            state = trajectory[k + 1]
            x, y, phi, psi, u, v, roll_rate, yaw_rate = state
            current_pos = state[:2]
            current_dist = float(np.linalg.norm(current_pos - target_pos))

            target_angle = np.arctan2(target_pos[1] - y, target_pos[0] - x)
            target_to_wind = wrap_angle(target_angle - wind_d)
            target_in_no_go_zone = abs(target_to_wind) > self.no_go_zone_angle
            heading_to_wind = wrap_angle(psi - wind_d)

            speed_quality = u / (1.0 + abs(v)) if (1.0 + abs(v)) > 0.01 else 0.0
            total_cost -= self.w_effective_vel * 0.5 * speed_quality

            if current_dist > 0.1:
                to_target = (target_pos - current_pos) / current_dist
                vel_world = np.array(
                    [
                        u * np.cos(psi) - v * np.sin(psi),
                        u * np.sin(psi) + v * np.cos(psi),
                    ],
                    dtype=float,
                )
                effective_vel = float(np.dot(vel_world, to_target))
                total_cost -= self.w_effective_vel * effective_vel

                step_dist_reduction = prev_dist - current_dist
                time_weight = 1.0 + 0.5 * k / max(self.N, 1)
                total_cost -= self.w_distance * step_dist_reduction * time_weight
                if k == self.N - 1:
                    total_cost -= self.w_distance * (initial_dist - current_dist) * 1.5

                heading_error = wrap_angle(psi - target_angle)
                if target_in_no_go_zone:
                    best_left = wind_d + self.optimal_tacking_angle
                    best_right = wind_d - self.optimal_tacking_angle
                    error_left = abs(wrap_angle(psi - best_left))
                    error_right = abs(wrap_angle(psi - best_right))
                    cross_product = np.cos(psi) * (target_pos[1] - y) - np.sin(psi) * (target_pos[0] - x)
                    if cross_product > 0.0:
                        tacking_error = min(error_left * 0.7, error_right * 1.3)
                    else:
                        tacking_error = min(error_left * 1.3, error_right * 0.7)
                    total_cost += self.w_heading_error * tacking_error * tacking_error * 0.3

                    speed = np.hypot(u, v)
                    angle_from_downwind = np.pi - abs(heading_to_wind)
                    vmg_upwind = speed * np.cos(angle_from_downwind)
                    if vmg_upwind > 0.0:
                        total_cost -= self.w_vmg * vmg_upwind
                else:
                    total_cost += self.w_heading_error * heading_error * heading_error

            if k < self.M:
                action = action_sequence[k]
                total_cost += self.w_control_rate_sail * action[0] * action[0]
                total_cost += self.w_control_rate_rudder * action[1] * action[1]

            if abs(heading_to_wind) > self.no_go_zone_angle:
                penetration = abs(heading_to_wind) - self.no_go_zone_angle
                max_penetration = np.pi - self.no_go_zone_angle
                total_cost += self.w_no_go_zone * (penetration / max_penetration) ** 2

            prev_dist = current_dist
            total_cost += 0.1

        terminal_dist = float(np.linalg.norm(trajectory[-1, :2] - target_pos))
        total_cost += self.w_terminal_distance * terminal_dist

        if not np.isfinite(total_cost):
            total_cost = 1e10
        return float(total_cost)

    def _warm_start_guess(self):
        if self.last_solution is not None and len(self.last_solution) == self.action_dim:
            if self.M == 1:
                guess = self.last_solution.copy()
            else:
                guess = np.roll(self.last_solution.copy(), -2)
                guess[-2:] = 0.0
        else:
            guess = np.zeros(self.action_dim, dtype=float)
        return np.clip(guess, self.lower, self.upper)

    def _candidate_score(self, candidate, state_8d, target_pos, current_sail, current_rudder, wind_v, wind_d):
        first_action = np.asarray(candidate, dtype=float).reshape(self.M, 2)[0]
        psi = float(state_8d[3])
        phi = float(state_8d[2])
        x = float(state_8d[0])
        y = float(state_8d[1])

        target_angle = np.arctan2(target_pos[1] - y, target_pos[0] - x)
        target_to_wind = wrap_angle(target_angle - wind_d)
        if abs(target_to_wind) > self.no_go_zone_angle:
            left_heading = wind_d + self.optimal_tacking_angle
            right_heading = wind_d - self.optimal_tacking_angle
            cross = np.cos(psi) * (target_pos[1] - y) - np.sin(psi) * (target_pos[0] - x)
            desired_heading = left_heading if cross > 0.0 else right_heading
        else:
            desired_heading = target_angle

        sail = np.clip(current_sail + first_action[0], -self.max_sail_angle, self.max_sail_angle)
        rudder = np.clip(current_rudder + first_action[1], -self.max_rudder_angle, self.max_rudder_angle)

        try:
            state_dot = self.state_derivative(state_8d, sail, rudder, wind_v, wind_d)
            approx_heading = wrap_angle(
                psi
                + self.dt * state_dot[3]
                + 0.5 * self.dt * self.dt * np.cos(phi) * state_dot[7]
            )
            heading_score = abs(wrap_angle(approx_heading - desired_heading))
        except Exception:
            heading_score = np.pi

        magnitude_score = 0.04 * np.linalg.norm(candidate / max(self.max_delta, 1e-9))
        sail_score = 0.01 * abs(first_action[0]) / max(self.max_delta, 1e-9)
        return float(heading_score + magnitude_score + sail_score)

    def _candidate_pool(self, base_guess, state_8d, target_pos, current_sail, current_rudder, wind_v, wind_d):
        candidates = [np.clip(base_guess, self.lower, self.upper), np.zeros(self.action_dim, dtype=float)]

        if self.M == 1:
            for d in self.action_levels:
                patterns = [
                    [0.0, d],
                    [0.0, -d],
                    [d, 0.0],
                    [-d, 0.0],
                    [d, d],
                    [d, -d],
                    [-d, d],
                    [-d, -d],
                ]
                for pattern in patterns:
                    candidates.append(np.array(pattern, dtype=float))
        else:
            for d in self.action_levels:
                for pattern in ([0.0, d], [0.0, -d], [d, 0.0], [-d, 0.0], [d, -d], [-d, d]):
                    candidates.append(np.tile(np.array(pattern, dtype=float), self.M))

        unique = []
        seen = set()
        for candidate in candidates:
            clipped = np.clip(candidate, self.lower, self.upper)
            key = tuple(np.round(clipped, 10))
            if key not in seen:
                seen.add(key)
                unique.append(clipped)

        protected = unique[:2]
        ranked = sorted(
            unique[2:],
            key=lambda candidate: self._candidate_score(
                candidate,
                state_8d,
                target_pos,
                current_sail,
                current_rudder,
                wind_v,
                wind_d,
            ),
        )
        return (protected + ranked)[: self.n_candidates]

    def solve_mpc(self, current_state_18d, target_pos):
        state_8d = np.asarray(current_state_18d[:8], dtype=float)
        current_sail = float(current_state_18d[8])
        current_rudder = float(current_state_18d[9])
        target_pos = np.asarray(target_pos, dtype=float)
        wind_v, wind_d = self.get_wind_info()

        self.cost_evals = 0
        base_guess = self._warm_start_guess()
        candidates = self._candidate_pool(
            base_guess,
            state_8d,
            target_pos,
            current_sail,
            current_rudder,
            wind_v,
            wind_d,
        )

        candidate_costs = [
            self.compute_cost(candidate, state_8d, target_pos, current_sail, current_rudder, wind_v, wind_d)
            for candidate in candidates
        ]
        best_idx = int(np.argmin(candidate_costs))
        best_solution = candidates[best_idx].copy()
        best_cost = float(candidate_costs[best_idx])
        optimizer_success = False
        optimizer_message = "candidate-only"
        optimizer_runs = 0
        parallel_error = ""

        if self.solver in {"hybrid", "slsqp"} and self.max_iter > 0:
            def objective(u):
                return self.compute_cost(u, state_8d, target_pos, current_sail, current_rudder, wind_v, wind_d)

            if self.solver == "hybrid":
                start_indices = [best_idx]
            else:
                ranked_indices = list(np.argsort(candidate_costs))
                preferred_indices = [best_idx, 0, 1] + ranked_indices
                start_indices = []
                seen_indices = set()
                for idx in preferred_indices:
                    idx = int(idx)
                    if idx in seen_indices or idx >= len(candidates):
                        continue
                    seen_indices.add(idx)
                    start_indices.append(idx)
                    if len(start_indices) >= self.slsqp_starts:
                        break

            messages = []
            if self.parallel_starts > 1 and len(start_indices) > 1:
                try:
                    worker_count = min(self.parallel_starts, len(start_indices))
                    self._start_pool(worker_count)
                    worker_args = [
                        (
                            start_idx,
                            candidates[start_idx],
                            state_8d,
                            target_pos,
                            current_sail,
                            current_rudder,
                            wind_v,
                            wind_d,
                            self.max_iter,
                            self.ftol,
                            self.optimizer_eps,
                            self.bounds,
                        )
                        for start_idx in start_indices
                    ]
                    results = self.pool.map(_parallel_slsqp_worker, worker_args)
                    for start_idx, solution, fun, success, message, evals in results:
                        optimizer_runs += 1
                        self.cost_evals += int(evals)
                        optimizer_success = optimizer_success or bool(success)
                        messages.append(f"start{start_idx}: {message}")
                        if np.isfinite(fun) and fun < best_cost:
                            best_solution = np.clip(np.asarray(solution, dtype=float), self.lower, self.upper)
                            best_cost = float(fun)
                except Exception as exc:
                    parallel_error = str(exc)
                    messages.append(f"parallel-fallback: {parallel_error}")
                    self._close_pool()

            if optimizer_runs == 0:
                for start_idx in start_indices:
                    try:
                        result = minimize(
                            objective,
                            candidates[start_idx],
                            method="SLSQP",
                            bounds=self.bounds,
                            options={
                                "maxiter": self.max_iter,
                                "ftol": self.ftol,
                                "eps": self.optimizer_eps,
                                "disp": False,
                            },
                        )
                        optimizer_runs += 1
                        optimizer_success = optimizer_success or bool(result.success)
                        messages.append(f"start{start_idx}: {result.message}")
                        if np.isfinite(result.fun) and result.fun < best_cost:
                            best_solution = np.clip(np.asarray(result.x, dtype=float), self.lower, self.upper)
                            best_cost = float(result.fun)
                    except Exception as exc:
                        messages.append(f"start{start_idx}: optimizer-error: {exc}")
            optimizer_message = "; ".join(messages) if messages else "slsqp-skipped"

        best_solution = np.clip(np.asarray(best_solution, dtype=float), self.lower, self.upper)
        self.last_solution = best_solution.copy()
        action = best_solution.reshape(self.M, 2)[0]
        self.last_action = action.copy()
        self.last_info = {
            "cost": best_cost,
            "cost_evals": self.cost_evals,
            "candidate_count": len(candidates),
            "candidate_best_idx": best_idx,
            "optimizer_success": optimizer_success,
            "optimizer_runs": optimizer_runs,
            "parallel_workers": self.pool_size,
            "parallel_error": parallel_error,
            "optimizer_message": optimizer_message,
        }
        return action

    def select_action(self, state, deterministic=True, total_steps=None):
        target_pos = np.array([state[12], state[13]], dtype=float)
        return self.solve_mpc(state, target_pos)


_PARALLEL_MPC = None


def _parallel_worker_init(payload):
    global _PARALLEL_MPC
    _PARALLEL_MPC = FastSailboatMPC.from_worker_payload(payload)


def _parallel_slsqp_worker(args):
    (
        start_idx,
        start,
        state_8d,
        target_pos,
        current_sail,
        current_rudder,
        wind_v,
        wind_d,
        max_iter,
        ftol,
        optimizer_eps,
        bounds,
    ) = args
    controller = _PARALLEL_MPC
    controller.cost_evals = 0

    def objective(u):
        return controller.compute_cost(
            u,
            state_8d,
            target_pos,
            current_sail,
            current_rudder,
            wind_v,
            wind_d,
        )

    try:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "ftol": ftol,
                "eps": optimizer_eps,
                "disp": False,
            },
        )
        return (
            int(start_idx),
            np.asarray(result.x, dtype=float),
            float(result.fun),
            bool(result.success),
            str(result.message),
            int(controller.cost_evals),
        )
    except Exception as exc:
        return (
            int(start_idx),
            np.asarray(start, dtype=float),
            1e10,
            False,
            f"optimizer-error: {exc}",
            int(controller.cost_evals),
        )


def set_fixed_wind(wind_angle_deg):
    wind_rad = np.deg2rad(wind_angle_deg)
    globals.w0[1] = wind_rad
    globals.w_ini_d = wind_rad
    globals.w_last_d = wind_rad
    globals.w_change_d = wind_rad


def save_arrays(output_path, observations, actions, rewards, wind_v, wind_d, computation_times, diagnostics):
    np.savetxt(os.path.join(output_path, "observations.csv"), observations, delimiter=",")
    np.savetxt(os.path.join(output_path, "actions.csv"), actions, delimiter=",")
    np.savetxt(os.path.join(output_path, "rewards.csv"), rewards, delimiter=",")
    np.savetxt(os.path.join(output_path, "wind_speed.csv"), wind_v, delimiter=",")
    np.savetxt(os.path.join(output_path, "wind_direction.csv"), wind_d, delimiter=",")
    np.savetxt(os.path.join(output_path, "computation_times.csv"), computation_times, delimiter=",")
    np.savetxt(os.path.join(output_path, "diagnostics.csv"), diagnostics, delimiter=",")
    np.savez_compressed(
        os.path.join(output_path, "run_data.npz"),
        observations=observations,
        actions=actions,
        rewards=rewards,
        wind_speed=wind_v,
        wind_direction=wind_d,
        computation_times=computation_times,
        diagnostics=diagnostics,
    )


def plot_results(output_path, observations, actions, rewards, wind_v, wind_d, computation_times, opt):
    if len(rewards) == 0:
        return

    steps = np.arange(len(rewards))
    x = observations[:, 0]
    y = observations[:, 1]
    target_x = observations[0, 12]
    target_y = observations[0, 13]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(x, y, "b-", linewidth=2, label="Fast MPC")
    ax.plot(x[0], y[0], "gs", markersize=11, label="start")
    ax.plot(x[-1], y[-1], "r^", markersize=11, label="end")
    ax.plot(target_x, target_y, "mp", markersize=13, label="target")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Fast MPC trajectory (N={opt.prediction_horizon}, sub_dt={opt.sub_dt})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "trajectory.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    distances = np.sqrt((observations[:-1, 0] - target_x) ** 2 + (observations[:-1, 1] - target_y) ** 2)
    ax.plot(steps, distances, "m-", linewidth=2)
    ax.axhline(y=10.0, color="g", linestyle="--", linewidth=1.5, label="success threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance to target")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "distance.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, rewards, "r-", linewidth=1.5, label="reward")
    ax.plot(steps, np.cumsum(rewards) / (steps + 1), "b--", linewidth=1.5, label="running mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "rewards.png"), dpi=220)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(steps, np.rad2deg(actions[:, 0]), "g-", linewidth=1.5)
    ax1.set_ylabel("Sail delta (deg)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(steps, np.rad2deg(actions[:, 1]), color="orange", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Rudder delta (deg)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Actions")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "actions.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    comp_ms = computation_times * 1000.0
    ax.plot(steps, comp_ms, "c-", linewidth=1.2)
    ax.axhline(np.mean(comp_ms), color="r", linestyle="--", linewidth=1.5, label=f"mean {np.mean(comp_ms):.1f} ms")
    ax.set_xlabel("Step")
    ax.set_ylabel("Computation time (ms)")
    ax.set_title("MPC computation time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "computation_time.png"), dpi=220)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(steps, wind_v, color="#1E90FF", linewidth=1.5)
    ax1.set_ylabel("Wind speed (m/s)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(steps, np.rad2deg(wind_d), color="purple", linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Wind direction (deg)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Wind")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "wind.png"), dpi=220)
    plt.close(fig)


def run_mpc_test(opt):
    print("=" * 72)
    print("Fast MPC experiment")
    print("=" * 72)

    register_environments()
    env = gym.make(NOWIND_ENV_ID)
    if hasattr(env.unwrapped, "obstacle_min_count"):
        env.unwrapped.obstacle_min_count = 0
        env.unwrapped.obstacle_max_count = 0

    solver = "hybrid" if opt.refine and opt.solver == "candidate" else opt.solver

    mpc = FastSailboatMPC(
        env=env.unwrapped,
        prediction_horizon=opt.prediction_horizon,
        control_horizon=opt.control_horizon,
        dt=1.0,
        sub_dt=opt.sub_dt,
        integrator=opt.integrator,
        n_candidates=opt.n_candidates,
        action_levels_deg=parse_action_levels(opt.action_levels_deg),
        solver=solver,
        slsqp_starts=opt.slsqp_starts,
        parallel_starts=opt.parallel_starts,
        refine=opt.refine,
        max_iter=opt.max_iter,
        ftol=opt.ftol,
        optimizer_eps=np.deg2rad(opt.optimizer_eps_deg),
    )

    reset_options = (
        None
        if opt.wind_angle is None
        else {"initial_wind_direction_deg": float(opt.wind_angle)}
    )
    state, _ = env.reset(seed=opt.seed, options=reset_options)
    if opt.wind_angle is not None:
        print(f"Fixed wind direction: {opt.wind_angle} deg")

    observations = []
    actions = []
    rewards = []
    wind_changes_v = []
    wind_changes_d = []
    computation_times = []
    diagnostics = []

    mpc.reset_hidden_state()
    step = 0
    total_reward = 0.0
    done = False

    print(f"Initial position: x={state[0]:.2f}, y={state[1]:.2f}")
    print(f"Target position:  x={state[12]:.2f}, y={state[13]:.2f}")
    print("-" * 72)

    try:
        while not done and step < opt.max_steps:
            start = time.perf_counter()
            action = mpc.select_action(state, deterministic=True)
            comp_time = time.perf_counter() - start

            next_state, reward, terminated, truncated, info = env.step(action)

            observations.append(state.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            wind_changes_v.append(float(globals.w_change_v))
            wind_changes_d.append(float(globals.w_change_d))
            computation_times.append(float(comp_time))
            diagnostics.append(
                [
                    float(mpc.last_info.get("cost", np.nan)),
                    float(mpc.last_info.get("cost_evals", 0)),
                    float(mpc.last_info.get("candidate_count", 0)),
                    float(mpc.last_info.get("candidate_best_idx", -1)),
                    float(bool(mpc.last_info.get("optimizer_success", False))),
                    float(mpc.last_info.get("optimizer_runs", 0)),
                    float(mpc.last_info.get("parallel_workers", 0)),
                ]
            )

            total_reward += float(reward)
            state = next_state
            step += 1
            done = bool(terminated or truncated)

            if step % opt.log_interval == 0 or step == 1:
                dist = np.hypot(state[0] - state[12], state[1] - state[13])
                print(
                    f"Step {step:4d}: pos=({state[0]:7.1f}, {state[1]:7.1f}), "
                    f"dist={dist:7.1f}m, comp={comp_time * 1000.0:7.1f}ms, "
                    f"evals={mpc.last_info.get('cost_evals', 0)}"
                )
    finally:
        mpc.close()
        env.close()

    observations.append(state.copy())

    observations = np.asarray(observations, dtype=float)
    actions = np.asarray(actions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    wind_changes_v = np.asarray(wind_changes_v, dtype=float)
    wind_changes_d = np.asarray(wind_changes_d, dtype=float)
    computation_times = np.asarray(computation_times, dtype=float)
    diagnostics = np.asarray(diagnostics, dtype=float)

    final_dist = float(np.hypot(state[0] - state[12], state[1] - state[13]))
    success = final_dist <= 10.0
    mean_comp = float(np.mean(computation_times)) if len(computation_times) else 0.0
    max_comp = float(np.max(computation_times)) if len(computation_times) else 0.0
    min_comp = float(np.min(computation_times)) if len(computation_times) else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = (
        f"FastMPC_{timestamp}_{solver}_N{opt.prediction_horizon}_sub{opt.sub_dt}"
        f"_starts{opt.slsqp_starts}_par{opt.parallel_starts}"
    )
    output_path = os.path.abspath(os.path.join(opt.output_dir, output_folder))
    try:
        os.makedirs(output_path, exist_ok=True)
    except PermissionError:
        fallback_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpc_fast_results")
        output_path = os.path.abspath(os.path.join(fallback_root, output_folder))
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory was not writable; using fallback: {output_path}")

    save_arrays(
        output_path,
        observations,
        actions,
        rewards,
        wind_changes_v,
        wind_changes_d,
        computation_times,
        diagnostics,
    )

    config = vars(opt).copy()
    config.update(
        {
            "output_path": output_path,
            "steps": int(len(rewards)),
            "total_reward": float(total_reward),
            "final_distance": final_dist,
            "success": bool(success),
            "mean_computation_time_ms": mean_comp * 1000.0,
            "max_computation_time_ms": max_comp * 1000.0,
            "min_computation_time_ms": min_comp * 1000.0,
            "diagnostics_columns": [
                "best_cost",
                "cost_evals",
                "candidate_count",
                "candidate_best_idx",
                "optimizer_success",
                "optimizer_runs",
                "parallel_workers",
            ],
        }
    )
    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=json_default)

    plot_results(
        output_path,
        observations,
        actions,
        rewards,
        wind_changes_v,
        wind_changes_d,
        computation_times,
        opt,
    )

    print("=" * 72)
    print("Fast MPC result")
    print(f"  steps: {len(rewards)}")
    print(f"  total reward: {total_reward:.2f}")
    print(f"  final distance: {final_dist:.2f} m")
    print(f"  success: {success}")
    print(f"  mean computation time: {mean_comp * 1000.0:.2f} ms")
    print(f"  max computation time:  {max_comp * 1000.0:.2f} ms")
    print(f"  min computation time:  {min_comp * 1000.0:.2f} ms")
    print(f"  output: {output_path}")
    print("=" * 72)


def build_parser():
    parser = argparse.ArgumentParser(description="Fast MPC test runner")
    parser.add_argument("--wind_angle", type=int, default=None)
    parser.add_argument("--prediction_horizon", type=int, default=20)
    parser.add_argument("--control_horizon", type=int, default=1)
    parser.add_argument("--sub_dt", type=float, default=0.1)
    parser.add_argument("--integrator", type=str, default="rk4", choices=["euler", "rk2", "rk4"])
    parser.add_argument("--n_candidates", type=int, default=34)
    parser.add_argument("--action_levels_deg", type=str, default="1,2,5,10")
    parser.add_argument("--solver", type=str, default="slsqp", choices=["candidate", "hybrid", "slsqp"])
    parser.add_argument("--slsqp_starts", type=int, default=8)
    parser.add_argument("--parallel_starts", type=int, default=8)
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument("--refine", dest="refine", action="store_true")
    refine_group.add_argument("--no_refine", dest="refine", action="store_false")
    parser.set_defaults(refine=False)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--ftol", type=float, default=1e-2)
    parser.add_argument("--optimizer_eps_deg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpc_fast_results"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_mpc_test(args)
