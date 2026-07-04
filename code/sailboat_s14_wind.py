import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from typing import Optional
import globals

import math
import random
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp
from scipy.integrate import ode

FPS = 60

class SailboatWindEnv(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_Fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        ):

        EzPickle.__init__(
            self,
            render_mode,)

        # 定义常量
        super().__init__()

        #仿真时间相关
        self.dt = 0.05
        self.step_per_action = int(1.0 / self.dt)

        # ========== Dryden 风场模型参数 ==========
        self.dryden_params = {
            'h': 10.0,              # 参考高度 (m)
            'V': 8.0,               # 平均风速 (m/s)
            'L_u': 300.0,           # 纵向湍流尺度 (m)
            'L_v': 150.0,           # 横向湍流尺度 (m)
            'sigma_u': 1.2,         # 纵向湍流强度 (m/s)
            'sigma_v': 1.0,         # 横向湍流强度 (m/s)
        }
        # Dryden 滤波器状态
        self.dryden_state_u = 0.0   # 纵向风速扰动状态
        self.dryden_state_v = 0.0   # 横向风速扰动状态
        # ========================================

        self.par = {
            'm': 25900,                         # (kg),mass of the vehicle
            'Ixx': 133690,
            'Izz': 24760,
            'Ixz': 2180,                        # moment of inertia
            'a11': 970,
            'a22': 17430,
            'a44': 106500,
            'a66': 101650,
            'a24': -13160,
            'a26': -6190,
            'a46': 4730,                        # (kg),added mass coef.
            'rho_a': 1.2,                       # (kg/m^3), air density
            'As': 170,                          # (m^2), sail area, ini_area is 170
            'h0': 0.0005,                       # (m), roughness height ini is 0.0005
            'h1': 11.58,                        # (m), reference height
            'z_s': -11.58,                      # (m), (x,y,z) is the CoE
            'xs': 0,
            'ys': 0,
            'zs': -11.58,                       # (m), (x,y,z) is the CoE
            'Xce': 0.6,                         # (m), distance along the mast to the CoE
            'Xm': 0.3,                          # (m), x-coordinate of the mast
            'rho_w': 1025,                      # (kg/m^3), water density
            'Ar': 1.17,                         # (m^2), rudder area
            'd_r': 1.9,                         # rudder draft
            'zeta_r': 0.8,                      # rudder efficiency
            'x_r': -8.2,
            'z_r': -0.78,                       # (m), (x,y,z) is the CoE
            'xr': -8.2,
            'yr': 0,
            'zr': -0.78,                        # (m), (x,y,z) is the CoE
            'Ak': 8.7,                          # (m^2), keel area
            'd_k': 2.49,                        # keel draft
            'zeta_k': 0.7,                      # keel efficiency
            'x_k': 0,
            'z_k': -0.58,                       # (m), (x,y,z) is the CoE
            'xk': 0,
            'yk': 0,
            'zk': -0.58,                        # (m), (x,y,z) is the CoE
            'x_h': 0,
            'z_h': -1.18,                       # (m), (x,y,z) is the CoE
            'xh': 0,
            'yh': 0,
            'zh': -1.18,                        # (m), (x,y,z) is the CoE
            'w_c': 60000,                       # (N), crew weight 20000
            'x_c': -8,                          # (m), crew position
            'y_bm': 3.6,                        # (m), yacht beam
            'a': -5.89,
            'b': 8160,
            'c': 120000,
            'd': 50000
        }
        self.M_inv = self._compute_mass_matrix_inverse()


        # ========== 新增：预计算插值器 ==========
        # 帆系数插值器
        xdata_sail = np.linspace(-np.pi, np.pi, 73) / np.pi * 180
        yldata_sail = np.concatenate([
            np.flip([0, 0.15, 0.32, 0.48, 0.7, 0.94, 1.15, 1.3, 1.28, 1.15, 1.1, 1.05, 1, 0.9, 0.82, 0.72, 0.68, 0.56, 0.48, 0.32, 0.21, 0.08, -0.06, -0.18, -0.3, -0.4, -0.53, -0.64, -0.72, -0.84, -0.95, -1.04, -1.1, -1.14, -1.08, -0.76, 0]) * (-1),
            [0.15, 0.32, 0.48, 0.7, 0.94, 1.15, 1.3, 1.28, 1.15, 1.1, 1.05, 1, 0.9, 0.82, 0.72, 0.68, 0.56, 0.48, 0.32, 0.21, 0.08, -0.06, -0.18, -0.3, -0.4, -0.53, -0.64, -0.72, -0.84, -0.95, -1.04, -1.1, -1.14, -1.08, -0.76, 0]
        ])
        yddata_sail = np.concatenate([
            np.flip([0.1, 0.12, 0.14, 0.16, 0.19, 0.26, 0.35, 0.46, 0.54, 0.62, 0.7, 0.78, 0.9, 0.98, 1.04, 1.08, 1.16, 1.2, 1.24, 1.26, 1.28, 1.34, 1.36, 1.37, 1.33, 1.31, 1.28, 1.26, 1.25, 1.2, 1.1, 1.04, 0.88, 0.8, 0.64, 0.38, 0.1]),
            [0.12, 0.14, 0.16, 0.19, 0.26, 0.35, 0.46, 0.54, 0.62, 0.7, 0.78, 0.9, 0.98, 1.04, 1.08, 1.16, 1.2, 1.24, 1.26, 1.28, 1.34, 1.36, 1.37, 1.33, 1.31, 1.28, 1.26, 1.25, 1.2, 1.1, 1.04, 0.88, 0.8, 0.64, 0.38, 0.1]
        ])
        self.sail_cl_interp = interp1d(xdata_sail, yldata_sail, kind='cubic')
        self.sail_cd_interp = interp1d(xdata_sail, yddata_sail, kind='cubic')

        # 舵系数插值器
        xdata_rudder = np.linspace(-np.pi, np.pi, 73) / np.pi * 180
        yl = np.flip([0, 0.42, 0.73, 0.95, 1.1, 1.165, 1.18, 1.155, 1.12, 1.065, 1, 0.92, 0.83, 0.72, 0.62, 0.48, 0.33, 0.16]) * (-1)
        yldata_rudder = np.concatenate([
            np.flip([0, 0.42, 0.73, 0.95, 1.1, 1.165, 1.18, 1.155, 1.12, 1.065, 1, 0.92, 0.83, 0.72, 0.62, 0.48, 0.33, 0.16, 0, *yl]) * (-1),
            [0.42, 0.73, 0.95, 1.1, 1.165, 1.18, 1.155, 1.12, 1.065, 1, 0.92, 0.83, 0.72, 0.62, 0.48, 0.33, 0.16, 0, *yl]
        ])
        yd = np.flip([0, 0.03, 0.06, 0.1, 0.17, 0.3, 0.48, 0.74, 0.98, 1.18, 1.34, 1.5, 1.65, 1.76, 1.89, 1.97, 2.01, 2.05])
        yddata_rudder = np.concatenate([
            np.flip([0, 0.03, 0.06, 0.1, 0.17, 0.3, 0.48, 0.74, 0.98, 1.18, 1.34, 1.5, 1.65, 1.76, 1.89, 1.97, 2.01, 2.05, 2.08, *yd]),
            [0.03, 0.06, 0.1, 0.17, 0.3, 0.48, 0.74, 0.98, 1.18, 1.34, 1.5, 1.65, 1.76, 1.89, 1.97, 2.01, 2.05, 2.08, *yd]
        ])
        self.rudder_cl_interp = interp1d(xdata_rudder, yldata_rudder, kind='cubic')
        self.rudder_cd_interp = interp1d(xdata_rudder, yddata_rudder, kind='cubic')

        # 龙骨系数插值器
        xdata_keel = np.linspace(-np.pi, np.pi, 73) / np.pi * 180
        yl_k = np.flip([0, 0.425, 0.74, 0.94, 1.1, 1.17, 1.19, 1.16, 1.12, 1.07, 0.99, 0.92, 0.84, 0.74, 0.63, 0.49, 0.345, 0.185]) * (-1)
        yldata_keel = np.concatenate([
            np.flip([0, 0.425, 0.74, 0.94, 1.1, 1.17, 1.19, 1.16, 1.12, 1.07, 0.99, 0.92, 0.84, 0.74, 0.63, 0.49, 0.345, 0.185, 0, *yl_k]) * (-1),
            [0.425, 0.74, 0.94, 1.1, 1.17, 1.19, 1.16, 1.12, 1.07, 0.99, 0.92, 0.84, 0.74, 0.63, 0.49, 0.345, 0.185, 0, *yl_k]
        ])
        yd_k = np.flip([0, 0.04, 0.07, 0.1, 0.17, 0.3, 0.49, 0.76, 0.98, 1.19, 1.34, 1.5, 1.65, 1.77, 1.88, 1.96, 2.01, 2.05])
        yddata_keel = np.concatenate([
            np.flip([0, 0.04, 0.07, 0.1, 0.17, 0.3, 0.49, 0.76, 0.98, 1.19, 1.34, 1.5, 1.65, 1.77, 1.88, 1.96, 2.01, 2.05, 2.09, *yd_k]),
            [0.04, 0.07, 0.1, 0.17, 0.3, 0.49, 0.76, 0.98, 1.19, 1.34, 1.5, 1.65, 1.77, 1.88, 1.96, 2.01, 2.05, 2.09, *yd_k]
        ])
        self.keel_cl_interp = interp1d(xdata_keel, yldata_keel, kind='cubic')
        self.keel_cd_interp = interp1d(xdata_keel, yddata_keel, kind='cubic')

        # 船体阻力插值器
        xdata_hull = np.linspace(0, 6, 13)
        ydata_hull = np.array([0, 0.15, 0.35, 0.5, 0.675, 0.825, 1.175, 1.4, 2, 4.85, 9.85, 18.46, 27.5]) * 1000
        self.hull_resistance_interp = interp1d(xdata_hull, ydata_hull, kind='cubic', fill_value='extrapolate')
        # ========================================

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-np.pi/18, -np.pi/18]), high=np.array([np.pi/18, np.pi/18]), dtype=np.float32)
        self.max_sail_rate = np.pi/18
        self.max_rudder_rate = np.pi/18
        self.target_sail_angle = 0.0
        self.target_rudder_angle = 0.0

        self.state = None
        self.screen = None

        self.reset()

        self.viewer = None
        self.clock = pygame.time.Clock()

    def _compute_mass_matrix_inverse(self):
        """预计算质量矩阵的逆"""
        M_RB = np.array([
            [self.par['m'], 0, 0, 0],
            [0, self.par['m'], 0, 0],
            [0, 0, self.par['Ixx'], -self.par['Ixz']],
            [0, 0, -self.par['Ixz'], self.par['Izz']]
        ])
        M_A = np.array([
            [self.par['a11'], 0, 0, 0],
            [0, self.par['a22'], self.par['a24'], self.par['a26']],
            [0, self.par['a24'], self.par['a44'], self.par['a46']],
            [0, self.par['a26'], self.par['a46'], self.par['a66']]
        ])
        return np.linalg.inv(M_RB + M_A)

    def step(self, action):
        t = 0
        tf = 1
        dt = 0.05
        tol = 1e-6
        flag = True
        pre_dis = 0
        cur_dis = 0
        pre_x_dis = 0
        cur_x_dis = 0

        if flag:
            previous_state = np.copy(self.state[0:12])
            flag = False
        globals.current_state = self.state[0:12]
        globals.current_action = action
        globals.state = np.copy(self.state[0:16])
        globals.path.append((self.state[0], self.state[1]))
        self.target_sail_angle = np.clip(self.target_sail_angle + action[0] // (np.pi/180) * (np.pi/180), -np.pi/2, np.pi/2)
        self.target_rudder_angle = np.clip(self.target_rudder_angle + action[1] // (np.pi/180) * (np.pi/180), -np.pi/6, np.pi/6)

        # ========== 修正：在积分前一次性计算本步的角度变化 ==========
        initial_sail = globals.current_state[8]
        initial_rudder = globals.current_state[9]

        sail_diff = self.target_sail_angle - initial_sail
        rudder_diff = self.target_rudder_angle - initial_rudder

        # 一个完整step(tf=1秒)内允许的最大变化量
        max_sail_change_per_step = self.max_sail_rate * tf
        max_rudder_change_per_step = self.max_rudder_rate * tf

        # 本步实际变化量（限幅后）
        step_delta_sail = np.clip(sail_diff, -max_sail_change_per_step, max_sail_change_per_step)
        step_delta_rudder = np.clip(rudder_diff, -max_rudder_change_per_step, max_rudder_change_per_step)
        # ==============================================================

        def state_derivatives(state, action, t_local):
            """
            t_local: 当前积分时间（相对于本步开始的时间，范围0到tf）
            """
            x, y, phi, psi, u, v, p, r = state[0:8]
            nu = np.array([[u], [v], [p], [r]])
            delta_s, delta_r = action[0:2]
            y_w = 0

            # ========== 修正：使用线性插值计算当前时刻的角度 ==========
            # 而不是每次都修改 globals.current_state[8/9]
            progress = t_local / tf  # 进度比例 [0, 1]
            current_sail = initial_sail + progress * step_delta_sail
            current_rudder = initial_rudder + progress * step_delta_rudder

            sail = np.clip(current_sail, -np.pi/2, np.pi/2)
            rudder = np.clip(current_rudder, -np.pi/6, np.pi/6)
            # =========================================================

            M_RB = np.array([
                [self.par['m'], 0, 0, 0],
                [0, self.par['m'], 0, 0],
                [0, 0, self.par['Ixx'], -self.par['Ixz']],
                [0, 0, -self.par['Ixz'], self.par['Izz']]
            ])

            C_RB = np.array([
                [0,-self.par['m']*r,0,0],
                [self.par['m']*r,0,0,0],
                [0,0,0,0],
                [0,0,0,0]
            ])

            M_A = np.array([
                [self.par['a11'], 0, 0, 0],
                [0, self.par['a22'], self.par['a24'], self.par['a26']],
                [0, self.par['a24'], self.par['a44'], self.par['a46']],
                [0, self.par['a26'], self.par['a46'], self.par['a66']]
            ])

            C_A = np.array([
                [0,0,0,-self.par['a22']*nu[1,0] - self.par['a24']*nu[2,0] - self.par['a26']*nu[3,0]],
                [0, 0, 0, self.par['a11']*nu[0,0]],
                [0, 0, 0, 0],
                [self.par['a22']*nu[1,0] + self.par['a24']*nu[2,0] + self.par['a26']*nu[3,0], -self.par['a11']*nu[0,0], 0, 0]
                ])

            M = M_RB + M_A

            v_t = np.array([
                [globals.w_change_v*np.cos(globals.w_change_d)],
                [globals.w_change_v*np.sin(globals.w_change_d)],
                [0]
                ])

            numerator = abs(self.par['z_s']) * np.cos(phi) / self.par['h0']

            v_tw = np.log(numerator)/np.log(self.par['h1']/self.par['h0'])*v_t
            R1 = np.array([
                [np.cos(-psi), -np.sin(-psi), 0],
                [np.sin(-psi), np.cos(-psi), 0],
                [0, 0, 1]
            ])
            R2 = np.array([
                [1, 0, 0],
                [0, np.cos(-phi), -np.sin(-phi)],
                [0, np.sin(-phi), np.cos(-phi)]
            ])

            v_tb = R2 @ R1 @ v_tw
            V_in = np.array([[u], [v], [0]])

            cross_1 = np.array([[p],[0],[r]])
            cross_2 = np.array([[self.par['xs']],[self.par['ys']],[self.par['zs']]])
            v_awb = v_tb - V_in - np.cross(cross_1.flatten(),cross_2.flatten())
            v_awu = v_awb[0][0]
            v_awv = v_awb[1][0]
            alpha_aw = np.arctan2(v_awv,-v_awu)

            # ========== 使用预计算的帆系数插值器 ==========
            alpha_as = alpha_aw - sail
            alpha_as = (alpha_as + np.pi) % (2 * np.pi) - np.pi
            alpha_as_deg = alpha_as * 180 / np.pi

            Cls = self.sail_cl_interp(alpha_as_deg)
            Cds = self.sail_cd_interp(alpha_as_deg)
            # ============================================

            Ls = 0.5 * self.par['rho_a'] * (v_awu**2 + v_awv**2) * self.par['As'] * Cls
            Ds = 0.5 * self.par['rho_a'] * (v_awu**2 + v_awv**2) * self.par['As'] * Cds
            tau_sail = np.array([
                [Ls * np.sin(alpha_aw) - Ds * np.cos(alpha_aw)],
                [Ls * np.cos(alpha_aw) + Ds * np.sin(alpha_aw)],
                [-(Ls * np.cos(alpha_aw) + Ds * np.sin(alpha_aw)) * self.par['zs']],
                [-(Ls * np.sin(alpha_aw) - Ds * np.cos(alpha_aw)) * self.par['Xce'] * np.sin(sail) + (Ls * np.cos(alpha_aw) + Ds * np.sin(alpha_aw)) * (self.par['Xm'] - self.par['Xce'] * np.cos(sail))]
            ])

            Mzs = tau_sail[3]

            # ========== 使用预计算的舵系数插值器 ==========
            v_aru = -u+r*self.par['yr']
            v_arv = -v-r*self.par['xr']+p*self.par['zr']
            alpha_ar = np.arctan2(v_arv,-v_aru)
            alpha_a = alpha_ar - rudder
            alpha_a = (alpha_a + np.pi) % (2 * np.pi) - np.pi
            alpha_a_deg = alpha_a * 180 / np.pi

            Clr = self.rudder_cl_interp(alpha_a_deg)
            Cdr_base = self.rudder_cd_interp(alpha_a_deg)
            # ============================================

            Cdr = Cdr_base+Clr**2*self.par['Ar']/(np.pi*2*self.par['zeta_r']*self.par['d_r']**2)
            Lr = 0.5*self.par['rho_w']*self.par['Ar']*(v_aru**2+v_arv**2)*Clr
            Dr = 0.5*self.par['rho_w']*self.par['Ar']*(v_aru**2+v_arv**2)*Cdr

            tau_rudder = np.array([
                [Lr*np.sin(alpha_ar)-Dr*np.cos(alpha_ar)],
                [Lr*np.cos(alpha_ar)+Dr*np.sin(alpha_ar)],
                [-(Lr*np.cos(alpha_ar)+Dr*np.sin(alpha_ar))*self.par['zr']],
                [(Lr*np.cos(alpha_ar)+Dr*np.sin(alpha_ar))*self.par['xr']]
            ])

            Mzr = tau_rudder[3]
            tau = tau_sail + tau_rudder

            # ========== 使用预计算的龙骨系数插值器 ==========
            v_aku = -u+r*self.par['yk']
            v_akv = -v-r*self.par['xk']+p*self.par['zk']
            alpha_ak = np.arctan2(v_akv,-v_aku)
            alpha_e = alpha_ak
            alpha_e = (alpha_e + np.pi) % (2 * np.pi) - np.pi
            alpha_e_deg = alpha_e * 180 / np.pi

            clk = self.keel_cl_interp(alpha_e_deg)
            cdk_base = self.keel_cd_interp(alpha_e_deg)
            # ============================================

            cdk = cdk_base + (clk**2)*self.par['Ak']/(np.pi*2*self.par['zeta_k']*(self.par['d_k']**2))

            Lk = 0.5 * self.par['rho_w'] * self.par['Ak'] * (v_aku**2 + v_akv**2) * clk
            Dk = 0.5 * self.par['rho_w'] * self.par['Ak'] * (v_aku**2 + v_akv**2) * cdk

            D_keel = np.array([
                [-Lk * np.sin(alpha_ak) + Dk * np.cos(alpha_ak)],
                [-Lk * np.cos(alpha_ak) - Dk * np.sin(alpha_ak)],
                [-(-Lk * np.cos(alpha_ak) - Dk * np.sin(alpha_ak)) * self.par['zk']],
                [-(Lk * np.cos(alpha_ak) + Dk * np.sin(alpha_ak)) * self.par['xk']]
            ])

            # ========== 使用预计算的船体阻力插值器 ==========
            v_ahu = -u + r * self.par['yh']
            v_ahv = (-v - r * self.par['xh'] + p * self.par['zh'])/np.cos(phi)
            v_ah = np.sqrt(v_aku**2 + v_akv**2)
            alpha_ah = np.arctan2(v_ahv, -v_ahu)
            Frh = self.hull_resistance_interp(v_ah)
            # ============================================

            # 计算 D_hull
            D_hull = np.array([
                [Frh * np.cos(alpha_ah)],
                [-Frh * np.sin(alpha_ah) * np.cos(phi)],
                [Frh * np.sin(alpha_ah) * np.cos(phi) * self.par['zh']],
                [-Frh * np.sin(alpha_ah) * np.cos(phi) * self.par['xh']]
            ])

            # 计算 heel 和 yaw 的阻尼力
            J = np.array([
                [np.cos(psi), -np.sin(psi) * np.cos(phi), 0, 0],
                [np.sin(psi), np.cos(psi) * np.cos(phi), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.cos(phi)]
            ])
            eta_dot = J @ nu
            phi_dot = eta_dot[2,0]
            psi_dot = eta_dot[3,0]

            D_heelandyaw = np.array([
                [0],
                [0],
                [self.par['c'] * phi_dot * abs(phi_dot)],
                [self.par['d'] * psi_dot * abs(psi_dot) * np.cos(phi)]
            ])

            # 计算总阻尼向量 D
            D = D_keel + D_hull + D_heelandyaw

            # 计算复原力矩和内部移动质量系统（即横向重量）
            phi_deg = phi * 180 / np.pi
            M_xw = -y_w * self.par['w_c'] * self.par['y_bm'] * np.cos(phi)
            M_zw = -y_w * self.par['w_c'] * self.par['x_c'] * np.sin(abs(phi))
            G = np.array([
                [0],
                [0],
                [self.par['a'] * phi_deg**2 + self.par['b'] * phi_deg + M_xw],
                [M_zw]
            ])

            # 计算 nu_dot
            nu_dot = self.M_inv @ (-(C_RB @ nu + C_A @ nu) - D - G + tau)

            # 输出状态导数扩展与帆角
            X_dot_ext = np.concatenate([eta_dot.flatten(), nu_dot.flatten()])

            return X_dot_ext


        while t < tf:
            y = globals.current_state[0:8]

            # ========== 修正：传入当前积分时间 t ==========
            # Dormand-Prince coefficients (from MATLAB ode45)
            k1 = dt * state_derivatives(y, action, t)
            k2 = dt * state_derivatives(y + (1/5) * k1, action, t + dt/5)
            k3 = dt * state_derivatives(y + (3/40) * k1 + (9/40) * k2, action, t + 3*dt/10)
            k4 = dt * state_derivatives(y + (44/45) * k1 - (56/15) * k2 + (32/9) * k3, action, t + 4*dt/5)
            k5 = dt * state_derivatives(y + (19372/6561) * k1 - (25360/2187) * k2 +
                                        (64448/6561) * k3 - (212/729) * k4, action, t + 8*dt/9)
            k6 = dt * state_derivatives(y + (9017/3168) * k1 - (355/33) * k2 +
                                        (46732/5247) * k3 + (49/176) * k4 -
                                        (5103/18656) * k5, action, t + dt)

            # 5th-order accurate solution
            y_high = y + (35/384) * k1 + (500/1113) * k3 + (125/192) * k4 - \
                    (2187/6784) * k5 + (11/84) * k6

            # 4th-order embedded solution for error estimate
            k7 = dt * state_derivatives(y_high, action, t + dt)
            y_low = y + (5179/57600) * k1 + (7571/16695) * k3 + (393/640) * k4 - \
                    (92097/339200) * k5 + (187/2100) * k6 + (1/40) * k7
            # =============================================

            # Error estimation
            error = np.abs(y_high - y_low)

            if np.max(error) > tol:
                # Error too large, reduce timestep
                dt *= 0.9 * (tol / np.max(error))**0.2
            else:
                # Accept step
                t += dt
                globals.current_state[0:8] = y_high
                self.state[0:8] = y_high
                # Adjust step for next iteration
                dt *= 0.9 * (tol / np.max(error))**0.25

            # Prevent overshooting final time
            if t + dt > tf:
                dt = tf - t

        # ========== 积分完成后，更新最终角度 ==========
        final_sail = initial_sail + step_delta_sail
        final_rudder = initial_rudder + step_delta_rudder

        globals.current_state[8] = np.clip(final_sail, -np.pi/2, np.pi/2)
        globals.current_state[9] = np.clip(final_rudder, -np.pi/6, np.pi/6)
        # ============================================

        globals.current_time += tf
        globals.current_state[3] = np.arctan2(np.sin(globals.current_state[3]), np.cos(globals.current_state[3]))
        pre_dis = math.sqrt((previous_state[0] - globals.state[12])**2 + (previous_state[1] - globals.state[13])**2)
        cur_dis = math.sqrt((globals.current_state[0] - globals.state[12])**2 + (globals.current_state[1] - globals.state[13])**2)
        globals.current_state[10] = pre_dis - cur_dis
        pre_x_dis = abs(previous_state[0] - globals.state[12])
        cur_x_dis = abs(globals.current_state[0] - globals.state[12])
        globals.current_state[11] = pre_x_dis - cur_x_dis

        # 更新观测空间的后2位：风速和风向
        self.state[14] = globals.w_change_v
        self.state[15] = globals.w_change_d

        if globals.current_state[8] > np.pi/2:
            globals.current_state[8] = np.pi/2
        elif globals.current_state[8] < -np.pi/2:
            globals.current_state[8] = -np.pi/2

        if globals.current_state[9] > np.pi/6:
            globals.current_state[9] = np.pi/6
        elif globals.current_state[9] < -np.pi/6:
            globals.current_state[9] = -np.pi/6

        dis_to_yuandian = math.sqrt((globals.current_state[0])**2 + (globals.current_state[1])**2)

        # truncated 条件：超出边界或超时
        if dis_to_yuandian >= 750 or globals.current_time >= globals.total_time:
            truncated = True
        else:
            truncated = False

        globals.state[14:16] = np.copy(self.state[14:16])

        self.update_wind()
        reward = self.calculate_reward(previous_state, truncated)
        terminated = self.check_done()

        globals.state[0:12] = np.copy(globals.current_state)

        if terminated or truncated:
            globals.path.append((self.state[0], self.state[1]))

        if self.render_mode == 'human':
            self.render()

        return globals.state, reward, terminated, truncated, {}


    def _dryden_wind_update(self, dt, effective_w_ini_d = None):
        """
        使用 Dryden 模型更新风场

        Dryden 模型使用一阶低通滤波器来生成时间相关的湍流扰动：
        dx/dt = -V/L * x + sqrt(2*V/L) * sigma * w(t)

        其中 w(t) 是白噪声

        返回:
            wind_speed: 风速 (m/s)
            wind_direction: 风向 (rad)
        """
        w_ini_d = effective_w_ini_d if effective_w_ini_d is not None else globals.w_ini_d



        V = self.dryden_params['V']
        L_u = self.dryden_params['L_u']
        L_v = self.dryden_params['L_v']
        sigma_u = self.dryden_params['sigma_u']
        sigma_v = self.dryden_params['sigma_v']

        # 时间常数
        tau_u = L_u / V
        tau_v = L_v / V

        # 离散化系数 (一阶指数衰减)
        alpha_u = np.exp(-dt / tau_u)
        alpha_v = np.exp(-dt / tau_v)

        # 噪声增益 (保持方差稳定)
        beta_u = sigma_u * np.sqrt(1 - alpha_u**2)
        beta_v = sigma_v * np.sqrt(1 - alpha_v**2)

        # 生成白噪声
        noise_u = np.random.randn()
        noise_v = np.random.randn()

        # 更新 Dryden 状态 (一阶自回归过程)
        self.dryden_state_u = alpha_u * self.dryden_state_u + beta_u * noise_u
        self.dryden_state_v = alpha_v * self.dryden_state_v + beta_v * noise_v

        # 计算风速分量 (基础风 + 湍流扰动)
        # 基础风沿初始风向
        base_wind_x = globals.w_ini_v * np.cos(w_ini_d)
        base_wind_y = globals.w_ini_v * np.sin(w_ini_d)

        # 湍流扰动 (u 沿风向，v 垂直于风向)
        turb_along = self.dryden_state_u      # 沿风向的扰动
        turb_cross = self.dryden_state_v      # 垂直风向的扰动

        # 将扰动转换到全局坐标系
        cos_d = np.cos(w_ini_d)
        sin_d = np.sin(w_ini_d)
        turb_x = turb_along * cos_d - turb_cross * sin_d
        turb_y = turb_along * sin_d + turb_cross * cos_d

        # 合成风速
        total_wind_x = base_wind_x + turb_x
        total_wind_y = base_wind_y + turb_y

        # 转换为风速和风向
        wind_speed = np.sqrt(total_wind_x**2 + total_wind_y**2)
        wind_direction = np.arctan2(total_wind_y, total_wind_x)

        # 确保风速非负
        wind_speed = max(wind_speed, 0.1)

        return wind_speed, wind_direction

    def update_wind(self):
        """Advance the Dryden wind process by one control interval."""
        new_v, new_d = self._dryden_wind_update(
            dt=1.0,
            effective_w_ini_d=np.copy(globals.w_ini_d),
        )
        globals.w_change_v = new_v
        globals.w_change_d = new_d
        globals.w_last_v = new_v
        globals.w_last_d = new_d


    def calculate_reward(self, laststate, truncated):
        reward = 0
        reward_scale = 0.2
        reward_scale_sail = 0
        reward_scale_rudder = 0

        dis_diff = 0
        x_dis_diff = 0
        x = np.copy(globals.current_state[0])
        y = np.copy(globals.current_state[1])

        dis_diff = np.copy(globals.current_state[10])
        x_dis_diff = np.copy(globals.current_state[11])
        p = np.copy(globals.current_state[6])
        r = np.copy(globals.current_state[7])
        psi = np.copy(globals.current_state[3])
        u = np.copy(globals.current_state[4])
        v = np.copy(globals.current_state[5])

        reward += (2 * dis_diff + x_dis_diff) + 0.5 * (u / (1 + abs(v)))

        # 3. 禁航区惩罚 - 基于风向设置禁航区
        wind_angle = np.copy(self.state[14]) # 获取当前风向
        heading = np.copy(globals.current_state[3])  # 船的艏向

        # 将风向和艏向标准化到 [-pi, pi] 范围
        wind_angle = ((wind_angle + np.pi) % (2 * np.pi)) - np.pi
        heading = ((heading + np.pi) % (2 * np.pi)) - np.pi

        # 计算船艏向与风向之间的相对角度，范围为 [-pi, pi]
        relative_angle = ((heading - wind_angle + np.pi) % (2 * np.pi)) - np.pi

        # 定义禁航区 - 当艏向和风向夹角的绝对值大于150度时，处于禁航区
        no_go_zone_threshold = 150 * (np.pi / 180)  # 150度转换为弧度

        # 检查是否在禁航区内
        if abs(relative_angle) > no_go_zone_threshold:
            # 在禁航区内，给予惩罚
            reward -= 2
        else:
            reward += 0

        # ========== 动作惩罚 ==========
        sail = np.copy(globals.current_action[0])
        rudder = np.copy(globals.current_action[1])

        if abs(sail) <= np.pi/180:
            reward_scale_sail = reward_scale * 0.1
        else:
            reward -= abs(sail) // (np.pi/180) * 0.5

        if abs(rudder) <= np.pi/180:
            reward_scale_rudder = reward_scale * 0.1
        else:
            reward -= abs(rudder) // (np.pi/180) * 0.5

        reward = reward * (reward_scale + reward_scale_sail + reward_scale_rudder)

        # ========== 里程碑奖励 ==========
        dis_to_target = math.sqrt((x - globals.state[12])**2 + (y - globals.state[13])**2)
        if dis_to_target <= 375 and globals.one_check == 0:
            reward += 3
            globals.one_check = 1
        if dis_to_target <= 250 and globals.two_check == 0:
            reward += 5
            globals.two_check = 1
        if dis_to_target <= 125 and globals.three_check == 0:
            reward += 12
            globals.three_check = 1
        if dis_to_target <= 10.0:
            reward += 40

        if truncated:
            reward -= 100

        reward -= 0.2

        return reward

    def check_done(self):
        # 检查是否结束的逻辑
        x = np.copy(globals.current_state[0])
        y = np.copy(globals.current_state[1])

        distance_to_target = np.sqrt((x - self.state[12])**2 + (y - self.state[13])**2)
        reached_target = distance_to_target <= 10.0

        return reached_target

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            ):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        globals.current_state = None
        globals.current_action = None
        globals.previous_state = None
        globals.state = None
        globals.previous_action = [0, 0]
        globals.current_time = 0

        globals.path = []

        # 风场状态
        globals.w0 = [8.0, 0]
        initial_wind_direction = None
        if options is not None:
            initial_wind_direction = options.get("initial_wind_direction_rad")
            if initial_wind_direction is None and options.get("initial_wind_direction_deg") is not None:
                initial_wind_direction = np.deg2rad(float(options["initial_wind_direction_deg"]))
        if initial_wind_direction is None:
            globals.w0[1] = random.uniform(-np.pi, np.pi)
        else:
            globals.w0[1] = float(initial_wind_direction)
        globals.w0[1] = np.arctan2(np.sin(globals.w0[1]), np.cos(globals.w0[1]))

        globals.var_v = 0.5
        globals.var_d = 1/18
        globals.w_change_v = 8.0
        globals.w_change_d = np.copy(globals.w0[1])
        globals.w_last_v = globals.w0[0]
        globals.w_last_d = globals.w0[1]
        globals.w_ini_v = globals.w0[0]
        globals.w_ini_d = globals.w0[1]

        # ========== 重置 Dryden 状态 ==========
        self.dryden_state_u = 0.0
        self.dryden_state_v = 0.0
        # 更新 Dryden 参数中的平均风速
        self.dryden_params['V'] = globals.w_ini_v
        # =====================================

        globals.one_check = 0
        globals.two_check = 0
        globals.three_check = 0
        globals.total_energy = 0

        self.target_rudder_angle = 0.0
        self.target_sail_angle = 0.0

        # 初始化状态（16维）
        self.state = np.zeros(16, dtype=np.float32)

        self.state[12] = 500
        self.state[13] = 0

        # Randomize the initial heading.
        self.state[3] = random.uniform(-np.pi, np.pi)
        self.state[4] = 0

        # 初始化风速风向到观测空间
        self.state[14] = globals.w_change_v
        self.state[15] = globals.w_change_d

        globals.state = np.copy(self.state)

        return self.state, {
            "wind_speed_index": 14,
            "wind_direction_index": 15,
            "initial_wind_speed": float(globals.w0[0]),
            "initial_wind_rad": float(globals.w0[1]),
            "initial_wind_deg": float(np.rad2deg(globals.w0[1]) % 360.0),
        }

    def render(self):

        global path, end_pos, current_state
        # 渲染环境
        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((800, 600))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((800, 600))

        # 绘制背景
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # 绘制起点和终点
        # pygame.draw.circle(self.surf, (0, 255, 0), self.start_pos, 10)  # 起点为绿色圆点
        pygame.draw.circle(self.surf, (255, 0, 0), self.state[11:13], 10)  # 终点为红色圆点

        # 绘制帆船
        boat_pos = (int(self.state[0]), int(self.state[1]))
        pygame.draw.circle(self.surf, (0, 0, 255), boat_pos, 5)  # 帆船为蓝色小圆点

        # 绘制运动路径
        if len(self.path) > 1:
            pygame.draw.lines(self.surf, (255, 0, 0), False, globals.path, 2)  # 路径为红色线条

        if self.render_mode == 'human':
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)  # 控制帧率为 60 FPS
        elif self.render_mode == 'rgb_array':
            return np.array(pygame.surfarray.pixels3d(self.surf))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
