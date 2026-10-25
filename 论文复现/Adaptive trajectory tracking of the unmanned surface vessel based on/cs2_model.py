"""Cyber Ship II 三自由度平面运动模型（论文式(2)-(6)）与风/流环境干扰（式(4)-(5)）。

论文正文未给出 C、D 的具体矩阵，这里按 Fossen (2011) 公布的 CyberShip II 水动力系数
实现；非线性阻尼统一取正（耗散）形式，保证数值积分稳定。

状态量 state = [x, y, psi, u, v, r]，控制量 tau = [tau_u, tau_v, tau_r]。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

# ------------------------------------------------------------------ 主尺度
LENGTH = 1.255          # Table 1
WIDTH = 0.29
MASS = 23.8             # kg
SCALE_RATIO = 70.0      # 1:70 船模

# 惯性矩阵 M（含附加质量，M11 = m - X_udot 等）
_M11 = MASS + 2.0       # X_udot = -2.0
_M22 = MASS + 10.0      # Y_vdot = -10.0
_M23 = MASS * 0.046     # m * xg - Y_rdot, Y_rdot = 0
_M33 = 1.76 + 1.0       # Iz - N_rdot, N_rdot = -1.0
INERTIA_MATRIX = np.array([[_M11, 0.0, 0.0],
                           [0.0, _M22, _M23],
                           [0.0, _M23, _M33]])

# 线性阻尼系数（Fossen 2011, CyberShip II）
# 注意 D(1,2) = Y_r 取正号：D 必须是耗散的，负号会让艏摇-横荡耦合变成
# 正反馈，线性化后出现正实部特征值（实测 ad 谱半径 6.7，预测发散）。
_D11, _D22, _D23, _D32, _D33 = 0.72253, 0.889665, 7.250, -0.0313, 1.900
# 二次阻尼系数（正阻尼形式）
_Q11, _Q22V, _Q22R = 1.32742, 36.47287, 0.805
_Q33V, _Q33R = 3.45, 0.08

# 环境模型常数（式(4)(5)）
RHO_AIR = 1.225         # kg/m^3
A_FRONTAL = 0.29 * 0.10     # 迎风投影面积 A_f / m^2
A_LATERAL = 1.255 * 0.10    # 侧向投影面积 A_s / m^2
CX_WIND, CY_WIND, CN_WIND = 0.60, 1.00, 0.10   # 风压系数 Cx, Cy, Cn


def wrap_angle(angle):
    """把角度归一化到 (-pi, pi]。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class NoDisturbance:
    """无环境干扰。"""

    active = False
    scale = 0.0

    def force(self, eta, nu, t=0.0):
        return np.zeros(3)

    def describe(self):
        return "no disturbance"


class WindCurrentDisturbance:
    """式(4)(5)：定常风 + 定常海流。

    风、流方向采用气象学惯例（给的是“来向”），因此作用在船上的力指向
    (beta + pi)。按式(4)(5) 用 0.3 m/s 风、0.03 m/s 流算出的等效力矩对 1:70
    船模只有 1e-2 N 量级，不足以产生论文 Fig.6 中的 0.45 m 误差，故乘放大
    系数 scale 把量级折算到船模尺度（scale 由 calibrate 阶段按论文峰值误差标定）。
    """

    active = True

    def __init__(self, wind_speed=0.3, wind_dir_deg=45.0,
                 current_speed=0.03, current_dir_deg=30.0, scale=1.0,
                 max_force=(15.0, 12.0, 4.0)):
        self.wind_speed = float(wind_speed)
        self.wind_dir = np.deg2rad(wind_dir_deg)
        self.current_speed = float(current_speed)
        self.current_dir = np.deg2rad(current_dir_deg)
        self.scale = float(scale)
        self.max_force = np.asarray(max_force, dtype=float)

    def force(self, eta, nu, t=0.0):
        psi = eta[2]
        # ---- 相对风，式(4)：来向 beta_w 的风速矢量投影到船体系
        u_rw = -self.wind_speed * np.cos(self.wind_dir - psi) - nu[0]
        v_rw = -self.wind_speed * np.sin(self.wind_dir - psi) - nu[1]
        q_wind = u_rw * u_rw + v_rw * v_rw
        beta_rw = np.arctan2(v_rw, u_rw)
        tau_wind = 0.5 * RHO_AIR * q_wind * np.array([
            CX_WIND * np.cos(beta_rw) * A_FRONTAL,
            CY_WIND * np.sin(beta_rw) * A_LATERAL,
            CN_WIND * np.sin(2.0 * beta_rw) * A_LATERAL * LENGTH,
        ])

        # ---- 相对海流，式(5)：海流相当于改变阻尼项的相对速度，
        #      其等效附加力为线性阻尼矩阵作用于流速度矢量
        u_rc = -self.current_speed * np.cos(self.current_dir - psi)
        v_rc = -self.current_speed * np.sin(self.current_dir - psi)
        nu_c = np.array([u_rc, v_rc, 0.0])
        tau_current = np.array([_D11 * nu_c[0],
                                _D22 * nu_c[1] + _D23 * nu_c[2],
                                _D32 * nu_c[1] + _D33 * nu_c[2]])

        return np.clip(self.scale * (tau_wind + tau_current),
                       -self.max_force, self.max_force)

    def describe(self):
        return (f"wind {self.wind_speed} m/s @ {np.rad2deg(self.wind_dir):.0f} deg, "
                f"current {self.current_speed} m/s @ {np.rad2deg(self.current_dir):.0f} deg, "
                f"scale x{self.scale:.1f}")


class CyberShipII:
    """式(2)的 3 自由度 USV，四阶 Runge-Kutta 离散。"""

    def __init__(self, ts=0.05, disturbance=None, substeps=2):
        self.ts = float(ts)
        self.substeps = max(1, int(substeps))
        self.disturbance = disturbance if disturbance is not None else NoDisturbance()
        self._lu = lu_factor(INERTIA_MATRIX)

    # ------------------------------------------------------------------ 模型
    @staticmethod
    def kinematics(psi, nu):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([c * nu[0] - s * nu[1],
                         s * nu[0] + c * nu[1],
                         nu[2]])

    @staticmethod
    def coriolis(nu):
        u, v, r = nu
        return np.array([[0.0, 0.0, -_M22 * v - _M23 * r],
                         [0.0, 0.0, _M11 * u],
                         [_M22 * v + _M23 * r, -_M11 * u, 0.0]])

    @staticmethod
    def damping(nu):
        u, v, r = nu
        return np.array([
            [_D11 + _Q11 * abs(u), 0.0, 0.0],
            [0.0, _D22 + _Q22V * abs(v) + _Q22R * abs(r), _D23],
            [0.0, _D32, _D33 + _Q33V * abs(v) + _Q33R * abs(r)],
        ])

    def derivative(self, state, tau, t=0.0):
        eta, nu = state[:3], state[3:]
        force = tau + self.disturbance.force(eta, nu, t)
        acc = lu_solve(self._lu, force - self.coriolis(nu) @ nu - self.damping(nu) @ nu)
        return np.concatenate((self.kinematics(eta[2], nu), acc))

    def step(self, state, tau, t=0.0):
        h = self.ts / self.substeps
        x = np.asarray(state, dtype=float)
        for k in range(self.substeps):
            tk = t + k * h
            k1 = self.derivative(x, tau, tk)
            k2 = self.derivative(x + 0.5 * h * k1, tau, tk + 0.5 * h)
            k3 = self.derivative(x + 0.5 * h * k2, tau, tk + 0.5 * h)
            k4 = self.derivative(x + h * k3, tau, tk + h)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    # ------------------------------------------------------- 供 MPC 线性化用
    @staticmethod
    def input_matrix():
        """tau 线性进入动力学，B = [0; M^-1] 为常矩阵（式(6)）。"""
        return np.vstack([np.zeros((3, 3)), np.linalg.solve(INERTIA_MATRIX, np.eye(3))])

    def state_jacobian(self, state):
        """解析 Jacobian df/dx（6x6），用于 LTV-MPC 的逐点线性化。"""
        psi = state[2]
        u, v, r = state[3], state[4], state[5]
        c, s = np.cos(psi), np.sin(psi)

        jac = np.zeros((6, 6))
        # 运动学 eta_dot = R(psi) nu
        jac[0, 2] = -u * s - v * c
        jac[0, 3], jac[0, 4] = c, -s
        jac[1, 2] = u * c - v * s
        jac[1, 3], jac[1, 4] = s, c
        jac[2, 5] = 1.0

        # d(C nu)/d nu
        dc = np.array([[0.0, -_M22 * r, -_M22 * v - 2.0 * _M23 * r],
                       [_M11 * r, 0.0, _M11 * u],
                       [_M22 * v + _M23 * r - _M11 * v, _M22 * u - _M11 * u, _M23 * u]])
        # d(D nu)/d nu
        dd = np.zeros((3, 3))
        dd[0, 0] = _D11 + 2.0 * _Q11 * abs(u)
        dd[1, 1] = _D22 + 2.0 * _Q22V * abs(v) + _Q22R * abs(r)
        dd[1, 2] = _Q22R * np.sign(r) * v + _D23
        dd[2, 1] = _D32 + _Q33V * np.sign(v) * r
        dd[2, 2] = _D33 + _Q33V * abs(v) + 2.0 * _Q33R * abs(r)

        jac[3:, 3:] = -lu_solve(self._lu, dc + dd)
        return jac


def reference_error(state, eta_d):
    """式(1)：位置误差；同时返回航向误差（已归一化）。"""
    ex = state[0] - eta_d[0]
    ey = state[1] - eta_d[1]
    epsi = wrap_angle(state[2] - eta_d[2])
    return float(np.hypot(ex, ey)), float(epsi)
