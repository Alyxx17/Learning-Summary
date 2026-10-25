"""正弦参考轨迹（论文式(33)）。

论文写的是 x_d = t、y_d = 10 sin(2t)，但按该式在 18 m 内会跑出约 5.7 个周期，
与 Fig.5 的形状不符。Fig.5 实际是振幅 10 m、周期 12 m、x 由 0 到 18 m 的正弦，
因此这里取 y_d = 10 sin(2*pi*x_d/12)。x_d 以恒定参考纵速推进，与 Fig.5 中
x 随时间均匀增长一致。
"""
from __future__ import annotations

import numpy as np


class SinusoidReference:
    """x_d = vx * t，y_d = A * sin(2*pi*x_d / P)。"""

    def __init__(self, amplitude=10.0, period=12.0, vx=0.24):
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.vx = float(vx)
        self.omega = 2.0 * np.pi / self.period

    # ------------------------------------------------------------------ 轨迹
    def position(self, t):
        x = self.vx * np.asarray(t, dtype=float)
        y = self.amplitude * np.sin(self.omega * x)
        return x, y

    def state(self, t):
        """返回 (eta_d, nu_d)：eta_d = [x_d, y_d, psi_d]，nu_d = [u_d, v_d, r_d]。"""
        x, y = self.position(t)
        slope = self.amplitude * self.omega * np.cos(self.omega * x)
        ddx = self.vx
        ddy = slope * self.vx
        # 二阶导（x_d 匀速，ddx = 0）
        dddy = -self.amplitude * self.omega ** 2 * np.sin(self.omega * x) * self.vx ** 2
        speed2 = ddx * ddx + ddy * ddy
        psi = np.arctan2(ddy, ddx)
        r = (ddx * dddy - ddy * 0.0) / speed2
        u = np.sqrt(speed2)
        return (np.array([x, y, psi]),
                np.array([u, 0.0, r]))

    def state_array(self, times):
        """批量采样（向量化），返回 (M, 3) 的参考位姿（供 MPC 预测时域使用）。"""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        x = self.vx * t
        y = self.amplitude * np.sin(self.omega * x)
        ddy = self.amplitude * self.omega * np.cos(self.omega * x) * self.vx
        psi = np.arctan2(ddy, self.vx)
        return np.column_stack((x, y, psi))

    def sample(self, t_end, count=400):
        """整段轨迹的高密度采样（用于绘制期望轨迹）。"""
        t = np.linspace(0.0, float(t_end), int(count))
        x, y = self.position(t)
        return t, x, y
