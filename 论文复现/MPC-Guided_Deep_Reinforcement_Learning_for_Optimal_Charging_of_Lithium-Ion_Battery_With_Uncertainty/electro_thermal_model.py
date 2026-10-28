# -*- coding: utf-8 -*-
"""电热耦合电池模型（论文式 1-9）。

状态向量 x = [U_1, U_2, SOC, T_s, T_c]，
输入 u = [I]（充电为正），输出 y = [U_t]。

电气部分用零阶保持精确离散（RC 网络解析解），热部分用矩阵指数精确离散，
避免 Δt = 10 s 时对 R_c*C_s 这类小时间常数采用显式欧拉法导致发散。
"""

import numpy as np
from scipy.linalg import expm

from battery_parameters import (
    active_parameters,
    ohmic_resistance,
    open_circuit_voltage,
    open_circuit_voltage_slope,
    polarization_capacitance_1,
    polarization_capacitance_2,
    polarization_resistance_1,
    polarization_resistance_2,
)

U1_INDEX, U2_INDEX, SOC_INDEX, TS_INDEX, TC_INDEX = 0, 1, 2, 3, 4
STATE_DIMENSION = 5


class ElectroThermalModel:
    """标称模型 f(·)、h(·) 及其雅可比矩阵。"""

    def __init__(self, parameters=None, sample_time=None):
        self.parameters = parameters or active_parameters()
        self.sample_time = float(sample_time or self.parameters.sample_time)
        self._build_thermal_discretization()

    # -- 离散化 ------------------------------------------------------------
    def _build_thermal_discretization(self):
        """对 [T_s, T_c] 的线性热网络做精确离散化：x+ = A_d x + B_d [Q, T_a]。"""
        p = self.parameters
        ru, rc = p.convection_resistance, p.conduction_resistance
        cs, cc = p.surface_heat_capacity, p.core_heat_capacity
        state_matrix = np.array(
            [
                [-(1.0 / (ru * cs) + 1.0 / (rc * cs)), 1.0 / (rc * cs)],
                [1.0 / (rc * cc), -1.0 / (rc * cc)],
            ]
        )
        input_matrix = np.array([[0.0, 1.0 / (ru * cs)], [1.0 / cc, 0.0]])
        augmented = np.zeros((4, 4))
        augmented[:2, :2] = state_matrix
        augmented[:2, 2:] = input_matrix
        discrete = expm(augmented * self.sample_time)
        self._thermal_a = discrete[:2, :2]
        self._thermal_b = discrete[:2, 2:]  # 第 0 列对应产热 Q，第 1 列对应环境温度 T_a

    # -- 参数 --------------------------------------------------------------
    def electrical_parameters(self, soc, temperature):
        p = self.parameters
        return (
            ohmic_resistance(p, soc, temperature),
            polarization_resistance_1(p, soc, temperature),
            polarization_resistance_2(p, soc, temperature),
            polarization_capacitance_1(p, soc),
            polarization_capacitance_2(p, soc),
        )

    # -- 状态方程 ----------------------------------------------------------
    def step_electrical(self, u1, u2, soc, current, soc_disturbance=0.0, temperature=None):
        """电气状态推进（式 1-3 的精确离散）。"""
        temperature = self.parameters.ambient_temperature if temperature is None else temperature
        r0, r1, r2, c1, c2 = self.electrical_parameters(soc, temperature)
        decay_1 = np.exp(-self.sample_time / (r1 * c1))
        decay_2 = np.exp(-self.sample_time / (r2 * c2))
        next_u1 = decay_1 * u1 + (1.0 - decay_1) * r1 * current
        next_u2 = decay_2 * u2 + (1.0 - decay_2) * r2 * current
        next_soc = soc + self.sample_time * (
            current / (3600.0 * self.parameters.capacity_ah) + soc_disturbance
        )
        return next_u1, next_u2, next_soc

    def electrical_output(self, u1, u2, soc, current, temperature=None):
        """端电压输出（式 4），充电电流为正。"""
        temperature = self.parameters.ambient_temperature if temperature is None else temperature
        r0 = ohmic_resistance(self.parameters, soc, temperature)
        return open_circuit_voltage(soc) + u1 + u2 + r0 * current

    def heat_generation(self, u1, u2, soc, current, temperature=None):
        """产热 Q = |I(U_t - U_oc)|（式 7）。"""
        temperature = self.parameters.ambient_temperature if temperature is None else temperature
        r0 = ohmic_resistance(self.parameters, soc, temperature)
        return abs(current * (u1 + u2 + r0 * current))

    def step_thermal(self, state, current):
        """热状态推进（式 5-6 的精确离散）。"""
        state = np.asarray(state, dtype=float)
        u1, u2, soc, ts, tc = state
        heat = self.heat_generation(u1, u2, soc, current, temperature=tc)
        return (
            self._thermal_a @ np.array([ts, tc])
            + self._thermal_b[:, 0] * heat
            + self._thermal_b[:, 1] * self.parameters.ambient_temperature
        )

    def step(self, state, current, soc_disturbance=0.0):
        """完整状态推进：x_{k+1} = f(x_k, u_k)。"""
        state = np.asarray(state, dtype=float)
        u1, u2, soc = state[U1_INDEX], state[U2_INDEX], state[SOC_INDEX]
        next_u1, next_u2, next_soc = self.step_electrical(
            u1, u2, soc, current, soc_disturbance, temperature=state[TC_INDEX]
        )
        thermal = self.step_thermal(state, current)
        return np.array([next_u1, next_u2, next_soc, thermal[0], thermal[1]])


    def terminal_voltage(self, state, current):
        state = np.asarray(state, dtype=float)
        return self.electrical_output(state[U1_INDEX], state[U2_INDEX], state[SOC_INDEX],
                                      current, temperature=state[TC_INDEX])

    # -- 线性化（式 13）----------------------------------------------------
    def linearize(self, state, previous_current):
        """在 (x_k, u_{k-1}) 处线性化，返回 A_k、B_k、δ_k。"""
        state = np.asarray(state, dtype=float)
        u1, u2, soc, ts, tc = state
        r0, r1, r2, c1, c2 = self.electrical_parameters(soc, tc)
        decay_1 = np.exp(-self.sample_time / (r1 * c1))
        decay_2 = np.exp(-self.sample_time / (r2 * c2))

        a_matrix = np.eye(STATE_DIMENSION)
        a_matrix[U1_INDEX, U1_INDEX] = decay_1
        a_matrix[U2_INDEX, U2_INDEX] = decay_2

        b_matrix = np.zeros(STATE_DIMENSION)
        b_matrix[U1_INDEX] = (1.0 - decay_1) * r1
        b_matrix[U2_INDEX] = (1.0 - decay_2) * r2
        b_matrix[SOC_INDEX] = self.sample_time / (3600.0 * self.parameters.capacity_ah)

        # 产热项对状态与输入的偏导（|·| 的符号在上一时刻取值处确定）
        heat_value = previous_current * (u1 + u2 + r0 * previous_current)
        sign = 1.0 if heat_value >= 0.0 else -1.0
        heat_derivative_current = sign * ((u1 + u2) + 2.0 * r0 * previous_current)
        heat_derivative_u1 = sign * previous_current
        heat_derivative_u2 = sign * previous_current

        a_matrix[TS_INDEX, U1_INDEX] = self._thermal_b[0, 0] * heat_derivative_u1
        a_matrix[TS_INDEX, U2_INDEX] = self._thermal_b[0, 0] * heat_derivative_u2
        a_matrix[TS_INDEX, TS_INDEX] = self._thermal_a[0, 0]
        a_matrix[TS_INDEX, TC_INDEX] = self._thermal_a[0, 1]

        a_matrix[TC_INDEX, U1_INDEX] = self._thermal_b[1, 0] * heat_derivative_u1
        a_matrix[TC_INDEX, U2_INDEX] = self._thermal_b[1, 0] * heat_derivative_u2
        a_matrix[TC_INDEX, TS_INDEX] = self._thermal_a[1, 0]
        a_matrix[TC_INDEX, TC_INDEX] = self._thermal_a[1, 1]

        b_matrix[TS_INDEX] = self._thermal_b[0, 0] * heat_derivative_current
        b_matrix[TC_INDEX] = self._thermal_b[1, 0] * heat_derivative_current

        offset = self.step(state, previous_current) - a_matrix @ state - b_matrix * previous_current
        return a_matrix, b_matrix, offset

    def output_jacobian(self, state, current):
        """输出方程 h(·) 对状态与输入的偏导。"""
        state = np.asarray(state, dtype=float)
        soc, tc = state[SOC_INDEX], state[TC_INDEX]
        r0 = ohmic_resistance(self.parameters, soc, tc)
        state_gradient = np.zeros(STATE_DIMENSION)
        state_gradient[U1_INDEX] = 1.0
        state_gradient[U2_INDEX] = 1.0
        state_gradient[SOC_INDEX] = open_circuit_voltage_slope(soc)
        return state_gradient, r0


def initial_state(soc, surface_temperature, core_temperature=None):
    """构造初始状态向量 [U_1, U_2, SOC, T_s, T_c]。"""
    core_temperature = surface_temperature if core_temperature is None else core_temperature
    return np.array([0.0, 0.0, float(soc), float(surface_temperature), float(core_temperature)])
