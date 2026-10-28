# -*- coding: utf-8 -*-
"""线性化模型预测控制模块（论文 Section III-C，式 13-15）。

每个采样时刻在当前状态 (x_k^m, u_{k-1}^m) 处线性化标称模型，得到
    x_{k+1}^m = A_k^m x_k^m + B_k^m u_k^m + δ_k^m,
并在预测域内视为 LTI，从而把式 (14)(15) 化成一个小规模凸二次规划：

    min  Σ_{i=0}^{N-1} ||SOC_{k+i} - SOC_d||_P^2 + ||Δu_{k+i}||_Q^2 + ||SOC_{k+N} - SOC_d||_R^2
    s.t. 0 ≤ u ≤ I_max,  T_s ≤ T_s,max,  U_t ≤ V_max

其中 Δu 项按 I_max 无量纲化后加权（Δu 单位是安培、SOC 是 0-1 小数，直接使用
Q = 1e2 会让 Δu 项比 SOC 项大约三个数量级，控制器几乎不敢改变电流）。
二次规划由自带的内点法求解（不引入额外依赖）。
"""

import numpy as np

from battery_parameters import (
    MAXIMUM_CURRENT,
    MAXIMUM_SURFACE_TEMPERATURE,
    MAXIMUM_VOLTAGE,
    TARGET_SOC,
    open_circuit_voltage,
    open_circuit_voltage_slope,
)
from electro_thermal_model import SOC_INDEX, TS_INDEX, U1_INDEX, U2_INDEX


# ---------------------------------------------------------------------------
# 小规模凸二次规划求解
# ---------------------------------------------------------------------------
def solve_convex_qp(
    hessian,
    gradient,
    constraint_matrix,
    upper_bound,
    initial_guess=None,
    tolerance=1e-10,
    max_iterations=None,
):
    """求解 min 0.5 x'Hx + g'x  s.t.  Ax ≤ b，返回 (x, 是否可行)。

    采用原始有效集法（Nocedal & Wright, Algorithm 16.3）。问题规模很小
    （N = 5 个决策变量、约 20 条不等式），有效集法在这里比内点法更稳：
    内点法在 u = 0（下界恰好起作用）这类边界起点上步长会退化。
    """
    dimension = hessian.shape[0]
    constraint_count = constraint_matrix.shape[0]
    symmetric_hessian = 0.5 * (hessian + hessian.T) + 1e-10 * np.eye(dimension)

    if constraint_count == 0:
        return -np.linalg.solve(symmetric_hessian, gradient), True

    max_iterations = max_iterations or (4 * dimension + 20)

    x = _feasible_start(constraint_matrix, upper_bound, initial_guess)
    if x is None:
        return _solve_qp_with_slsqp(
            symmetric_hessian, gradient, constraint_matrix, upper_bound, np.zeros(dimension)
        ), True

    active = [
        index
        for index in range(constraint_count)
        if upper_bound[index] - constraint_matrix[index] @ x <= 1e-12
    ]

    for _ in range(max_iterations):
        base_gradient = symmetric_hessian @ x + gradient
        if active:
            active_matrix = constraint_matrix[active]
            kkt = np.zeros((dimension + len(active), dimension + len(active)))
            kkt[:dimension, :dimension] = symmetric_hessian
            kkt[:dimension, dimension:] = active_matrix.T
            kkt[dimension:, :dimension] = active_matrix
            rhs = np.concatenate([-base_gradient, np.zeros(len(active))])
            solution = np.linalg.solve(kkt, rhs)
            direction, multipliers = solution[:dimension], solution[dimension:]
        else:
            direction = -np.linalg.solve(symmetric_hessian, base_gradient)
            multipliers = np.zeros(0)

        if np.max(np.abs(direction)) <= tolerance:
            if multipliers.size == 0 or np.min(multipliers) >= -1e-12:
                violation = np.max(constraint_matrix @ x - upper_bound)
                return x, bool(violation <= 1e-6)
            del active[int(np.argmin(multipliers))]
            continue

        # 沿方向前进，遇到第一条阻挡约束就停下
        step_length = 1.0
        blocking = None
        distance = constraint_matrix @ direction
        for index in range(constraint_count):
            if index in active or distance[index] <= 1e-12:
                continue
            limit = (upper_bound[index] - constraint_matrix[index] @ x) / distance[index]
            if limit < step_length:
                step_length = max(limit, 0.0)
                blocking = index
        x = x + step_length * direction
        if blocking is not None:
            active.append(blocking)

    violation = np.max(constraint_matrix @ x - upper_bound)
    return x, bool(violation <= 1e-6)


def _feasible_start(constraint_matrix, upper_bound, initial_guess):
    """找一个满足 Ax ≤ b 的初始点；找不到则返回 None。"""
    candidates = []
    if initial_guess is not None:
        candidates.append(np.asarray(initial_guess, dtype=float).reshape(-1))
    candidates.append(np.zeros(constraint_matrix.shape[1]))
    for candidate in candidates:
        if np.max(constraint_matrix @ candidate - upper_bound) <= 1e-9:
            return candidate
    return None


def _solve_qp_with_slsqp(hessian, gradient, constraint_matrix, upper_bound, start):
    from scipy.optimize import minimize

    result = minimize(
        fun=lambda z: 0.5 * z @ hessian @ z + gradient @ z,
        x0=start,
        jac=lambda z: hessian @ z + gradient,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda z: upper_bound - constraint_matrix @ z,
                "jac": lambda z: -constraint_matrix,
            }
        ],
        method="SLSQP",
        options={"maxiter": 300, "ftol": 1e-12},
    )
    return np.asarray(result.x, dtype=float)


# ---------------------------------------------------------------------------
# MPC 控制器
# ---------------------------------------------------------------------------
class LinearModelPredictiveController:
    """基于逐点线性化的 LTI 预测控制器。"""

    def __init__(
        self,
        model,
        horizon=5,
        soc_weight=1e2,
        current_change_weight=1e2,
        terminal_soc_weight=0.1,
        target_soc=TARGET_SOC,
        maximum_current=MAXIMUM_CURRENT,
        maximum_voltage=MAXIMUM_VOLTAGE,
        maximum_surface_temperature=MAXIMUM_SURFACE_TEMPERATURE,
        output_linearization_passes=2,
        current_change_normalization=MAXIMUM_CURRENT,
    ):
        self.model = model
        self.horizon = int(horizon)
        self.soc_weight = float(soc_weight)
        self.current_change_weight = float(current_change_weight)
        self.terminal_soc_weight = float(terminal_soc_weight)
        self.target_soc = float(target_soc)
        self.maximum_current = float(maximum_current)
        self.maximum_voltage = float(maximum_voltage)
        self.maximum_surface_temperature = float(maximum_surface_temperature)
        self.output_linearization_passes = max(1, int(output_linearization_passes))
        # 电流变化率按 I_max 无量纲化：Δu 的量级是 A，而 SOC 是 0-1 的小数，
        # 若直接使用 Q = 1e2，Δu 项会比 SOC 项大三个数量级，控制器几乎不敢动电流。
        self.current_change_normalization = float(current_change_normalization)

    # -- 预测方程 ----------------------------------------------------------
    def _condense(self, state, a_matrix, b_matrix, offset, previous_current, reference):
        n = self.horizon
        dimension = state.size

        powers = [np.eye(dimension)]
        for _ in range(n):
            powers.append(powers[-1] @ a_matrix)

        biases = [np.zeros(dimension)]
        for i in range(1, n + 1):
            biases.append(a_matrix @ biases[i - 1] + offset)

        free_response = [powers[i] @ state + biases[i] for i in range(n + 1)]

        gains = np.zeros((n + 1, n, dimension))
        for i in range(1, n + 1):
            for k in range(i):
                gains[i, k] = powers[i - 1 - k] @ b_matrix

        hessian = np.zeros((n, n))
        gradient = np.zeros(n)

        def add_squared_term(weight, coefficients, constant):
            hessian[:] += 2.0 * weight * np.outer(coefficients, coefficients)
            gradient[:] += 2.0 * weight * constant * coefficients

        # 中间域 SOC 跟踪：i = 0, ..., N-1（式 14 第二项前的求和项）
        for i in range(n):
            add_squared_term(
                self.soc_weight,
                gains[i, :, SOC_INDEX].copy(),
                free_response[i][SOC_INDEX] - self.target_soc,
            )
        # 终端 SOC 跟踪
        add_squared_term(
            self.terminal_soc_weight,
            gains[n, :, SOC_INDEX].copy(),
            free_response[n][SOC_INDEX] - self.target_soc,
        )
        # 电流变化率惩罚（Δu 以 I_max 为单位，与 SOC 的 0-1 尺度对齐）
        normalized_weight = self.current_change_weight / self.current_change_normalization**2
        for i in range(n):
            coefficients = np.zeros(n)
            coefficients[i] = 1.0
            constant = 0.0
            if i == 0:
                constant = -previous_current
            else:
                coefficients[i - 1] = -1.0
            add_squared_term(normalized_weight, coefficients, constant)

        # 不等式约束
        rows = []
        bounds = []
        identity = np.eye(n)
        rows.append(-identity)                       # -u ≤ 0
        bounds.append(np.zeros(n))
        rows.append(identity)                        # u ≤ I_max
        bounds.append(np.full(n, self.maximum_current))

        temperature_limit = max(self.maximum_surface_temperature, state[TS_INDEX])
        for i in range(1, n + 1):
            rows.append(gains[i, :, TS_INDEX].reshape(1, n))
            bounds.append(np.array([temperature_limit - free_response[i][TS_INDEX]]))

        _, r0 = self.model.output_jacobian(state, previous_current)
        for i in range(1, n + 1):
            soc_reference = free_response[i][SOC_INDEX] + gains[i, :, SOC_INDEX] @ reference
            slope = float(open_circuit_voltage_slope(soc_reference))
            ocv_reference = float(open_circuit_voltage(soc_reference))
            coefficients = (
                gains[i, :, U1_INDEX] + gains[i, :, U2_INDEX] + slope * gains[i, :, SOC_INDEX]
            ).copy()
            coefficients[i - 1] += r0
            # U_oc 在 soc_reference 处线性化：U_oc ≈ U_oc(ref) + slope*(SOC_i - ref)，
            # 其中 SOC_i = free_response[i] + gains[i,:]*u，故常数项必须含 slope*free_response[i]
            constant = (
                free_response[i][U1_INDEX]
                + free_response[i][U2_INDEX]
                + ocv_reference
                + slope * (free_response[i][SOC_INDEX] - soc_reference)
            )
            rows.append(coefficients.reshape(1, n))
            bounds.append(np.array([self.maximum_voltage - constant]))

        return hessian, gradient, np.vstack(rows), np.concatenate(bounds)

    # -- 求解 --------------------------------------------------------------
    def solve(self, nominal_state, previous_current):
        """返回 (第一个最优输入, 附加信息)。"""
        state = np.asarray(nominal_state, dtype=float)
        previous_current = float(previous_current)
        a_matrix, b_matrix, offset = self.model.linearize(state, previous_current)

        reference = np.full(self.horizon, previous_current)
        solution = reference.copy()
        feasible = True
        for _ in range(self.output_linearization_passes):
            hessian, gradient, rows, bounds = self._condense(
                state, a_matrix, b_matrix, offset, previous_current, reference
            )
            candidate, feasible = solve_convex_qp(
                hessian,
                gradient,
                rows,
                bounds,
                initial_guess=np.clip(solution, 0.0, self.maximum_current),
            )
            if not feasible:
                candidate = np.zeros(self.horizon)
            solution = candidate
            reference = candidate

        current = float(np.clip(solution[0], 0.0, self.maximum_current))
        return current, {"prediction": solution, "feasible": feasible}
