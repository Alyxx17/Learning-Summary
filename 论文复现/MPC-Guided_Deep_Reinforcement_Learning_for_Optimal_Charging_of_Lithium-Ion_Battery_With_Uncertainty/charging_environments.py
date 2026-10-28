# -*- coding: utf-8 -*-
"""充电环境：真实系统（含未建模 SOC 误差）与两种控制器的交互接口。

* StandaloneDRLChargingEnvironment : 单独 DRL，状态 s = [SOC, T_s, U_t]（3 维）
* MPCGuidedDRLChargingEnvironment  : MPC 引导 DRL，状态 s^l = [z^m, z, u^m]（7 维）

奖励按论文式 (22)：
    r = r_volt + r_temp + r_step
    r_volt = λ_V (U_t - V_max) 若 U_t ≥ V_max，否则 0
    r_temp = λ_T (T_s - T_s,max) 若 T_s ≥ T_s,max，否则 0
    r_step = λ_S

真实系统相对标称模型只有 SOC 的未建模误差（式 31）：dSOC/dt 额外叠加 p = 0.001 sin(X)。
"""

import numpy as np

from battery_parameters import (
    MAXIMUM_CURRENT,
    MAXIMUM_SURFACE_TEMPERATURE,
    MAXIMUM_VOLTAGE,
    SOC_PERTURBATION_AMPLITUDE,
    TARGET_SOC,
    active_parameters,
)
from electro_thermal_model import SOC_INDEX, TC_INDEX, TS_INDEX, ElectroThermalModel, initial_state
from linear_mpc import LinearModelPredictiveController
from unscented_kalman_filter import ElectricalStateEstimator

# 论文 Section IV-B 的奖励系数
VOLTAGE_PENALTY = -1e2
TEMPERATURE_PENALTY = -1.0
STEP_PENALTY = -0.1

# 状态归一化（网络输入尺度统一，避免 SOC≈0.2 与 T≈25 量级悬殊）
TEMPERATURE_OFFSET = 25.0
TEMPERATURE_SCALE = 20.0
VOLTAGE_OFFSET = 3.2
VOLTAGE_SCALE = 0.5
CURRENT_SCALE = 46.0

MAXIMUM_STEPS = 600

# 动作映射与探索噪声。
# 单独 DRL：u = 23(tanh(z)+1) ∈ [0, 46] A（对称映射，无裁剪死区）。
#   曾试用原文可能采用的 u = 46·tanh(z) 再裁剪到 [0, 46]（初始输出 0 A），
#   理由是原文 Fig.4(c)(d) 有 100 min 的起点；但核对后发现：
#     - 原文 Fig.4(a) 中 DRL 的 ep1 回报约 -400（0 A 恰好对应 -60 = 600 步 × λS）；
#     - 原文 Fig.4(d) 中 DRL 的 ep1 约 78 min，100 min 是 MPC-guided 的起点；
#     - 实测 46·tanh 裁剪后平均电流仅 0.29 A，agent 30 回合内学不动，且每回合跑满 600 步。
#   因此仍采用对称映射。
# MPC-guided：补偿电流 u^l = 6·tanh(z) ∈ [-6, 6] A，本身对称。
# noise_scale 使 OU 噪声的稳态标准差约为动作区间的 12%。
ACTION_SETTINGS = {
    "drl": {
        "state_dimension": 3,
        "action_scale": MAXIMUM_CURRENT / 2.0,
        "action_bias": MAXIMUM_CURRENT / 2.0,
        "noise_scale": 15.0,
    },
    "mpc_drl": {
        "state_dimension": 7,
        "action_scale": 6.0,
        "action_bias": 0.0,
        "noise_scale": 3.0,
    },
}


def normalize_measurements(soc, surface_temperature, terminal_voltage):
    return np.array(
        [
            float(soc),
            (float(surface_temperature) - TEMPERATURE_OFFSET) / TEMPERATURE_SCALE,
            (float(terminal_voltage) - VOLTAGE_OFFSET) / VOLTAGE_SCALE,
        ],
        dtype=np.float32,
    )


def compute_reward(terminal_voltage, surface_temperature):
    reward = STEP_PENALTY
    if terminal_voltage >= MAXIMUM_VOLTAGE:
        reward += VOLTAGE_PENALTY * (terminal_voltage - MAXIMUM_VOLTAGE)
    if surface_temperature >= MAXIMUM_SURFACE_TEMPERATURE:
        reward += TEMPERATURE_PENALTY * (surface_temperature - MAXIMUM_SURFACE_TEMPERATURE)
    return float(reward)


class _ChargingEnvironmentBase:
    """两种环境共用的记录、奖励与终止逻辑。"""

    state_dimension = 0

    def __init__(self, maximum_steps=MAXIMUM_STEPS, target_soc=TARGET_SOC, random_generator=None):
        self.maximum_steps = int(maximum_steps)
        self.target_soc = float(target_soc)
        self.random = random_generator or np.random.RandomState()
        self.parameters = active_parameters()
        self.model = ElectroThermalModel(self.parameters)
        self.reset()

    # -- 需要子类实现 ------------------------------------------------------
    def observation(self):
        raise NotImplementedError

    # -- 公共流程 ----------------------------------------------------------
    def reset(self):
        initial_soc = self.random.uniform(0.2, 0.3)
        initial_temperature = self.random.uniform(25.0, 27.0)
        self.state = initial_state(initial_soc, initial_temperature)
        self.step_index = 0
        self.charging_time = None
        self.last_current = 0.0
        self.last_terminal_voltage = float(self.model.terminal_voltage(self.state, 0.0))
        self.maximum_temperature = float(initial_temperature)
        self.maximum_voltage = self.last_terminal_voltage
        self.accumulated_reward = 0.0
        self.done = False
        self._after_reset()
        return self.observation()

    def _after_reset(self):
        pass

    def _record_and_reward(self, current, terminal_voltage):
        surface_temperature = float(self.state[TS_INDEX])
        self.last_current = float(current)
        self.last_terminal_voltage = float(terminal_voltage)
        self.maximum_temperature = max(self.maximum_temperature, surface_temperature)
        self.maximum_voltage = max(self.maximum_voltage, terminal_voltage)
        if self.charging_time is None and self.state[SOC_INDEX] >= self.target_soc:
            self.charging_time = (self.step_index + 1) * self.parameters.sample_time / 60.0

        reward = compute_reward(terminal_voltage, surface_temperature)
        self.accumulated_reward += reward
        self.step_index += 1
        self.done = (
            self.state[SOC_INDEX] >= self.target_soc
            or self.step_index >= self.maximum_steps
            or self.state[SOC_INDEX] > 1.0
        )
        info = {
            "soc": float(self.state[SOC_INDEX]),
            "surface_temperature": surface_temperature,
            "terminal_voltage": float(terminal_voltage),
            "current": float(current),
            "time_minutes": self.step_index * self.parameters.sample_time / 60.0,
            "charging_time": self.charging_time,
        }
        return float(reward), self.done, info

    def _soc_disturbance(self):
        return SOC_PERTURBATION_AMPLITUDE * np.sin(self.random.standard_normal())


class StandaloneDRLChargingEnvironment(_ChargingEnvironmentBase):
    """单独 DRL：智能体直接输出总电流 u ∈ [0, 46] A，与环境为真实系统。"""

    state_dimension = 3

    def observation(self):
        return normalize_measurements(
            self.state[SOC_INDEX], self.state[TS_INDEX], self.last_terminal_voltage
        )

    def step(self, action):
        current = float(np.clip(action, 0.0, MAXIMUM_CURRENT))
        self.state = self.model.step(self.state, current, soc_disturbance=self._soc_disturbance())
        terminal_voltage = self.model.terminal_voltage(self.state, current)
        reward, done, info = self._record_and_reward(current, terminal_voltage)
        info["baseline_current"] = 0.0
        info["compensation_current"] = current
        return self.observation(), reward, done, info


class MPCGuidedDRLChargingEnvironment(_ChargingEnvironmentBase):
    """MPC 引导 DRL：MPC 给出基线 u^m，智能体输出补偿 u^l ∈ [-6, 6] A，u = clip(u^m + u^l)。"""

    state_dimension = 7

    def __init__(
        self,
        maximum_steps=MAXIMUM_STEPS,
        target_soc=TARGET_SOC,
        random_generator=None,
        maximum_compensation=6.0,
        initial_soc_estimate=None,
        mpc_horizon=5,
        mpc_soc_weight=1e2,
        mpc_current_change_weight=1e2,
        mpc_terminal_soc_weight=0.1,
        use_estimator=True,
        initial_soc_uncertainty=1e-6,
    ):
        self.maximum_compensation = float(maximum_compensation)
        self.initial_soc_estimate = initial_soc_estimate
        self.use_estimator = bool(use_estimator)
        self.mpc = None
        self.mpc_settings = {
            "horizon": mpc_horizon,
            "soc_weight": mpc_soc_weight,
            "current_change_weight": mpc_current_change_weight,
            "terminal_soc_weight": mpc_terminal_soc_weight,
            "target_soc": target_soc,
        }
        # 训练时每个 episode 的初始 SOC 由库仑计数给出（论文随机采样 0.2-0.3），
        # 因此初始不确定度取小值。若沿用论文中"估计初值 0.5"的 P0 = 2.45，
        # 首次量测更新会把本来正确的估计冲走，标称模型与真实系统的偏差反而变大。
        self.initial_soc_uncertainty = float(initial_soc_uncertainty)
        super().__init__(maximum_steps=maximum_steps, target_soc=target_soc,
                         random_generator=random_generator)

    def _after_reset(self):
        if self.mpc is None:
            self.mpc = LinearModelPredictiveController(self.model, **self.mpc_settings)

        estimate = (
            float(self.state[SOC_INDEX])
            if self.initial_soc_estimate is None
            else float(self.initial_soc_estimate)
        )
        self.estimator = ElectricalStateEstimator(
            self.model,
            initial_state=(0.0, 0.0, estimate),
            initial_covariance=np.diag([1e-7, 1e-7, self.initial_soc_uncertainty]),
        )
        # 标称模型状态：热状态与真实系统初值一致，电气状态由观测器给出
        self.nominal_state = np.array(
            [0.0, 0.0, estimate, self.state[TS_INDEX], self.state[TC_INDEX]], dtype=float
        )
        self.baseline_current = 0.0
        self.maximum_compensation_used = 0.0

    def observation(self):
        nominal_voltage = self.model.terminal_voltage(self.nominal_state, self.baseline_current)
        nominal = normalize_measurements(
            self.nominal_state[SOC_INDEX], self.nominal_state[TS_INDEX], nominal_voltage
        )
        real = normalize_measurements(
            self.state[SOC_INDEX], self.state[TS_INDEX], self.last_terminal_voltage
        )
        return np.concatenate([nominal, real, [self.baseline_current / CURRENT_SCALE]]).astype(
            np.float32
        )

    def step(self, action):
        compensation = float(np.clip(action, -self.maximum_compensation, self.maximum_compensation))
        self.maximum_compensation_used = max(
            self.maximum_compensation_used, abs(compensation)
        )

        baseline, _ = self.mpc.solve(self.nominal_state, self.baseline_current)
        current = float(np.clip(baseline + compensation, 0.0, MAXIMUM_CURRENT))
        self.baseline_current = baseline

        self.state = self.model.step(self.state, current, soc_disturbance=self._soc_disturbance())
        terminal_voltage = self.model.terminal_voltage(self.state, current)

        thermal = self.model.step_thermal(self.nominal_state, current)
        self.estimator.temperature = float(thermal[1])
        if self.use_estimator:
            electrical = self.estimator.observe(terminal_voltage, current)
        else:
            electrical = self.model.step_electrical(
                self.nominal_state[0], self.nominal_state[1], self.nominal_state[SOC_INDEX], current
            )
        self.nominal_state = np.array(
            [electrical[0], electrical[1], electrical[2], thermal[0], thermal[1]], dtype=float
        )

        reward, done, info = self._record_and_reward(current, terminal_voltage)
        info["baseline_current"] = baseline
        info["compensation_current"] = compensation
        return self.observation(), reward, done, info
