# -*- coding: utf-8 -*-
"""单独 MPC 控制器的充电仿真（论文 Fig. 3 与 Table II 的数据来源）。

流程与论文一致：MPC 基于标称模型滚动优化得到基线电流 u^m，
UKF 用真实端电压修正 SOC 估计；真实系统仅在 SOC 动态上叠加未建模误差 p。
"""

import numpy as np

from battery_parameters import (
    INITIAL_SOC,
    MAXIMUM_VOLTAGE,
    SOC_PERTURBATION_AMPLITUDE,
    TARGET_SOC,
    active_parameters,
)
from electro_thermal_model import ElectroThermalModel, initial_state
from linear_mpc import LinearModelPredictiveController
from unscented_kalman_filter import ElectricalStateEstimator


def simulate_standalone_mpc(
    seed=None,
    horizon_minutes=8.0,
    initial_soc=INITIAL_SOC,
    initial_temperature=25.0,
    initial_soc_estimate=0.5,
    parameters=None,
    mpc_settings=None,
):
    """运行一次单独 MPC 充电过程，返回时间序列与统计量。"""
    random = np.random.RandomState(seed)
    model = ElectroThermalModel(parameters or active_parameters())
    mpc = LinearModelPredictiveController(model, **(mpc_settings or {}))
    estimator = ElectricalStateEstimator(
        model, initial_state=(0.0, 0.0, initial_soc_estimate)
    )

    real_state = initial_state(initial_soc, initial_temperature)
    nominal_state = np.array(
        [0.0, 0.0, initial_soc_estimate, initial_temperature, initial_temperature], dtype=float
    )

    sample_time = model.parameters.sample_time
    step_count = int(round(horizon_minutes * 60.0 / sample_time))

    time_minutes = []
    soc_real = []
    soc_estimated = []
    terminal_voltage = []
    current = []

    applied_current = 0.0
    charging_time = None
    maximum_voltage = -np.inf
    violating_steps = 0

    for step in range(step_count + 1):
        voltage_now = model.terminal_voltage(real_state, applied_current)
        time_minutes.append(step * sample_time / 60.0)
        soc_real.append(float(real_state[2]))
        soc_estimated.append(float(nominal_state[2]))
        terminal_voltage.append(float(voltage_now))
        current.append(float(applied_current))

        maximum_voltage = max(maximum_voltage, voltage_now)
        if voltage_now > MAXIMUM_VOLTAGE:
            violating_steps += 1
        if charging_time is None and real_state[2] >= TARGET_SOC:
            charging_time = step * sample_time / 60.0

        if step == step_count:
            break

        applied_current, _ = mpc.solve(nominal_state, applied_current)
        disturbance = SOC_PERTURBATION_AMPLITUDE * np.sin(random.standard_normal())
        real_state = model.step(real_state, applied_current, soc_disturbance=disturbance)

        thermal = model.step_thermal(nominal_state, applied_current)
        estimator.temperature = float(thermal[1])
        measured_voltage = model.terminal_voltage(real_state, applied_current)
        electrical = estimator.observe(measured_voltage, applied_current)
        nominal_state = np.array(
            [electrical[0], electrical[1], electrical[2], thermal[0], thermal[1]], dtype=float
        )

    return {
        "time_minutes": np.array(time_minutes),
        "soc_real": np.array(soc_real),
        "soc_estimated": np.array(soc_estimated),
        "terminal_voltage": np.array(terminal_voltage),
        "current": np.array(current),
        "charging_time": charging_time,
        "maximum_voltage": float(maximum_voltage),
        "violation_rate": 100.0 * violating_steps / len(terminal_voltage),
    }


def summarize_standalone_mpc(simulations=50, seed=0, **kwargs):
    """多次实现（不同未建模误差实现）的平均指标，对应 Table II。"""
    results = [simulate_standalone_mpc(seed=seed + index, **kwargs) for index in range(simulations)]
    charging_times = [r["charging_time"] for r in results if r["charging_time"] is not None]
    return {
        "results": results,
        "average_charging_time": float(np.mean(charging_times)) if charging_times else float("nan"),
        "maximum_voltage": float(max(r["maximum_voltage"] for r in results)),
        "average_violation_rate": float(np.mean([r["violation_rate"] for r in results])),
    }
