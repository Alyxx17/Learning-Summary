# -*- coding: utf-8 -*-
"""电池模型参数与控制器约束（电流以充电为正，与论文一致）。

参数来源说明
------------
1) 热参数（严格取自文献 [26] Lin et al., J. Power Sources 257 (2014) 1-11, Table 1）：
   导热热阻 R_c = 1.94 K/W、对流热阻 R_u = 3.19 K/W。

2) 热容（标定项）：Lin 表中 C_c = 62.7 J/K、C_s = 4.5 J/K 只对应电芯卷绕体。
   若直接使用该值，7 min 快充时稳态温升为 Q*R_u ≈ 10 K（T_s 将达 35 °C），
   与本文 Fig.4(b)（收敛后表面温度约 30.5 °C）及原文"温升不明显"的表述矛盾。
   因此按本文 Fig.4(b) 反标定为 C_c = 150 J/K、C_s = 10 J/K（保持 Lin 的 C_c/C_s ≈ 14 之比），
   使 7 min 快充末表面温度落在 30-31 °C。
   把 USE_LIN_HEAT_CAPACITY 置 True 可切回 Lin 原值（此时温度约束会成为主导约束）。

3) 电气参数（标定项）：Lin 原文只以图 8-13 给出 R_s/R_1/R_2/C_1/C_2，没有系数表。
   这里按图中形状（R_s 基本与 SOC 无关；R_1、R_2 在 SOC 两端略升；C_1 随 SOC 缓升）
   建模，并按本文"20%→80% 约 7 min、端电压顶到 3.6 V"标定总内阻约 26.5 mΩ@25 °C。

4) OCV(SOC)：按 Lin 图 7 的充放电平均曲线取点（LFP 平台型），用单调三次插值。
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# 基本工况（本文 Section IV-B）
# ---------------------------------------------------------------------------
NOMINAL_CAPACITY_AH = 2.3          # 电池额定容量 C_bat
AMBIENT_TEMPERATURE = 25.0         # 环境温度 T_a
MAXIMUM_CURRENT = 46.0             # 最大充电电流 I_max = 20C
MAXIMUM_VOLTAGE = 3.6              # 端电压硬约束 V_max
MAXIMUM_SURFACE_TEMPERATURE = 32.0 # 表面温度约束 T_s,max
TARGET_SOC = 0.8                   # 期望 SOC_d
INITIAL_SOC = 0.2                  # 单独 MPC 的初始 SOC
SAMPLE_TIME = 10.0                 # 决策周期 Δt

# 真实系统相对标称模型的 SOC 未建模误差 p = 0.001*sin(X), X~N(0,1)（式 31）
SOC_PERTURBATION_AMPLITUDE = 0.001

# ---------------------------------------------------------------------------
# 开路电压 OCV(SOC)
# ---------------------------------------------------------------------------
_OCV_SOC_KNOTS = np.array(
    [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
)
_OCV_VOLTAGE_KNOTS = np.array(
    [2.950, 3.160, 3.200, 3.235, 3.255, 3.270, 3.285, 3.300, 3.315, 3.335, 3.365, 3.395, 3.450]
)
_OCV_INTERPOLATOR = PchipInterpolator(_OCV_SOC_KNOTS, _OCV_VOLTAGE_KNOTS)


def open_circuit_voltage(soc):
    """开路电压 U_oc(SOC)，LFP 平台型曲线。"""
    return _OCV_INTERPOLATOR(np.clip(soc, 0.0, 1.0))


def open_circuit_voltage_slope(soc):
    """dU_oc/dSOC，用于 MPC 输出方程线性化。"""
    return _OCV_INTERPOLATOR.derivative()(np.clip(soc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 电热模型参数
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatteryParameters:
    """电热耦合模型参数集合。"""

    capacity_ah: float = NOMINAL_CAPACITY_AH
    ambient_temperature: float = AMBIENT_TEMPERATURE
    sample_time: float = SAMPLE_TIME

    # 热模型：Lin (2014) Table 1 的 R_c、R_u；热容为标定值
    conduction_resistance: float = 1.94      # R_c [K/W] 核心-表面
    convection_resistance: float = 3.19      # R_u [K/W] 表面对环境
    core_heat_capacity: float = 150.0         # C_c [J/K]（标定，Lin 原值 62.7）
    surface_heat_capacity: float = 10.0       # C_s [J/K]（标定，Lin 原值 4.5）

    # 电气模型：25 °C、充电方向（标定值，按 Lin 图 8-13 的形状）
    ohmic_resistance: float = 0.0120          # R_0 [Ω]
    polarization_resistance_1: float = 0.0065 # R_1 [Ω]
    polarization_resistance_2: float = 0.0080 # R_2 [Ω]
    polarization_capacitance_1: float = 2200.0 # C_1 [F]
    polarization_capacitance_2: float = 4500.0 # C_2 [F]

    # SOC 对极化电阻的影响（SOC 两端略升，形状取自 Lin 图 10/11）
    resistance_soc_curvature_1: float = 0.006
    resistance_soc_curvature_2: float = 0.006

    # 温度对电阻的影响（Lin 图 8：温度升高电阻下降）
    resistance_temperature_coefficient: float = -0.01
    resistance_reference_temperature: float = 25.0


DEFAULT_PARAMETERS = BatteryParameters()

# 需要严格复现 Lin 热容时切换到该参数集
LIN_HEAT_CAPACITY_PARAMETERS = replace(
    DEFAULT_PARAMETERS, core_heat_capacity=62.7, surface_heat_capacity=4.5
)

USE_LIN_HEAT_CAPACITY = False


def active_parameters():
    """返回当前生效的参数集（预留 Lin 原始热容开关）。"""
    return LIN_HEAT_CAPACITY_PARAMETERS if USE_LIN_HEAT_CAPACITY else DEFAULT_PARAMETERS


# ---------------------------------------------------------------------------
# 参数的 SOC / 温度依赖性
# ---------------------------------------------------------------------------
def _temperature_scale(parameters, temperature):
    return 1.0 + parameters.resistance_temperature_coefficient * (
        temperature - parameters.resistance_reference_temperature
    )


def ohmic_resistance(parameters, soc, temperature):
    """欧姆内阻 R_0(·)，Lin 图 8 显示其基本与 SOC 无关。"""
    return parameters.ohmic_resistance * _temperature_scale(parameters, temperature)


def polarization_resistance_1(parameters, soc, temperature):
    curvature = parameters.resistance_soc_curvature_1 * (np.asarray(soc) - 0.5) ** 2
    return (parameters.polarization_resistance_1 + curvature) * _temperature_scale(
        parameters, temperature
    )


def polarization_resistance_2(parameters, soc, temperature):
    curvature = parameters.resistance_soc_curvature_2 * (np.asarray(soc) - 0.5) ** 2
    return (parameters.polarization_resistance_2 + curvature) * _temperature_scale(
        parameters, temperature
    )


def polarization_capacitance_1(parameters, soc):
    """C_1 随 SOC 缓升（Lin 图 12 放电侧趋势）。"""
    return parameters.polarization_capacitance_1 * (0.85 + 0.30 * np.asarray(soc))


def polarization_capacitance_2(parameters, soc):
    """C_2 取常值近似（Lin 图 13 呈双峰，此处只保留量级）。"""
    return parameters.polarization_capacitance_2
