# -*- coding: utf-8 -*-
"""无迹卡尔曼滤波状态观测器（论文 Algorithm 1）。

观测量：端电压 U_t；估计量：x_e = [U_1, U_2, SOC]。
初始参数取自论文 Section IV-B：
    P_0^+ = diag(1e-7, 1e-7, 2.45)，Q_u = diag(1e-7, 1e-7, 1e-7)，R_u = 1.5e-7，
    SOC 初值故意取 0.5（真值 0.2），用于展示观测器的收敛能力。
"""

import numpy as np


class ElectricalStateEstimator:
    """基于 UKF 的电气状态观测器。"""

    def __init__(
        self,
        model,
        initial_state=(0.0, 0.0, 0.5),
        initial_covariance=None,
        process_covariance=None,
        measurement_covariance=1.5e-7,
        alpha=0.1,
        beta=2.0,
        kappa=0.0,
        real_voltage_feedback=True,
        temperature=None,
    ):
        self.model = model
        self.state = np.array(initial_state, dtype=float)
        self.covariance = (
            np.diag([1e-7, 1e-7, 2.45])
            if initial_covariance is None
            else np.array(initial_covariance, dtype=float)
        )
        self.process_covariance = (
            np.diag([1e-7, 1e-7, 1e-7])
            if process_covariance is None
            else np.array(process_covariance, dtype=float)
        )
        self.measurement_covariance = float(measurement_covariance)
        self.real_voltage_feedback = real_voltage_feedback
        # 观测器内部使用的是标称模型，其参数应按标称模型的核心温度取值，
        # 而不是环境温度；两者相差 7 K 时内阻偏差约 7%，会把 SOC 估计拽偏。
        self.temperature = (
            model.parameters.ambient_temperature if temperature is None else float(temperature)
        )

        dimension = self.state.size
        self._lambda = alpha**2 * (dimension + kappa) - dimension
        self._mean_weights = np.full(2 * dimension + 1, 0.5 / (dimension + self._lambda))
        self._covariance_weights = self._mean_weights.copy()
        self._mean_weights[0] = self._lambda / (dimension + self._lambda)
        self._covariance_weights[0] = self._mean_weights[0] + (1.0 - alpha**2 + beta)

    # -- 内部工具 ----------------------------------------------------------
    def _sigma_points(self, state, covariance):
        dimension = state.size
        scaled = (dimension + self._lambda) * covariance
        scaled = 0.5 * (scaled + scaled.T)
        try:
            root = np.linalg.cholesky(scaled + 1e-12 * np.eye(dimension))
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(scaled)
            root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
        points = np.empty((2 * dimension + 1, dimension))
        points[0] = state
        for i in range(dimension):
            points[1 + i] = state + root[:, i]
            points[1 + dimension + i] = state - root[:, i]
        return points

    def _propagate(self, electrical_state, current):
        return np.array(
            self.model.step_electrical(
                electrical_state[0],
                electrical_state[1],
                electrical_state[2],
                current,
                temperature=self.temperature,
            )
        )

    def _measure(self, electrical_state, current):
        return self.model.electrical_output(
            electrical_state[0],
            electrical_state[1],
            electrical_state[2],
            current,
            temperature=self.temperature,
        )

    # -- 滤波 --------------------------------------------------------------
    def predict(self, current):
        points = self._sigma_points(self.state, self.covariance)
        propagated = np.array([self._propagate(point, current) for point in points])
        mean = self._mean_weights @ propagated
        deviation = propagated - mean
        covariance = (deviation.T * self._covariance_weights) @ deviation
        self.state = mean
        self.covariance = covariance + self.process_covariance
        return self.state, self.covariance

    def update(self, measured_voltage, current):
        points = self._sigma_points(self.state, self.covariance)
        outputs = np.array([self._measure(point, current) for point in points])
        predicted_output = float(self._mean_weights @ outputs)

        output_deviation = outputs - predicted_output
        state_deviation = points - self.state
        output_covariance = float(
            self._covariance_weights @ (output_deviation**2)
        ) + self.measurement_covariance
        cross_covariance = (state_deviation.T * self._covariance_weights) @ output_deviation

        gain = cross_covariance / output_covariance
        self.state = self.state + gain * (measured_voltage - predicted_output)
        # SOC 是物理量，限制在 [0, 1]：否则初值 0.5、真值 0.2 时的大方差会让
        # 单次更新把 SOC 推到负值，OCV 查表被裁剪后滤波器将永久失锁。
        self.state[2] = np.clip(self.state[2], 0.0, 1.0)
        self.covariance = self.covariance - np.outer(gain, gain) * output_covariance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return self.state, self.covariance

    def observe(self, measured_voltage, current):
        """一个完整滤波周期：时间更新 + 量测更新。"""
        self.predict(current)
        if self.real_voltage_feedback:
            self.update(measured_voltage, current)
        return self.state
