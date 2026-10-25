"""小规模凸二次规划：min 0.5 U'HU + g'U  s.t.  A U <= b。

MPC 的对偶问题  min_{lam>=0} 0.5 lam' M lam + q' lam （M = A H^-1 A'）是带箱约束的
凸二次规划，用 L-BFGS-B 求解并对上一时刻的对偶解热启动；比起逐步 Hildreth 的
Python 级坐标循环要快得多，且对输入/速率约束这种约束集非常稳定。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class DualQPSolver:
    def __init__(self, max_iter=80):
        self.max_iter = int(max_iter)
        self._warm = {}

    def solve(self, H, g, A, b, key=None):
        """返回最优控制序列 U（长度与 A 的列数一致）。"""
        chol = cho_factor(H, lower=False, check_finite=False)
        h_inv_g = cho_solve(chol, g, check_finite=False)

        if A.shape[0] == 0:
            return -h_inv_g

        h_inv_at = cho_solve(chol, A.T, check_finite=False)      # (n, m)
        mat = A @ h_inv_at                                       # 对偶 Hessian
        lin = A @ h_inv_g + b                                    # 对偶线性项

        lam0 = self._warm.get(key)
        if lam0 is None or lam0.size != A.shape[0]:
            lam0 = np.zeros(A.shape[0])

        def objective(lam):
            return 0.5 * float(lam @ (mat @ lam)) + float(lin @ lam), mat @ lam + lin

        res = minimize(objective, lam0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, None)] * A.shape[0],
                       options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-9})
        lam = np.maximum(res.x, 0.0)
        if key is not None:
            self._warm[key] = lam
        return -(h_inv_g + h_inv_at @ lam)

    def reset_warm_start(self):
        self._warm.clear()
