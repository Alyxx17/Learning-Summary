"""式(7)-(15) 的模型预测控制器（LTV-MPC + 二次规划）。

* 预测模型：式(7)-(9) 的 3 自由度 USV。自由响应用**非线性模型真实前推**得到
  （式(7)(8) 的积分形式），输入响应用当前点的解析线性化得到——
  不能把线性化当成绝对坐标模型用，否则 J[0,2]·ψ 这类项会引入米级假误差。
* 代价函数：式(13)  J = sum ||e(i)||_Qw^2 + ||u(i-1)||_Rw^2，Q_w 由 AC 网络在线调整。
* 约束：式(14) 推力幅值、式(15) 推力变化率；不可行时按可行域裁剪，保证送出的控制量有界。
* 预测模型里没有环境干扰（控制器不知道干扰），这正是 Fig.6 中传统 MPC 在干扰下
  出现稳态偏差的原因。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from cs2_model import CyberShipII, NoDisturbance, wrap_angle
from dual_qp_solver import DualQPSolver


class LTVMPCTracker:
    def __init__(self, plant, ts=0.05, q_diag=(1.0, 1.0, 1.0), r_diag=(5e-6, 5e-6, 5e-6),
                 n_horizon=5, tau_max=(4.0, 4.0, 1.0), tau_rate=(3.0, 3.0, 1.0),
                 reg=1e-8, solver_max_iter=80, use_qp=False):
        self.plant = plant
        # 控制器内部的名义预测模型：与真实对象同参数，但不含环境干扰。
        # 预测用 RK4 单步（h = Ts）足够精确，比真实对象的 4 次子步快 3 倍。
        self.model = CyberShipII(plant.ts, NoDisturbance(), substeps=1)
        self.ts = float(ts)
        self.q_diag = np.asarray(q_diag, dtype=float)
        self.r_diag = np.asarray(r_diag, dtype=float)
        self.n_horizon = int(n_horizon)
        self.tau_max = np.asarray(tau_max, dtype=float)
        self.tau_rate = np.asarray(tau_rate, dtype=float)
        self.reg = float(reg)
        self.use_qp = bool(use_qp)
        self.input_matrix = self.model.input_matrix()
        self.solver = DualQPSolver(max_iter=solver_max_iter)
        self._constraints = {}
        self._r_bar_cache = {}
        self.last_sequence = None

    # -------------------------------------------------------------- 代价权重
    def _r_bar(self, n):
        if n not in self._r_bar_cache:
            self._r_bar_cache[n] = np.kron(np.eye(n), np.diag(self.r_diag))
        return self._r_bar_cache[n]

    # ---------------------------------------------------------------- 约束
    def _constraint_system(self, n):
        """返回 (A_c, b_base, rows_prev) —— rows_prev 为与 u_prev 相关的行索引。"""
        if n in self._constraints:
            return self._constraints[n]

        m = 12 * n
        ac = np.zeros((m, 3 * n))
        bb = np.zeros(m)
        eye = np.eye(3)
        for i in range(n):
            sl = slice(3 * i, 3 * i + 3)
            ac[3 * i:3 * i + 3, sl] = eye                      # u_i <= tau_max
            bb[3 * i:3 * i + 3] = self.tau_max
            ac[3 * n + 3 * i:3 * n + 3 * i + 3, sl] = -eye     # -u_i <= tau_max
            bb[3 * n + 3 * i:3 * n + 3 * i + 3] = self.tau_max

        base = 6 * n
        for i in range(n):
            sl = slice(3 * i, 3 * i + 3)
            row_up = base + 3 * i
            row_dn = base + 3 * n + 3 * i
            ac[row_up:row_up + 3, sl] = eye
            ac[row_dn:row_dn + 3, sl] = -eye
            if i > 0:
                prev = slice(3 * (i - 1), 3 * i)
                ac[row_up:row_up + 3, prev] = -eye
                ac[row_dn:row_dn + 3, prev] = eye
            bb[row_up:row_up + 3] = self.tau_rate * self.ts
            bb[row_dn:row_dn + 3] = self.tau_rate * self.ts

        rows_prev = (base, base + 3 * n)
        self._constraints[n] = (ac, bb, rows_prev)
        return self._constraints[n]

    def _constraint_bounds(self, n, u_prev):
        _, bb, (row_up, row_dn) = self._constraint_system(n)
        b = bb.copy()
        b[row_up:row_up + 3] += u_prev
        b[row_dn:row_dn + 3] -= u_prev
        return b

    # ------------------------------------------------------------ 可行性裁剪
    def _feasible_clip(self, seq, n, u_prev):
        """把任意控制序列逐点裁到 [max(-tau_max, u_{j-1}-rate*Ts), min(tau_max, u_{j-1}+rate*Ts)]。

        裁剪后的序列一定满足式(14)(15)，因此送进被控对象的控制量始终有界。
        """
        out = np.empty(3 * n)
        prev = np.asarray(u_prev, dtype=float)
        dr = self.tau_rate * self.ts
        for i in range(n):
            lo = np.maximum(-self.tau_max, prev - dr)
            hi = np.minimum(self.tau_max, prev + dr)
            u = np.clip(seq[3 * i:3 * i + 3], lo, hi)
            out[3 * i:3 * i + 3] = u
            prev = u
        return out

    @staticmethod
    def _quadratic_cost(hess, grad, seq):
        return 0.5 * float(seq @ (hess @ seq)) + float(grad @ seq)

    # ----------------------------------------------------------------- 求解
    def solve(self, state, t, ref, q_diag=None, n_horizon=None, u_prev=None):
        q = self.q_diag if q_diag is None else np.asarray(q_diag, dtype=float)
        n = self.n_horizon if n_horizon is None else int(n_horizon)
        u_prev = np.zeros(3) if u_prev is None else np.asarray(u_prev, dtype=float)
        x0 = np.asarray(state, dtype=float)

        # 名义模型在 x0 处线性化，再用指数矩阵做精确 ZOH 离散化：
        # Ts 与艏摇时间常数同量级时，二阶级数近似误差过大会导致高增益闭环失稳。
        jac = self.model.state_jacobian(x0)
        aug = np.zeros((9, 9))
        aug[:6, :6] = jac
        aug[:6, 6:] = self.input_matrix
        exp_aug = expm(aug * self.ts)
        ad = exp_aug[:6, :6]
        bd = exp_aug[:6, 6:]

        # 自由响应（式(7)(8)）：名义非线性模型在 u=0 下真实前推 n 步
        free_eta = np.zeros((n, 3))
        x_free = x0
        for i in range(n):
            x_free = self.model.step(x_free, np.zeros(3), t + i * self.ts)
            free_eta[i] = x_free[:3]

        # 输入响应矩阵：X = free + Su U
        powers = [np.eye(6)]
        for _ in range(n):
            powers.append(powers[-1] @ ad)
        blocks = np.zeros((n, n, 3, 3))
        for k in range(n):
            gk = powers[k][:3, :] @ bd
            idx = np.arange(n - k)
            blocks[idx + k, idx] = gk
        su = blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

        # 误差预测 e = free - eta_d + Su U，航向误差归一化
        ref_eta = ref.state_array(t + self.ts * np.arange(1, n + 1))
        err_free = (free_eta - ref_eta).reshape(-1)
        err_free[2::3] = wrap_angle(err_free[2::3])

        w = np.tile(q, n)
        sw = np.sqrt(w)
        su_w = su * sw[:, None]
        hess = su_w.T @ su_w + self._r_bar(n)
        hess.flat[::hess.shape[0] + 1] += self.reg
        grad = su_w.T @ (sw * err_free)

        # 无约束解在绝大多数步都可行（快路径）；不可行时先裁剪成可行解，
        # 再尝试用对偶 QP 求真正的最优解，只有在"收敛、可行、目标更小"时才采用。
        seq = -np.linalg.solve(hess, grad)
        bounds = self._constraint_bounds(n, u_prev)
        ac, _, _ = self._constraint_system(n)

        if np.all(ac @ seq <= bounds + 1e-9):
            out = seq
        else:
            out = self._feasible_clip(seq, n, u_prev)
            if self.use_qp:
                cand = self.solver.solve(hess, grad, ac, bounds, key=n)
                if (np.all(np.isfinite(cand)) and np.all(ac @ cand <= bounds + 1e-7)
                        and self._quadratic_cost(hess, grad, cand)
                        < self._quadratic_cost(hess, grad, out)):
                    out = cand
                else:
                    out = self._feasible_clip(out, n, u_prev)

        self.last_sequence = out
        return out[:3]
