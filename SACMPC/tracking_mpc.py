# =================== 跟踪 MPC（CasADi + IPOPT） ===================
# 每次求解: min Σ_{i=0}^{N-1} [ Σ_k q_k·(ê_i,k)² + Σ_j r_j·(Δû_i,j)² ]
#           s.t. x_{i+1} = x_i + δ·g(x_i)·u_i,  |v|≤1, |ω|≤4
# 其中 e_i = [x_i−x_i^ref, y_i−y_i^ref, Δθ_i]（Δθ 用 atan2 防跨 ±π），
#      ê = e/ERR_SCALE、Δû = (u−u^ref)/CTRL_SCALE 为归一化信号（见 unicycle_env）
#
# 关键设计（"学 5 个对角权重"版）:
#   · 符号问题只构建一次；参考窗口 (N×3 + N×2)、初始状态、5 个权重全部走参数 p
#     → SAC 每步改动的只是这 5 个权重参数，MPC 结构不变（"在线调参"的接口基础）
#   · 可学权重 = Q、R 对角线本身 (q1,q2,q3,r1,r2)，直接作用在归一化信号上
#   · 无终端代价、无状态硬约束（保持最小；后续均可再加开关）

import time
import numpy as np
import casadi as ca
import unicycle_env as env


class TrackingMPC:
    """跟踪 MPC 求解器封装（build 一次，反复 solve）"""

    def __init__(self, N=5):
        self.N = N
        # ---- 符号变量 ----
        U = ca.SX.sym("U", 2 * N)            # 决策变量: 未来 N 步控制序列 (2N,)
        #控制量是2*N是因为每一步有两个控制输入（线速度v和角速度ω），所以对于N步预测，总共有2*N个决策变量。
        x0_sym = ca.SX.sym("x0", 3)          # 初始状态参数 (3,)
        Rst = ca.SX.sym("Rst", 3 * N)        # 参考状态窗口 [x_ref,y_ref,θ_ref] × N
        Rin = ca.SX.sym("Rin", 2 * N)        # 参考输入窗口 [v_ref,ω_ref] × N
        qd_sym = ca.SX.sym("qd", 3)          # 误差权重 (q1,q2,q3)，作用在归一化误差上
        rd_sym = ca.SX.sym("rd", 2)          # 控制权重 (r1,r2)，作用在归一化控制偏差上
        err_inv  = ca.DM(1.0 / env.ERR_SCALE)    # 归一化尺度倒数（用乘法代替除法）
        ctrl_inv = ca.DM(1.0 / env.CTRL_SCALE)

        # ---- 累加代价 + 动力学展开（single shooting，与 env.dynamics 同一公式） ----
        x = x0_sym
        cost = 0
        for i in range(N):
            u = U[2 * i: 2 * i + 2]          # 第 i 步控制
            xr = Rst[3 * i: 3 * i + 3]       # 第 i 步参考状态
            ur = Rin[2 * i: 2 * i + 2]       # 第 i 步参考输入

            # 状态误差（θ 差用 atan2 归一化，IPOPT 友好）
            dth = ca.atan2(ca.sin(x[2] - xr[2]), ca.cos(x[2] - xr[2]))
            e = ca.vertcat(x[0] - xr[0], x[1] - xr[1], dth)
            du = u - ur
            # ℓ_i = Σ q_k·(ê_k)² + Σ r_j·(Δû_j)²（归一化信号上的加权平方和）
            e_n  = e * err_inv               # 归一化误差 (3,)
            du_n = du * ctrl_inv             # 归一化控制偏差 (2,)
            cost += ca.dot(qd_sym, e_n ** 2) + ca.dot(rd_sym, du_n ** 2)

            # 动力学 x_{i+1} = x_i + δ·g(x_i)·u_i（P02 式(36)，欧拉离散）
            theta = ca.atan2(ca.sin(x[2]), ca.cos(x[2]))
            g_mat = ca.vertcat(
                ca.horzcat(ca.cos(theta), 0),
                ca.horzcat(ca.sin(theta), 0),
                ca.horzcat(0, 1),
            )
            x_next = x + env.DT * g_mat @ u
            x_next[2] = ca.atan2(ca.sin(x_next[2]), ca.cos(x_next[2]))
            x = x_next

        # ---- 编译求解器（无一般约束 g，只有变量框约束 lbx/ubx） ----
        nlp = {"x": U, "p": ca.vertcat(x0_sym, Rst, Rin, qd_sym, rd_sym), "f": cost}
        opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
        self.solver = ca.nlpsol("tracking_mpc", "ipopt", nlp, opts)

        self.lbx = [env.V_MIN, env.W_MIN] * N    # 变量下界（每步 [v_min, ω_min]）
        self.ubx = [env.V_MAX, env.W_MAX] * N    # 变量上界

    def solve(self, x0, X_ref, U_ref, q_diag, r_diag, u_guess=None):
        """解一次跟踪 MPC
        参数:
          x0     —— 当前状态 (3,)
          X_ref  —— 参考状态窗口 (N,3)
          U_ref  —— 参考输入窗口 (N,2)
          q_diag —— 误差权重 (3,) = [q1,q2,q3]（>0，由动作映射保证）
          r_diag —— 控制权重 (2,) = [r1,r2]（>0，由动作映射保证）
          u_guess —— 决策变量初始猜测（热启动；None 时取零）
        返回: (u_opt (N,2), J, ok, status, solve_time[s])
        """
        p_val = np.concatenate([
            np.asarray(x0, float).ravel(),
            np.asarray(X_ref, float).ravel(),   # 逐时刻展开，与 Rst[3i:3i+3] 对齐
            np.asarray(U_ref, float).ravel(),   # 与 Rin[2i:2i+2] 对齐
            np.asarray(q_diag, float).ravel(),  # (3,) 与 qd_sym 对齐
            np.asarray(r_diag, float).ravel(),  # (2,) 与 rd_sym 对齐
        ])
        #.ravel() 是 NumPy 中的一个方法，用于将多维数组展平为一维数组。
        if u_guess is None:
            u_guess = np.zeros(2 * self.N)
        u_guess = np.clip(np.asarray(u_guess, float).ravel(), self.lbx, self.ubx)
        #np.asarray(u_guess, float).ravel() 将 u_guess 转换为浮点型的一维数组，然后使用 np.clip 将其限制在 self.lbx 和 self.ubx 之间，确保初始猜测的控制输入在允许的范围内。
        t_start = time.perf_counter()
        #perf_counter() 返回一个高精度的计时器，用于测量时间间隔。
        sol = self.solver(x0=u_guess, p=p_val, lbx=self.lbx, ubx=self.ubx)
        solve_time = time.perf_counter() - t_start
        #solve_time 是求解时间，即从开始求解到得到结果所用的时间。

        status = self.solver.stats()["return_status"]
        u_flat = np.array(sol["x"]).ravel()
        J = float(sol["f"])
        # 只要解有限就接受（与 P02 复现里 solve_traditional_mpc 的宽松判据一致）
        ok = bool(np.all(np.isfinite(u_flat)) and np.isfinite(J))
        return u_flat.reshape(-1, 2), J, ok, status, solve_time
        #reshape(-1, 2) 将一维数组 u_flat 重塑为二维数组，每行表示一个时间步的控制输入 [v, ω]。
