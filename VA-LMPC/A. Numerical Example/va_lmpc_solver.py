# =================== VA-LMPC 求解器：问题(16) ===================
# J_N^{i,*}(x_k^i) = min_{U_{0:N|k}} Σ_{t=0}^{N-1} l(x_{t|k}, u_{t|k}) + Q_v^{i-1}(x_{N|k})   （式 16a）
#   s.t. x_{t+1|k} = f(x_{t|k}, u_{t|k})，t = 0..N-1                                        （式 16b）
#        U_{0:N|k} ∈ U，X_{0:N|k} ∈ X                                                       （式 16c,d）
#        x_{0|k} = x_k^i                                                                     （式 16e）
# 终端代价即价值近似器 Q_v^{i-1}(x_N) = W^T Φ(x_N)（式 24），W 由上一轮学习得到。
# 本例子系统线性、代价二次、终端代价为四次多项式 → 有限维非线性规划，用 IPOPT 求解。

import numpy as np
import casadi as ca
import double_integrator_env as env
import va_basis as basis


def build_va_solver(N=env.N_PRED):
    """构造问题(16) 的 IPOPT 求解器
    返回：(solver, lbx, ubx, lbg, ubg)；solver 的参数 p = [x0(2); W(6)]"""
    U_sym = ca.SX.sym("U", N)                       # 决策变量：控制序列 U_{0:N|k}
    x0_sym = ca.SX.sym("x0", env.NX)                # 当前状态 x_k^i
    W_sym = ca.SX.sym("W", basis.P_DIM)             # 价值权重 W

    x = x0_sym                                      # 预测状态
    cost = 0                                        # 目标累加器
    g_list, lbg_list, ubg_list = [], [], []         # 约束与上下界收集器

    for i in range(N):
        u = U_sym[i]
        cost += x.T @ env.Q @ x + env.R * u ** 2    # 累加运行代价 Σ l(x_{t|k}, u_{t|k})
        x = env.A @ x + env.B * u                   # 式(16b) 动力学推进

        for j in range(env.NX):                      # 式(16d) 状态约束：t = 1..N 全部在 X 内
            g_list.append(x[j]); lbg_list.append(env.X_MIN[j]); ubg_list.append(ca.inf)
            g_list.append(x[j]); lbg_list.append(-ca.inf); ubg_list.append(env.X_MAX[j])

    cost += W_sym.T @ basis.phi_expr(x)             # 终端代价 Q_v^{i-1}(x_{N|k})（式 16a）

    nlp = {"x": U_sym, "p": ca.vertcat(x0_sym, W_sym),
           "f": cost, "g": ca.vertcat(*g_list)}
    opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
    solver = ca.nlpsol("va_lmpc", "ipopt", nlp, opts)

    lbx = np.full(N, env.U_MIN)                      # 式(16c) 输入约束
    ubx = np.full(N, env.U_MAX)
    lbg = ca.vertcat(*lbg_list)
    ubg = ca.vertcat(*ubg_list)
    return solver, lbx, ubx, lbg, ubg


def solve_va(solver, lbx, ubx, lbg, ubg, x0, W, u_guess=None):
    """解一次问题(16)
    返回：(u_opt(N,), J_opt, ok, status)；失败时前两项为 None
    说明：终端代价含 X^4 项，问题非凸，IPOPT 可能只到局部解；只要返回有限解
          就接受（与论文一致：非凸时只能保证局部最优，见 Remark 3）"""
    if u_guess is None:
        u_guess = np.zeros(len(lbx))
    p_val = np.concatenate([np.asarray(x0, dtype=float), np.asarray(W, dtype=float)])
    sol = solver(x0=u_guess, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats()["return_status"]
    f_val = float(sol["f"])
    u_val = np.array(sol["x"]).ravel()
    if not (np.isfinite(f_val) and np.all(np.isfinite(u_val))):
        return None, None, False, status
    return u_val, f_val, True, status


def warm_start_shift(u_prev):
    """热启动：把上一步的最优输入序列左移一位，末位补 0"""
    return np.concatenate([np.asarray(u_prev).ravel()[1:], [0.0]])
    #展为1维数组，并切片，去掉第一个元素，末尾补0

def rollout_va(solver, lbx, ubx, lbg, ubg, W, x0=None, max_steps=100,
               stop_tol=env.STOP_TOL):
    """闭环滚动：每步解问题(16)，施加最优序列首元素 u_{0|k}^{*,i}（式 18）
    返回：(X_hist(T+1,2), U_hist(T,), ok)；T 为止时已满足 ‖x_T‖ <= stop_tol 或达到 max_steps，
          ok = False 表示中途求解失败"""
    x = env.X0.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
    X_hist, U_hist = [x.copy()], []
    u_guess = np.zeros(len(lbx))                 # 决策变量初始猜测
    for _ in range(max_steps):
        u_opt, _, ok, status = solve_va(solver, lbx, ubx, lbg, ubg, x, W, u_guess)
        if not ok:
            print(f"  [警告] 问题(16) 求解失败：{status}")
            return np.array(X_hist), np.array(U_hist), False
        U_hist.append(float(u_opt[0]))
        u_guess = warm_start_shift(u_opt)        # 热启动
        x = env.dynamics(x, u_opt[0])
        X_hist.append(x.copy())
        if np.linalg.norm(x) <= stop_tol:        # Algorithm 1 第 8 行
            break
    return np.array(X_hist), np.array(U_hist), True
