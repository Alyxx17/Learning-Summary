# =================== 车辆 VA-LMPC 求解器：问题(16)（论文 IV-B） ===================
# J_N^{i,*}(z_k^i) = min_{U_{0:N|k}} Σ_{t=0}^{N-1} l(z_{t|k}) + Q_v^{i-1}(z_{N|k})   （式 16a）
#   s.t. z_{t+1|k} = z_{t|k} + [cos φ; sin φ]·v·Δt （式 28b）
#        u_{t|k} ∈ U（式 28c）、z_{t|k} ∈ X（式 28d）、椭圆障碍（式 28f）、z_{0|k} = z_k^i
# 运行代价 l(z) = ||z - z_e||_Q^2（论文 IV-B：不含输入项）
# 终端代价 Q_v(z) = W^T φ(z - z_e)（式 24；基函数式(26)，取相对平衡点的坐标）
# 求解器：CasADi + IPOPT（论文用 YALMIP）

import numpy as np
import casadi as ca
import vehicle_env as env
import va_basis as basis


def build_va_solver(N=env.N_PRED):
    """构造问题(16) 的 IPOPT 求解器；参数 p = [z0(2); W(6)]"""
    U_sym = ca.SX.sym("U", env.NU * N)                  # 决策变量：[φ_0, v_0, φ_1, v_1, ...]
    z0_sym = ca.SX.sym("z0", env.NX)
    W_sym = ca.SX.sym("W", basis.P_DIM)

    z, cost = z0_sym, 0
    g_list, lbg_list, ubg_list = [], [], []
    for k in range(N):
        phi, v = U_sym[2 * k], U_sym[2 * k + 1]
        cost += (z - env.ZE).T @ env.Q @ (z - env.ZE)                  # 累加 l(z_{t|k})
        z = z + env.DT * ca.vertcat(ca.cos(phi) * v, ca.sin(phi) * v)  # 式(28b)

        for j in range(env.NX):                                        # 式(28d) 盒约束
            g_list.append(z[j]); lbg_list.append(env.X_MIN[j]); ubg_list.append(env.X_MAX[j])
        # 式(28f) 椭圆障碍：((x-x_obs)/a_x)^2 + ((y-y_obs)/a_y)^2 >= 1
        g_list.append(((z[0] - env.OBS_CENTER[0]) / env.OBS_R[0]) ** 2
                      + ((z[1] - env.OBS_CENTER[1]) / env.OBS_R[1]) ** 2)
        lbg_list.append(1.0); ubg_list.append(ca.inf)

    cost += W_sym.T @ basis.phi_expr(z - env.ZE)                       # 终端代价（式 16a）

    nlp = {"x": U_sym, "p": ca.vertcat(z0_sym, W_sym),
           "f": cost, "g": ca.vertcat(*g_list)}
    opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
    solver = ca.nlpsol("veh_va_lmpc", "ipopt", nlp, opts)

    lbx = np.tile([env.PHI_MIN, env.V_MIN], N)                         # 式(28c) 输入约束
    ubx = np.tile([env.PHI_MAX, env.V_MAX], N)
    return solver, lbx, ubx, ca.vertcat(*lbg_list), ca.vertcat(*ubg_list)


def solve_va(solver, lbx, ubx, lbg, ubg, z0, W, u_guess=None):
    """解一次问题(16)；返回 (u_opt(2N,), J_opt, ok, status)"""
    if u_guess is None:
        u_guess = np.zeros(len(lbx))
    p_val = np.concatenate([np.asarray(z0, dtype=float), np.asarray(W, dtype=float)])
    sol = solver(x0=u_guess, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats()["return_status"]
    f_val = float(sol["f"])
    u_val = np.array(sol["x"]).ravel()
    if not (np.isfinite(f_val) and np.all(np.isfinite(u_val))):
        return None, None, False, status
    return u_val, f_val, True, status


def warm_start_shift(u_prev):
    """热启动：把上一步的最优输入序列左移一位，末位补 0"""
    u = np.asarray(u_prev).ravel()
    return np.concatenate([u[env.NU:], np.zeros(env.NU)])


def rollout_va(solver, lbx, ubx, lbg, ubg, W, z_start=None, max_steps=60):
    """闭环滚动：每步解问题(16)、施加首元素（式 18）
    终止判据（论文 IV-B）：J_N^{t,*}(z_k) <= J_STOP = 1e-4
    返回 (Z_hist(T+1,2), U_hist(T,2), ok)；ok=True 表示按终止判据正常结束
    （求解失败或达到 max_steps 仍未满足判据时 ok=False）"""
    z = env.Z0.copy() if z_start is None else np.asarray(z_start, dtype=float).copy()
    Z_hist, U_hist = [z.copy()], []
    u_guess = np.zeros(env.NU * env.N_PRED)
    for _ in range(max_steps):
        u_opt, J_opt, ok, status = solve_va(solver, lbx, ubx, lbg, ubg, z, W, u_guess)
        if not ok:
            print(f"  [警告] 问题(16) 求解失败：{status}")
            return np.array(Z_hist), np.array(U_hist), False
        U_hist.append(u_opt[:env.NU].copy())
        u_guess = warm_start_shift(u_opt)
        z = env.dynamics(z, U_hist[-1])
        Z_hist.append(z.copy())
        if J_opt <= env.J_STOP:                      # 论文 IV-B 的每轮终止条件
            #J_opt本身就是状态距离目标的代价，若小于阈值则认为已经到达目标
            return np.array(Z_hist), np.array(U_hist), True
    return np.array(Z_hist), np.array(U_hist), False  # 未收敛（达到步数上限）
