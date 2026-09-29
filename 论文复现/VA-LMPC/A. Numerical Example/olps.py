# =================== OLPS：开环最优解（直接法，论文 Table I 中 N = 100） ===================
# 无限时域最优问题(25) 的开环近似：
#       min_{U_{0:T-1}}  Σ_{t=0}^{T-1} l(x_t, u_t)   s.t. 式(25b)-(25e)
# 论文用它评估各控制器结果的"最优性"（Table II 中 OLPS 的 Objective = 1）。
# 直接法：把 T 步输入作为决策变量整体求解（本文例子轨迹约 10 步内收敛，T = 100 足够长）。
#solve_feasible_tail是用于求解某个状态的可行尾巴轨迹，用于构造初始轨迹
#olps.py 的起点就是x0，目的是提供一个性能上限（基准）
#时域足够大（T=100），求解开环最优问题，得到的闭环代价就是性能上限（基准），近似于LQR的最优值。
import numpy as np
import casadi as ca
import double_integrator_env as env

T_OLPS = 100                                    # 论文 Table I：OLPS 的 N = 100


def build_olps_solver(T=T_OLPS):
    """构造开环最优问题的 IPOPT 求解器；参数 p = x0(2,)"""
    U_sym = ca.SX.sym("U", T)
    x0_sym = ca.SX.sym("x0", env.NX)

    x = x0_sym
    cost = 0
    g_list, lbg_list, ubg_list = [], [], []
    for i in range(T):
        u = U_sym[i]
        cost += x.T @ env.Q @ x + env.R * u ** 2
        x = env.A @ x + env.B * u

        for j in range(env.NX):
            g_list.append(x[j]); lbg_list.append(env.X_MIN[j]); ubg_list.append(ca.inf)
            g_list.append(x[j]); lbg_list.append(-ca.inf); ubg_list.append(env.X_MAX[j])

    nlp = {"x": U_sym, "p": x0_sym, "f": cost, "g": ca.vertcat(*g_list)}
    opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
            "ipopt.max_iter": 3000}
    solver = ca.nlpsol("olps", "ipopt", nlp, opts)

    lbx = np.full(T, env.U_MIN)
    ubx = np.full(T, env.U_MAX)
    return solver, lbx, ubx, ca.vertcat(*lbg_list), ca.vertcat(*ubg_list)


def solve_olps(solver, lbx, ubx, lbg, ubg, x0=None, u_guess=None):
    """解开环最优问题，返回 (U_opt(T,), X_opt(T+1,2), J_opt, ok, status)"""
    if x0 is None:
        x0 = env.X0
    if u_guess is None:
        u_guess = np.zeros(len(lbx))
    sol = solver(x0=u_guess, p=np.asarray(x0, dtype=float),
                 lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats()["return_status"]
    U_opt = np.array(sol["x"]).ravel()
    if not (np.isfinite(float(sol["f"])) and np.all(np.isfinite(U_opt))):
        return None, None, None, False, status

    X_opt = np.zeros((len(U_opt) + 1, env.NX))                  # 由输入序列还原状态轨迹
    X_opt[0] = x0
    for k in range(len(U_opt)):
        X_opt[k + 1] = env.dynamics(X_opt[k], U_opt[k])
    return U_opt, X_opt, float(sol["f"]), True, status


if __name__ == "__main__":
    import time
    solver, lbx, ubx, lbg, ubg = build_olps_solver()
    t0 = time.perf_counter()
    U_opt, X_opt, J_opt, ok, status = solve_olps(solver, lbx, ubx, lbg, ubg)
    t_used = time.perf_counter() - t0
    if ok:
        J_ep = env.trajectory_cost(X_opt, U_opt)
        print(f"OLPS 求解成功：status = {status}，耗时 {t_used:.3f} s")
        print(f"  N = {T_OLPS} 步目标值 J = {J_opt:.4f}")
        print(f"  到 ||x||<=1e-2 的实际闭环代价 = {J_ep:.4f}")
        print("  前 10 步状态：")
        for k in range(10):
            print(f"    k={k:2d}  x=[{X_opt[k,0]:8.4f}, {X_opt[k,1]:7.4f}]  u={U_opt[k]:8.4f}")
    else:
        print(f"OLPS 求解失败：{status}")
