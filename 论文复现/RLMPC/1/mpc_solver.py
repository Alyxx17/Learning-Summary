# =================== MPC 求解器：OP2（RLMPC 策略生成器） ===================
# OP2: min Σ_{i=0}^{N-1} ℓ(x_i,u_i) + W^T Φ(x_N)
#      s.t. 动力学、|v|≤1、|ω|≤4、0≤x≤2（仅 x 分量，i=1..N，硬约束、无终端约束）

import numpy as np
import casadi as ca
import unicycle_env as env
import poly_basis as basis

def build_op2_solver(N, basis_indices):
    """构造 OP2 的 IPOPT 求解器
    参数：N 预测时域、basis_indices 基函数指数列表
    返回：(solver, lbx, ubx, lbg, ubg)；solver 的 p = [x0; W]"""
    p_dim = len(basis_indices)              # 基函数个数
    U     = ca.SX.sym("U", env.INPUT_DIM*N) # 决策变量：控制序列（2N,）
    x0_sym = ca.SX.sym("x0", 3)             # 初始状态参数（3,）
    W_sym  = ca.SX.sym("W", p_dim)          # 价值权重参数（p,）

    x = x0_sym                              # 当前预测状态
    cost = 0                                # 目标累加器（标量表达式）
    g_list, lbg_list, ubg_list = [], [], [] # 约束及边界收集器

    for i in range(N):
        u = U[2*i : 2*i+2]                  # 第 i 步控制 [v_i, ω_i]^T（2,）
        cost += x.T @ env.Q @ x + u.T @ env.R @ u   # 累加 ℓ(x_i, u_i)

        theta = ca.atan2(ca.sin(x[2]), ca.cos(x[2]))  # 角度归一化到 (-π,π]
        g_mat = ca.vertcat(
            ca.horzcat(ca.cos(theta), 0),
            ca.horzcat(ca.sin(theta), 0),
            ca.horzcat(0, 1),
        )                                   # g(x_k)（3x2）
        x_next = x + env.DT * g_mat @ u     # 欧拉离散（3,）
        # 每一步后 θ 折回 (-π,π]：atan2 平滑可导（IPOPT 友好），防 θ 无界累积
        x_next[2] = ca.atan2(ca.sin(x_next[2]), ca.cos(x_next[2]))

        # 状态约束 0 ≤ x_{i+1} ≤ 2
        g_list.append(x_next[0]); lbg_list.append(env.X_MIN); ubg_list.append(ca.inf)
        g_list.append(x_next[0]); lbg_list.append(-ca.inf); ubg_list.append(env.X_MAX)

        x = x_next                          # 状态前移

    # 终端代价 V(x_N) = W^T Φ(x_N)：梯度自动链式穿过全部 N 步动力学
    cost += W_sym.T @ basis.phi_expr(x, basis_indices, env.STATE_SCALE)

    g   = ca.vertcat(*g_list)               # 约束向量（2N,）
    lbg = ca.vertcat(*lbg_list)             # 约束下界
    ubg = ca.vertcat(*ubg_list)             # 约束上界

    nlp = {"x": U, "p": ca.vertcat(x0_sym, W_sym), "f": cost, "g": g}
    opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
    solver = ca.nlpsol("op2", "ipopt", nlp, opts)

    lbx = [env.V_MIN, env.W_MIN] * N        # 变量下界（每步重复）
    ubx = [env.V_MAX, env.W_MAX] * N        # 变量上界
    return solver, lbx, ubx, lbg, ubg

def solve_op2(solver, lbx, ubx, lbg, ubg, x0, W, u_guess):
    """在状态 x0、价值权重 W 下解一次 OP2
    返回：(u_opt(N,2), J_opt 标量, success 布尔, status 字符串)；失败时前两项为 None"""
    p_val = np.concatenate([x0, W])         # 参数 = [当前状态; 价值权重]
    sol = solver(x0=u_guess, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    status = solver.stats()["return_status"]#记录了算法停止运行时的具体原因或结果状态
    if not solver.stats()["success"]:
        return None, None, False, status
    u_opt = np.array(sol["x"]).reshape(-1, env.INPUT_DIM)   # (N,2)，N 由解向量长度自动推断
    J_opt = float(sol["f"])                 # 最优目标值 = 式(17) 的 J
    return u_opt, J_opt, True, status


# =================== 传统 MPC 求解器（论文 OP1 + [47] 终端设计） ===================
# 终端代价 F(x_N) = x_Nᵀ P_T x_N（论文单位权 P_T=I)
# 终端集 X_Ω = {F(x_N) ≤ ϱ_v}，ϱ_v = c·ϱ（按论文 η=ξ=8 计算）
#分为
TRAD_P_TERMINAL = np.diag([1,1,1])   # 终端代价加权（论文原值 diag(1,1,1)）
TRAD_ETA = 8.0                        # 终端控制器增益 η（论文值）
TRAD_XI  = 8.0                        # 终端控制器增益 ξ（论文值）
TRAD_RHO   = min(env.V_MAX**2 / TRAD_ETA**2, env.W_MAX**2 / TRAD_XI**2)   # ϱ
TRAD_C     = (1 + env.DT**2 * max(TRAD_ETA**2, TRAD_XI**2)
              - max(TRAD_ETA * env.DT**2 + env.Q[0, 0] / TRAD_ETA + env.R[0, 0] * TRAD_ETA,
                    2 * TRAD_XI * env.DT))                                 # c
TRAD_RHO_V = TRAD_C * TRAD_RHO                                   # ϱ_v = c·ϱ

def build_traditional_mpc_solver(N,MODE):
    """传统 MPC（OP1）：min Σ_{i=0}^{N-1} ℓ + F(x_N)
    s.t. 动力学、|v|≤1、|ω|≤4、0≤x≤2（i=1..N-1，式 5e）、终端约束 F(x_N)≤ϱ_v（式 5f）
    返回：(solver, lbx, ubx, lbg, ubg)；solver 的 p = [x0]"""
    U      = ca.SX.sym("U", 2*N)            # 决策变量（2N,）
    x0_sym = ca.SX.sym("x0", 3)             # 初始状态参数（3,）

    x = x0_sym
    cost = 0
    g_list, lbg_list, ubg_list = [], [], []

    for i in range(N):
        u = U[2*i : 2*i+2]
        cost += x.T @ env.Q @ x + u.T @ env.R @ u

        theta = ca.atan2(ca.sin(x[2]), ca.cos(x[2]))
        g_mat = ca.vertcat(
            ca.horzcat(ca.cos(theta), 0),
            ca.horzcat(ca.sin(theta), 0),
            ca.horzcat(0, 1),
        )
        x_next = x + env.DT * g_mat @ u     # 欧拉离散（3,）
        x_next[2] = ca.atan2(ca.sin(x_next[2]), ca.cos(x_next[2]))

        if i < N - 1:                       # 状态约束仅 i=1..N-1（式 5e）
            g_list.append(x_next[0]); lbg_list.append(env.X_MIN); ubg_list.append(ca.inf)
            g_list.append(x_next[0]); lbg_list.append(-ca.inf); ubg_list.append(env.X_MAX)

        x = x_next

    F = x.T @ TRAD_P_TERMINAL @ x            # 终端代价 F(x_N) = x_Nᵀ P_T x_N
    cost += F
    g_list.append(F)                        # 终端约束 F(x_N) ≤ ϱ_v（式 5f）
    lbg_list.append(-ca.inf); ubg_list.append(TRAD_RHO_V)

    g   = ca.vertcat(*g_list)
    lbg = ca.vertcat(*lbg_list)
    ubg = ca.vertcat(*ubg_list)
    #均为(9,)

    nlp = {"x": U, "p": x0_sym, "f": cost, "g": g}
    if MODE == 'branch1':
        opts = {
        "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False
                }
    elif MODE == 'branch2':
        opts = {
        "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
        "ipopt.max_iter": 10,                # 减少最大迭代次数
        "ipopt.max_cpu_time": 0.1,#0.05
        "ipopt.acceptable_tol": 1e0,
        "ipopt.acceptable_iter": 1,
        "ipopt.acceptable_constr_viol_tol": 10.0,
        "ipopt.acceptable_dual_inf_tol": 1e5,#1e5
        "ipopt.acceptable_compl_inf_tol": 1e4,#1e4
        "ipopt.acceptable_obj_change_tol": 1e5,#1e5
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.nlp_scaling_method": "none",
            }
    solver = ca.nlpsol("trad_mpc", "ipopt", nlp, opts)

    lbx = [env.V_MIN, env.W_MIN] * N        # 变量下界（每步重复）
    ubx = [env.V_MAX, env.W_MAX] * N        # 变量上界
    return solver, lbx, ubx, lbg, ubg

# def solve_traditional_mpc(solver, lbx, ubx, lbg, ubg, x0, u_guess):
#     """解一次传统 MPC（无 W 参数），返回 (u_opt(N,2), success)"""
#     sol = solver(x0=u_guess, p=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
#     if not solver.stats()["success"]:
#         return None, False
#     return np.array(sol["x"]).reshape(-1, 2), True

def solve_traditional_mpc(solver, lbx, ubx, lbg, ubg, x0, u_guess):
    """解一次传统 MPC（无 W 参数），返回 (u_opt(N,2), success)"""
    sol = solver(x0=u_guess, p=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    #status = stats["return_status"]
    # 只要解是有限的，就接受，不论 IPOPT 是否认为“成功”
    if np.isfinite(sol["f"]) and np.all(np.isfinite(sol["x"])):
        return np.array(sol["x"]).reshape(-1, 2), True
    else:
        return None, False
