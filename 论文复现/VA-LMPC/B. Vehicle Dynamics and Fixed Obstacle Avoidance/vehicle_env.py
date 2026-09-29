# =================== 车辆避障环境（论文 IV-B，式(27)(28)） ===================
# 运动学（简单自行车模型，式(27)）：[ẋ; ẏ] = [cos φ; sin φ] · v
#   状态 z = [x, y]^T（2 维），输入 u = [φ, v]^T（偏航角、速度，2 维）
# 离散化（式(28b)）：z_{k+1} = z_k + [cos φ_k; sin φ_k] · v_k · Δt
# 运行代价（论文 IV-B）：l(z_k, u_k) = ||z_k - z_e||_Q^2，Q = P = diag(1,1)，**不含输入代价**
# 约束（式(28c)(28d)）：U = {u | [-π/3; -5] <= u <= [π/3; 5]}，
#                      X = {z | [0; 0] <= z <= [50; 20]}
# 固定障碍（式(28f)）：椭圆 ((x-x_obs)/a_x)^2 + ((y-y_obs)/a_y)^2 >= 1
# 起点 z_o = [0.1, 0.1]^T，平衡点 z_e = [40, 0.1]^T
#
# 初始可行轨迹（论文："用 [29] 的方法给一段可行输入序列"）：
#   用一个"较大的椭圆" [a_x, a_y] = [10, 5]（保守避障）+ 有限时域 OCP（N = 150，末端 z_N = z_e）生成；
#   而 VA-LMPC 实际运行用的障碍是较小的椭圆 [a_x, a_y] = [2, 1]（论文 IV-B 设置）。

import numpy as np
import casadi as ca

# =================== 系统与代价参数 ===================
DT = 0.3                            # 采样时间 Δt = 0.3 s（论文 IV-B）
Q = np.eye(2)                       # 状态权重 Q = diag(1,1)（代价 l = ||z - z_e||_Q^2）
PHI_MIN, PHI_MAX = -np.pi / 3, np.pi / 3    # 偏航角约束 |φ| <= π/3
V_MIN, V_MAX = -5.0, 5.0            # 速度约束 |v| <= 5
X_MIN = np.array([0.0, 0.0])        # 状态下界
X_MAX = np.array([50.0, 20.0])      # 状态上界
Z0 = np.array([0.1, 0.1])           # 起点 z_o
ZE = np.array([40.0, 0.1])          # 平衡点（目标）z_e
NX, NU = 2, 2                       # 状态维数、输入维数

# =================== 障碍（椭圆） ===================
OBS_CENTER = np.array([20.0, 0.1])  # 椭圆中心 [x_obs, y_obs]
OBS_R_FEAS = np.array([10.0, 5.0])  # 生成初始可行轨迹用的"大椭圆"半径
OBS_R = np.array([2.0, 1.0])        # VA-LMPC 运行时用的椭圆半径

# =================== 算法参数 ===================
N_PRED = 8                          # VA-LMPC 预测时域 N = 8（论文 IV-B）
N_FEAS = 150                        # 初始可行轨迹的 OCP 步数（= 采样状态数，论文：150）
J_STOP = 1e-4                       # 每轮终止判据 J_N^{t,*}(z_k) <= 1e-4（论文 IV-B）
N_SAMPLES_INIT = 150                # 初始化价值近似器用的采样状态数（论文：150）


# =================== 动力学与代价（式(27)(28b)） ===================
def dynamics(z, u):
    """一步动力学：z(2,) 状态、u(2,) = [φ, v] -> z_next(2,)（欧拉离散，步长 Δt）"""
    phi, v = float(u[0]), float(u[1])
    return z + DT * np.array([np.cos(phi) * v, np.sin(phi) * v])


def stage_cost(z):
    """运行代价 l(z,u) = ||z - z_e||_Q^2（不含输入项）"""
    d = z - ZE
    return float(d @ Q @ d)


def in_constraints(z, obstacle=True, r_obs=OBS_R, tol=1e-9):
    """状态是否满足盒约束（以及是否在椭圆障碍外）；tol 为数值容差
    （注意：闭环轨迹上椭圆约束可能被激活，求解器精度 ~1e-8，检查时给 1e-6 这类容差）"""
    if np.any(z < X_MIN - tol) or np.any(z > X_MAX + tol):
        return False
    if obstacle:
        e = (z - OBS_CENTER) / r_obs
        if e @ e < 1.0 - tol:
            return False
    return True


# =================== 初始可行轨迹 / OLPS：同一族 OCP ===================
def build_ocp_solver(N, r_obs, terminal=True):
    """有限时域 OCP（初始可行轨迹与 OLPS 共用）：
        min Σ_{k=0}^{N-1} ||z_k - z_e||_Q^2
        s.t. 动力学、盒约束、椭圆障碍约束、z_N = z_e（terminal=True 时）
    参数 p = z_0；返回 (solver, lbx, ubx, lbg, ubg, nx_u)"""
    U_sym = ca.SX.sym("U", NU * N)                 # 决策变量：[φ_0, v_0, φ_1, v_1, ...]
    z0_sym = ca.SX.sym("z0", NX)
    z, cost = z0_sym, 0
    g_list, lbg_list, ubg_list = [], [], []
    for k in range(N):
        phi, v = U_sym[2 * k], U_sym[2 * k + 1]
        cost += (z - ZE).T @ Q @ (z - ZE)
        z = z + DT * ca.vertcat(ca.cos(phi) * v, ca.sin(phi) * v)
        # 盒约束（式(28d)）
        for j in range(NX):
            g_list.append(z[j]); lbg_list.append(X_MIN[j]); ubg_list.append(X_MAX[j])
        # 椭圆障碍（式(28f)）：((x-x_obs)/a_x)^2 + ((y-y_obs)/a_y)^2 >= 1
        g_list.append(((z[0] - OBS_CENTER[0]) / r_obs[0]) ** 2
                      + ((z[1] - OBS_CENTER[1]) / r_obs[1]) ** 2)
        lbg_list.append(1.0); ubg_list.append(ca.inf)#1≤((x-x_obs)/a_x)^2 + ((y-y_obs)/a_y)^2 <∞，此时状态位于椭圆外
    if terminal:                                   # 末端必须到达平衡点
        for j in range(NX):
            g_list.append(z[j]); lbg_list.append(ZE[j]); ubg_list.append(ZE[j])

    nlp = {"x": U_sym, "p": z0_sym, "f": cost, "g": ca.vertcat(*g_list)}
    opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
            "ipopt.max_iter": 3000}
    solver = ca.nlpsol("veh_ocp", "ipopt", nlp, opts)
    lbx = np.tile([PHI_MIN, V_MIN], N)
    ubx = np.tile([PHI_MAX, V_MAX], N)
    return solver, lbx, ubx, ca.vertcat(*lbg_list), ca.vertcat(*ubg_list)


def solve_ocp(N, r_obs, z_start=None, terminal=True):
    """解开环 OCP，返回 (Z_hist (N+1,2), U_hist (N,2))
    注意：N 太小时末端约束不可行，这里显式检查求解器状态并报错"""
    if z_start is None:
        z_start = Z0
    solver, lbx, ubx, lbg, ubg = build_ocp_solver(N, r_obs, terminal)
    sol = solver(x0=np.zeros(NU * N), p=np.asarray(z_start, dtype=float),
                 lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    if not solver.stats()["success"]:
        raise RuntimeError(f"车辆 OCP 求解失败（N={N}）：{solver.stats()['return_status']}")
    U_hist = np.array(sol["x"]).reshape(-1, NU)
    Z_hist = np.zeros((N + 1, NX))
    Z_hist[0] = z_start
    for k in range(N):                                        # 由输入序列还原状态
        Z_hist[k + 1] = dynamics(Z_hist[k], U_hist[k])
    return Z_hist, U_hist


def generate_feasible_trajectory(N=N_FEAS, r_obs=OBS_R_FEAS):
    """初始可行轨迹：绕"大椭圆"避障、N 步内到达 z_e（论文：N = 150）"""
    return solve_ocp(N, r_obs, Z0, terminal=True)


def cost_to_go(Z_hist, U_hist=None):
    """沿轨迹的 cost-to-go 标签（式(12)）：J_t = Σ_{j>=t} l(z_j)，末状态计入 l(z_T)
    （运行代价不含输入项，故不需要 U_hist，保留参数是为了与 IV-A 的接口一致）"""
    n = len(Z_hist) - 1
    J = np.zeros(n + 1)
    J[n] = stage_cost(Z_hist[n])
    for k in range(n - 1, -1, -1):
        J[k] = stage_cost(Z_hist[k]) + J[k + 1]
    return J


if __name__ == "__main__":
    Z_f, U_f = generate_feasible_trajectory()
    J_f = cost_to_go(Z_f)
    ok = all(in_constraints(Z_f[k], True, OBS_R_FEAS) for k in range(len(Z_f)))
    print(f"[初始可行轨迹] {len(U_f)} 步（{len(U_f)*DT:.1f} s），约束满足 = {ok}")
    print(f"  终点 = {np.round(Z_f[-1], 4)}，cost-to-go = {J_f[0]:.4f}（论文迭代 0 = 15668.9409 量级）")
    print(f"  离障碍最近距离（椭圆归一化）: "
          f"{min(((Z_f[k]-OBS_CENTER)/OBS_R_FEAS)**2 @ np.ones(2) for k in range(len(Z_f))):.3f}（需 >= 1）")
    print(f"  y 最大值 = {Z_f[:,1].max():.3f}，速度范围 = [{U_f[:,1].min():.3f}, {U_f[:,1].max():.3f}]")
    print(f"  前 6 步: {np.round(Z_f[:6], 3).tolist()}")
