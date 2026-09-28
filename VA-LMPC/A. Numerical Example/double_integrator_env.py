# =================== 双积分器数值例子环境（论文 IV-A，式(25)） ===================
# 离散系统：x_{k+1} = [[1,1],[0,1]] x_k + [0,-1]^T u_k                （式 25b，k>=0）
# 运行代价：l(x,u) = ||x||_Q^2 + ||u||_R^2，Q = 100 I_2，R = 0.5
# 约束：U = {u | -2 <= u <= 2}，X = {x | [-15,-1]^T <= x <= [0,6]^T}   （式 25）
# 初值 x_o = [-14, 2]^T，平衡点 x_e = [0,0]^T
#
# 说明：原点 x_e 落在状态约束边界 X = 0 上，“能否在越过 X = 0 之前把速度降到 0”
#       由 OCP（solve_feasible_tail）自动处理，无需手写可行性过滤器。

import numpy as np
import casadi as ca

# =================== 系统与约束参数（式 25） ===================
A = np.array([[1.0, 1.0],
              [0.0, 1.0]])          # 状态矩阵
B = np.array([[0.0],
              [-1.0]])              # 输入矩阵
Q = 100.0 * np.eye(2)               # 状态权重 Q = 100 I_2
R = 0.5                             # 输入权重 R = 0.5（标量）
U_MIN, U_MAX = -2.0, 2.0            # 输入约束
X_MIN = np.array([-15.0, -1.0])     # 状态下界
X_MAX = np.array([ 0.0,   6.0])     # 状态上界
X0 = np.array([-14.0, 2.0])         # 初始状态 x_o
XE = np.array([0.0, 0.0])           # 平衡点 x_e
NX = 2                              # 状态维数（输入为标量 u）

N_PRED   = 4                        # 预测时域 N = 4（论文数值例子）
STOP_TOL = 1e-2                     # 本轮终止判据 ||x_k^i|| <= 10^-2（Algorithm 1 第 8 行）
TRAJ_STEPS = 100                    # 初始可行轨迹长度 = 采样点数（论文：100 sampled states）
FEAS_COAST_TO = -4.0                # 初始可行轨迹：滑行段（开环 u = 0）结束的位置
                                    # —— 对应论文 Fig.2 中 First Feasible Solution 的水平段（y 恒为 2）
N_OCP_TAIL = 50                     # 尾巴 OCP 的步数（末端约束 x_N = x_e；实际 ~35 步即收敛）


# =================== 动力学与运行代价（式 25） ===================
def dynamics(x, u):
    """一步动力学 x_{k+1} = A x_k + B u_k：x(2,) 状态、u 标量 -> x_next(2,)"""
    return A @ x + B[:, 0] * u
#B[:, 0]代表B矩阵的第一列


def stage_cost(x, u):
    """运行代价 l(x,u) = x^T Q x + R u^2（式(25a) 的求和项）"""
    return float(x @ Q @ x + R * u * u)


def in_constraints(x):
    """判断状态是否满足 X 约束（用于检查轨迹可行性）"""
    return bool(np.all(x >= X_MIN - 1e-9) and np.all(x <= X_MAX + 1e-9))


# =================== 初始可行轨迹：开环滑行段 + OCP 求得的尾巴 ===================
# 论文原文："To initialize a value approximator, we first give a feasible k-step input
# control sequence, which is computed by using the similar method in [29]. Then, we use the
# feedback controller to compute a feasible solution that drives the system to the origin
# from the state at the kth step."
# → 本复现：阶段 1 取开环输入序列 u = 0 滑行（对应论文 Fig.2 中 First Feasible Solution
#    的水平段）；阶段 2 不手写控制器，而像官方 solve_feasible_obs.m 那样用有限时域 OCP
#    求出把系统驱动到原点的可行尾巴（系统换掉时也无需重新手写可行性逻辑）。

def solve_feasible_tail(x_start, N=N_OCP_TAIL):
    """有限时域 OCP：min Σ_{k=0}^{N-1} l(x_k, u_k)  s.t. 动力学、x∈X、u∈U、x_N = x_e
    返回：(X_tail (N+1,2)，U_tail (N,))"""
    U_sym = ca.SX.sym("U", N)
    x0_sym = ca.SX.sym("x0", NX)
    x, cost = x0_sym, 0
    g_list, lbg_list, ubg_list = [], [], []
    for k in range(N):
        u = U_sym[k]
        cost += x.T @ Q @ x + R * u ** 2
        x = A @ x + B[:, 0] * u
        for j in range(NX):                                     # 状态约束 x_k ∈ X
            g_list.append(x[j]); lbg_list.append(X_MIN[j]); ubg_list.append(ca.inf)
            g_list.append(x[j]); lbg_list.append(-ca.inf); ubg_list.append(X_MAX[j])
    for j in range(NX):                                         # 终端约束 x_N = x_e
        g_list.append(x[j]); lbg_list.append(XE[j]); ubg_list.append(XE[j])

    nlp = {"x": U_sym, "p": x0_sym, "f": cost, "g": ca.vertcat(*g_list)}
    opts = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False}
    solver = ca.nlpsol("feasible_tail", "ipopt", nlp, opts)
    sol = solver(x0=np.zeros(N), p=np.asarray(x_start, dtype=float),
                 lbx=np.full(N, U_MIN), ubx=np.full(N, U_MAX),
                 lbg=ca.vertcat(*lbg_list), ubg=ca.vertcat(*ubg_list))
    U_tail = np.array(sol["x"]).ravel()
    X_tail = np.zeros((N + 1, NX))                              # 由输入序列还原状态
    X_tail[0] = x_start
    for k in range(N):
        X_tail[k + 1] = dynamics(X_tail[k], U_tail[k])
    return X_tail, U_tail


def generate_feasible_trajectory(n_steps=TRAJ_STEPS, coast_to=FEAS_COAST_TO, N_tail=N_OCP_TAIL):
    """生成初始可行轨迹 X^0 = 阶段 1（开环滑行）+ 阶段 2（OCP 尾巴，末尾停在原点）
    返回：(X_hist (n_steps+1,2)，U_hist (n_steps,))"""
    X_hist = np.zeros((n_steps + 1, NX))
    U_hist = np.zeros(n_steps)
    X_hist[0] = X0

    k = 0
    while k < n_steps and X_hist[k, 0] < coast_to:               # 阶段 1：滑行（开环 u = 0）
        U_hist[k] = 0.0
        X_hist[k + 1] = dynamics(X_hist[k], 0.0)
        k += 1

    X_tail, U_tail = solve_feasible_tail(X_hist[k], N_tail)      # 阶段 2：OCP 尾巴
    n_tail = min(len(U_tail), n_steps - k)# 取 OCP 尾巴的长度和剩余步数的最小值
    X_hist[k:k + n_tail + 1] = X_tail[:n_tail + 1]
    U_hist[k:k + n_tail] = U_tail[:n_tail]
    for j in range(k + n_tail, n_steps):                         # 尾巴之后：已停在原点
        X_hist[j + 1] = X_hist[j]# 保持在原点
    return X_hist, U_hist


def cost_to_go(X_hist, U_hist):
    #从当前状态出发，沿着当前这条特定轨迹走到终点（或无穷远）的剩余累计代价
    """沿轨迹的 cost-to-go 标签（式(12) 的数据标签，式(8) 的定义）
    J_t = x_t'Qx_t + u_t'Ru_t + J_{t+1}，且末状态取 J_T = x_T'Qx_T
    —— 与官方 ComputeCost.m 一致（官方把末状态的状态代价也算进去）"""
    n = len(U_hist)
    J = np.zeros(n + 1)
    J[n] = float(X_hist[n] @ Q @ X_hist[n])# 末状态的 cost-to-go = x_T'Qx_T
    for k in range(n - 1, -1, -1):# 表示从 n 开始倒序循环到 0（包含 0）的整数序列
        J[k] = stage_cost(X_hist[k], U_hist[k]) + J[k + 1]
    return J
#计算每一个状态的cost-to-go，用于训练价值函数近似器，作为数据集里的标签
#并不是按照论文算了无穷时域的代价，而是计算控制量长度的代价，
#因为达到终点后，状态代价就为0了，所以cost-to-go的末状态取的是x_T'Qx_T，而不是0。

def trajectory_cost(X_hist, U_hist, stop_tol=STOP_TOL):
    """本轮闭环实际代价（论文 Table II 的 Objective）：从 x_0 累加到 ‖x_k‖ <= stop_tol 为止，
    同样含末状态的状态代价（与官方 ComputeCost.m 一致）"""
    total = 0.0
    for k in range(len(U_hist)):
        total += stage_cost(X_hist[k], U_hist[k])
        if np.linalg.norm(X_hist[k + 1]) <= stop_tol:
            total += float(X_hist[k + 1] @ Q @ X_hist[k + 1])
            break
    return total
#整条轨迹的总代价，约等于cost_to_go在初始状态的值。
#区别在于cost_to_go算的是每个状态下，剩余的代价，而trajectory_cost算的是从初始状态开始，直到终止条件满足的总代价。
#依然不是无限时域得代价，而是计算控制量长度的代价，然后加一个末状态的状态代价。
if __name__ == "__main__":
    X_f, U_f = generate_feasible_trajectory()
    J_f = cost_to_go(X_f, U_f)
    ok = all(in_constraints(X_f[k]) for k in range(len(X_f)))
    print(f"初始可行轨迹：{len(U_f)} 步，约束满足 = {ok}，"
          f"cost-to-go = {J_f[0]:.4f}")
    print("前 12 步状态：")
    for k in range(min(12, len(U_f))):
        print(f"  k={k:2d}  x=[{X_f[k,0]:8.4f}, {X_f[k,1]:7.4f}]  u={U_f[k]:7.4f}"
              f"  cost-to-go={J_f[k]:10.3f}")
