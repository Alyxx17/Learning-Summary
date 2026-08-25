# =================== unicycle 车辆系统模型与环境 ===================
# 严格按论文式(36)：x_{k+1} = x_k + δ·g(x_k)·u_k（欧拉离散）
# 包含：动力学、运行代价、约束范围、状态空间均匀采样（论文范围）

import numpy as np

# =================== 系统参数（论文非线性例子） ===================
DT    = 0.2                             # 采样间隔 δ = 0.2 s
Q     = np.diag([1.0, 2.0, 0.06])       # 状态权重矩阵（3x3）
R     = np.diag([0.01, 0.005])          # 输入权重矩阵（2x2）
V_MAX, V_MIN = 1.0, -1.0                # 线速度约束 |v| ≤ 1 m/s
W_MAX, W_MIN = 4.0, -4.0                # 角速度约束 |ω| ≤ 4 rad/s
X_MAX, X_MIN = 2.0, 0.0                 # 状态约束 0 ≤ x ≤ 2（论文只约束 x 分量）
X0    = np.array([1.98, 5.0, -np.pi/3]) # 初始状态 [x, y, θ]^T
INPUT_DIM = 2
# 基函数输入缩放
STATE_SCALE = np.array([2.0, 6, np.pi])

# =================== 动力学（式 36，欧拉离散） ===================
def dynamics(x, u):
    """一步动力学：x:(3,) 状态, u:(2,) 输入 -> x_next:(3,)"""
    theta = x[2]                        # 当前偏航角
    g = np.array([[np.cos(theta), 0.0],
                  [np.sin(theta), 0.0],
                  [0.0, 1.0]])          # g(x_k)（3x2）
    x_next = x + DT * g @ u             # x_{k+1} = x_k + δ·g(x_k)·u_k
    # 角度归一化到 (-π, π]
    x_next[2] = (x_next[2] + np.pi) % (2 * np.pi) - np.pi
    return x_next

# =================== 运行代价 ℓ(x,u) = x^T Q x + u^T R u ===================
def stage_cost(x, u):
    """单步运行代价：x:(3,), u:(2,) -> 标量"""
    return float(x @ Q @ x + u @ R @ u)

# =================== 状态空间均匀采样（论文采样范围） ===================
def sample_states(n, rng):
    """均匀采样 n 个状态点：x∈[0,2], y∈[-1,6], θ∈[-π,π]
    参数 rng 为 np.random.Generator（保证可复现），返回 (3,n)"""
    return np.stack([
        rng.uniform(X_MIN, X_MAX, n),   # x 分量（n,）
        rng.uniform(-1.0, 6.0, n),      # y 分量（n,）
        rng.uniform(-np.pi, np.pi, n),  # θ 分量（n,）
    ], axis=0)                          # 堆叠成 (3,n)
