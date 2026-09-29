# =================== 价值近似器的基函数 Φ(x)（论文式(24)、(26)） ===================
# 线性权重网络：Q_v(x) = W^T Φ(x)                                    （式 24）
# 基函数（论文式(26)）：φ(x) = [1, X^2, X Y, Y^2, X^4, Y^4]^T，n_v = 6
# 权重 W 的求法："obtained by using quadratic programming, e.g., by minimizing
# squared residual"（论文）→ 最小二乘
# 注：本文件为 IV-B 车辆例子的副本——除普通最小二乘 fit_weights 外，
#     额外提供带结构约束的 fit_weights_convex（见下文说明）；IV-A 用不到的 value_grid 已移除。

import numpy as np
import casadi as ca

P_DIM = 6                                # n_v：价值网络参数（神经元）个数 = 存储代价


# =================== 基函数数值版 / 符号版 ===================
def phi_matrix(X_pts):#x_pts是一个二维数组，表示状态点的集合
    """批量数值版：X_pts(n,2) -> Φ(n,6)（最小二乘设计矩阵）"""
    X, Y = X_pts[:, 0], X_pts[:, 1]
    return np.column_stack([np.ones_like(X), X ** 2, X * Y, Y ** 2, X ** 4, Y ** 4])


def phi_expr(x_sym):
    """符号版 Φ(x)：x_sym(2,) -> (6,1) 的 SX 表达式（嵌入 MPC 终端代价，梯度自动穿过动力学）"""
    X, Y = x_sym[0], x_sym[1]
    return ca.vertcat(1.0, X ** 2, X * Y, Y ** 2, X ** 4, Y ** 4)


# =================== 权重拟合：min ||Φ W - J||^2 ( + ridge ||W||^2 ) ===================
def fit_weights(X_pts, J, ridge=0.0):
    """最小二乘（可选岭正则）拟合价值权重 W
    X_pts(n,2)：状态样本；J(n,)：对应的 cost-to-go 标签
    用 SVD 最小二乘而非正规方程，数值上更稳（基含 X^4 项，列量级差异大）"""
    Phi = phi_matrix(X_pts)
    J = np.asarray(J, dtype=float).ravel()
    if ridge > 0.0:
        A_aug = np.vstack([Phi, np.sqrt(ridge) * np.eye(P_DIM)])
        b_aug = np.concatenate([J, np.zeros(P_DIM)])
        W, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        W, *_ = np.linalg.lstsq(Phi, J, rcond=None)
    return W


# =================== 带结构约束的权重拟合（车辆例子 IV-B 用） ===================
def fit_weights_convex(X_pts, J, n_restart=4, seed=0):
    """**带结构约束**的最小二乘拟合：
        Q_v(s) = W0 + W1 sx^2 + W2 sx·sy + W3 sy^2 + W4 sx^4 + W5 sy^4,  s = z - z_e
    约束（保证论文 Fig.3 所示的“非负、凸”性质）：
      (a) W0 = 0                          —— 平衡点处 cost-to-go 为 0；
      (b) 二次部分 P = [[W1, W2/2], [W2/2, W3]] ⪰ 0；
      (c) W4 >= 0, W5 >= 0；
      (d) W2 <= 0。
    于是 Q_v(s) = s^T P s + W4 sx^4 + W5 sy^4 >= 0 且全局凸。
    动机：车辆例子的训练数据集中在一条轨迹上（可行轨迹在障碍区一直贴在 y ≈ 5，
    “窄走廊” y ∈ [0.1, 3] 内没有数据），普通最小二乘（或只加 (b)(c) 的约束）会给出
    W2 > 0 的交叉项，使 Q_v 在 s_x<0、s_y 增大时反而下降 —— MPC 于是选择“绕宽”
    （实测 y_max = 2.85，最优为 1.08，代价多 +338%；甚至出现负值区，
    使 J* <= 1e-4 终止判据在离目标 ~10 m 处被误触发）。
    在操作区 s_x<0、s_y 有界内，W2 <= 0 保证 Q_v 关于 |s_y| 单调不减。
    参数化：P = L L^T（L 下三角，取 l11 >= 0、l21 <= 0 → W2 = 2 l11 l21 <= 0）：
            W1 = l11^2，W2 = 2 l11 l21，W3 = l21^2 + l22^2；W4 = a^2，W5 = b^2；
    再对参数 (l11, c, l22, a, b)（其中 l21 = -c^2）用 Levenberg–Marquardt 解无约束最小二乘。
    """
    phi = phi_matrix(X_pts)
    J = np.asarray(J, dtype=float).ravel()

    def unpack(theta):
        l11, c, l22, a, b = theta
        l21 = -c ** 2
        return np.array([0.0, l11 ** 2, 2.0 * l11 * l21, l21 ** 2 + l22 ** 2, a ** 2, b ** 2])

    def residual(theta):
        return phi @ unpack(theta) - J

    #设置四个初值是因为需要在不同的初始点进行优化，以找到全局最优解。
    #LM算法是一种局部优化方法，容易陷入局部最优解。通过提供多个初值，可以增加找到全局最优解的机会。
    #选取四个初值中最小的残差作为最终的拟合结果，确保拟合的权重 W 满足约束条件并且具有较好的性能。
    # ---- 初值 ①：普通最小二乘解投影到可行域（特征值截断 + Cholesky + 交叉项取负） ----
    W_ls = fit_weights(X_pts, J)
    P = np.array([[W_ls[1], 0.5 * W_ls[2]], [0.5 * W_ls[2], W_ls[3]]])
    w, V = np.linalg.eigh(P)
    P_psd = V @ np.diag(np.maximum(w, 1e-8)) @ V.T + 1e-12 * np.eye(2)
    L = np.linalg.cholesky(P_psd)
    l11a, l21a, l22a = abs(L[0, 0]), -abs(L[1, 0]), abs(L[1, 1])
    a_a, b_a = np.sqrt(max(W_ls[4], 1e-8)), np.sqrt(max(W_ls[5], 1e-8))
    s_ls = (l11a, np.sqrt(-l21a), l22a, a_a, b_a)
    # ---- 初值 ②：对角（不含交叉项） ----
    s_diag = (np.sqrt(max(W_ls[1], 1e-8)), 0.0, np.sqrt(max(W_ls[3], 1e-8)), a_a, b_a)
    # ---- 初值 ③：给定若干四次项量级（避免 LM 把四次项压到 0 的退化解） ----
    starts = [s_ls, s_diag]
    for a0, b0 in [(0.06, 0.0), (0.06, 0.03), (0.03, 0.0), (0.10, 0.05)]:
        starts.append((s_ls[0], s_ls[1], s_ls[2], a0, b0))
    rng = np.random.default_rng(seed)
    for _ in range(n_restart):
        base = np.array(starts[0], dtype=float)
        starts.append(tuple(np.abs(base * (1.0 + 0.3 * rng.standard_normal(5))) + 1e-9))

    from scipy.optimize import least_squares               # 局部导入：仅车辆例子用到
    best_theta, best_cost = np.array(starts[0]), np.inf
    for th0 in starts:
        try:
            res = least_squares(residual, np.maximum(np.array(th0, dtype=float), 1e-12),
                                method="lm", max_nfev=40000)
        except Exception:
            continue
        if res.cost < best_cost:
            best_cost, best_theta = res.cost, res.x
    return unpack(best_theta)


# =================== 凸性检查（论文 Fig.3："在紧集上为凸函数"） ===================
def psd_check(W):
    """全局半正定检查（仅对论文式(26)/(48) 的 6 项基）：二次部分矩阵半正定 + X^4、Y^4 系数非负"""
    M = np.array([[W[1], 0.5 * W[2]], [0.5 * W[2], W[3]]])
    return bool(np.linalg.eigvalsh(M).min() > 0 and W[4] >= 0 and W[5] >= 0), np.linalg.eigvalsh(M)


def hessian_expr(W, X, Y):
    """解析 Hessian（Q_v 为四次多项式，可直接写出）：
    [2W1 + 12W4 X^2,      W2        ]
    [     W2,        2W3 + 12W5 Y^2 ]"""
    h11 = 2.0 * W[1] + 12.0 * W[4] * X ** 2
    h12 = W[2] * np.ones_like(X)
    h22 = 2.0 * W[3] + 12.0 * W[5] * Y ** 2
    return h11, h12, h22


def convex_check_domain(W, X_MIN, X_MAX, Y_MIN, Y_MAX, n=201):
    """在紧集 Ω 上验证价值近似器的凸性（论文 Fig.3 的提法）
    返回 (最小特征值, 是否处处半正定)"""
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    h11, h12, h22 = hessian_expr(W, X, Y)
    tr = h11 + h22                                          # 迹
    det = h11 * h22 - h12 ** 2                              # 行列式
    lam_min = 0.5 * (tr - np.sqrt(np.maximum(tr ** 2 - 4.0 * det, 0.0)))
    return float(lam_min.min()), bool(lam_min.min() > 0.0)
