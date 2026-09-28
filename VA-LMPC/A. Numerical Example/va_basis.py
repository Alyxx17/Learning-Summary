# =================== 价值近似器的基函数 Φ(x)（论文式(24)、(26)） ===================
# 线性权重网络：Q_v(x) = W^T Φ(x)                                    （式 24）
# 基函数（论文式(26)）：φ(x) = [1, X^2, X Y, Y^2, X^4, Y^4]^T，n_v = 6
# 权重 W 的求法："obtained by using quadratic programming, e.g., by minimizing
# squared residual"（论文）→ 最小二乘

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


def value_grid(W, Xg, Yg):
    """在网格上求价值函数（画价值近似器曲面用）：Xg, Yg 为 meshgrid -> (ny,nx)"""
    X, Y = Xg, Yg
    return (W[0] * np.ones_like(X) + W[1] * X ** 2 + W[2] * X * Y
            + W[3] * Y ** 2 + W[4] * X ** 4 + W[5] * Y ** 4)


# =================== 凸性检查（论文 Fig.1："在紧集 Ω 上为凸函数"） ===================
def psd_check(W):
    """全局半正定检查（仅对论文式(26) 的 6 项基）：二次部分矩阵半正定 + X^4、Y^4 系数非负"""
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
    """在紧集 Ω 上验证价值近似器的凸性（论文 Fig.1 的提法）
    返回 (最小特征值, 是否处处半正定)"""
    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    h11, h12, h22 = hessian_expr(W, X, Y)
    tr = h11 + h22                                          # 迹
    det = h11 * h22 - h12 ** 2                              # 行列式
    lam_min = 0.5 * (tr - np.sqrt(np.maximum(tr ** 2 - 4.0 * det, 0.0)))
    return float(lam_min.min()), bool(lam_min.min() > 0.0)
