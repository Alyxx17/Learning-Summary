# =================== 多项式基函数 Φ(x) ===================
# 价值函数近似：V(x) = W^T Φ(x)，基为 1~order 阶全部单项式（不含常数项）
# 论文非线性例子 p=30；本复现采用 1~4 阶全排列 p=34（论文未列出 30 项具体构成）

import itertools
import numpy as np
import casadi as ca

def build_monomial_indices(order, dim):
    """生成 1~order 阶、dim 维的全部单项式指数组合（不含常数项）
    例 dim=3, order=2 共 9 项：x,y,θ, x²,y²,θ², xy,xθ,yθ
    返回 list[list[int]]，长度 p"""
    indices = []
    for d in range(1, order + 1):                       # 遍历总阶数 1..order
        for combo in itertools.combinations_with_replacement(range(dim), d):
            idx = [0] * dim                             # 指数向量 [i_x, i_y, i_θ]
            for c in combo:                             # 统计每个变量出现次数
                idx[c] += 1
            indices.append(idx)
    return indices

def phi_num(x, indices, scale=None):
    """数值版 Φ(x)：x:(dim,) 的 ndarray -> (p,) 的 ndarray（学习回归时用）
    scale: 输入缩放向量（可选），先归一化 x/scale 再取单项式，改善数值条件数"""
    xs = x if scale is None else x / scale
    terms = []
    for idx in indices:
        term = 1.0
        for i, e in enumerate(idx):
            if e > 0:
                term *= xs[i] ** e                        # x_i 的 e 次幂
        terms.append(term)
    return np.array(terms)

def phi_expr(x_sym, indices, scale=None):
    """符号版 Φ(x)：x_sym 是 (dim,) 的 SX 向量 -> (p,1) 的 SX 表达式
    （嵌入 MPC 目标函数时用，梯度自动链式穿过动力学）
    scale: 输入缩放向量（可选），与 phi_num 保持一致"""
    xs = x_sym if scale is None else x_sym / scale
    terms = []
    for idx in indices:
        term = ca.SX(1)                                  # 常数 1（SX 标量）
        for i, e in enumerate(idx):
            if e > 0:
                term *= xs[i] ** e
        terms.append(term)
    return ca.vertcat(*terms)                            # 垂直拼接成 (p,1)
