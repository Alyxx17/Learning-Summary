# =================== RLMPC Epi.1：在线学习 ===================
# 底层核心严格按论文 Algorithm 1 ：
#   ① 用当前 W 解 OP2 控制真实系统（施加 u_{0|k}）
#   ② 采样 100 点 + 实际轨迹点；逐点解 OP2 得标签 J（用本轮开始前的 W，标签冻结）
#   ③ 学习更新：最小化 Σ_j [J_j − WᵀΦ(x_j)]²
#      ——优化器采用"岭回归闭式解 + 部分更新"：
#        W_fit = (ΦᵀΦ + λI)⁻¹ ΦᵀJ；W ← W + LR·(W_fit − W)
#      LR 即有效学习率（论文 SGD 的逐点 α=1e-6 在此无折扣设定下会线性漂移，极难收敛）
#   ④ 外层停机：‖W^{t+1} − W^t‖ ≤ ε
# 验收基准（论文 Table II）：Epi.1 ACC = 625.0301

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from set_seed import set_seed
import unicycle_env as env
import poly_basis as basis
import mpc_solver as mpc

# =================== 超参数 ===================
N           = 5                     # 预测时域（论文 RLMPC 的 N=5）
SIM_STEPS   = 60                   # 闭环仿真步数（论文 60 步）
N_SAMPLES   = 100                   # 每轮随机采样点数 q=100
BASIS_ORDER = 4                     # 基函数阶数（1~4 阶全排列，p=34,论文没有给出具体的基函数）
RIDGE       = 1e-3                  # 岭正则系数 λ（压制多项式振荡）
LR          = 0.05                  # 部分更新系数（有效学习率）
EPS         = 1e-7                  # 外层停机阈值（论文 ε=10⁻⁷）
STOP_AT     = 25                    # 停学步数（论文"25 次迭代收敛"：第 25 轮学完后冻结 W；
                                    # 从论文的收敛图来看，似乎并未严格收敛，这里仿照论文
                                    # 用第25步的权重作为学习完毕的权重）
SEED        = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # 脚本所在目录（模型保存用）

set_seed(SEED)                      # 有随机采样，固定随机源保证可复现

# =================== 基函数与 OP2 求解器 ===================
indices = basis.build_monomial_indices(BASIS_ORDER, 3)
P_DIM   = len(indices)
solver, lbx, ubx, lbg, ubg = mpc.build_op2_solver(N, indices)
print(f"基函数：1~{BASIS_ORDER} 阶全排列，p = {P_DIM}")

# =================== Epi.1 在线学习主循环（Algorithm 1） ===================
rng = np.random.default_rng(SEED)   # 采样专用随机流（独立于 set_seed，仍可复现）

W = np.zeros(P_DIM)                 # 初始价值权重 W⁰ = 0（对应 MPCWTC）
x_hist   = [env.X0.copy()]          # 真实状态历史
u_hist   = []                       # 实际施加的控制历史
acc_hist = []                       # 累积代价历史 ACC
W_hist   = [W.copy()]               # 权重演化历史

x_cur    = env.X0.copy()            # 当前真实状态
u_guess  = np.zeros(2*N)            # 决策变量初始猜测（热启动）
acc      = 0.0                      # 累积代价
flag_converged = False              # 学习完成标志（Algorithm 1 的 Flag）
t_conv    = None                    # 收敛时的迭代步数

for k in range(1, SIM_STEPS + 1):
    # ---- ① 用当前 W 解 OP2，施加第一个动作到真实系统 ----
    u_opt, _, ok, _ = mpc.solve_op2(solver, lbx, ubx, lbg, ubg, x_cur, W, u_guess)
    if not ok:
        print(f"[步骤 {k}] 控制求解失败，停止"); break
    u_k = u_opt[0]                                  # u_{0|k}（2,）
    u_hist.append(u_k)
    acc += env.stage_cost(x_cur, u_k)               # 累加真实运行代价
    acc_hist.append(acc)
    x_cur = env.dynamics(x_cur, u_k)                # 真实系统前进一步
    x_hist.append(x_cur.copy())
    u_guess = np.concatenate([u_opt.reshape(-1)[2:], [0.0, 0.0]])   # 热启动

    # ---- ② 学习期：采样 + 生成训练数据 + 学习更新 ----
    if not flag_converged:
        W_old = W.copy()                            # 本轮开始前的 W（标签冻结用）
        samples = env.sample_states(N_SAMPLES, rng)                 # (3,100) 均匀随机
        samples = np.concatenate([samples, x_cur[:, None]], axis=1) # (3,101) 加实际轨迹点

        n_fail = 0                                  # 求解失败样本计数
        fail_statuses = Counter()                   # 失败原因统计
        J_list, Phi_list = [], []                   # 训练数据收集

        for j in range(samples.shape[1]):
            xj = samples[:, j]
            # 采样点用独立零初值：热启动初值与采样点最优解距离远会导致求解失败
            _, J_j, ok, status = mpc.solve_op2(solver, lbx, ubx, lbg, ubg, xj, W_old,
                                               np.zeros(2*N))
            if not ok:                              # 个别采样点求解失败则跳过
                n_fail += 1
                fail_statuses[status] += 1
                continue
            J_list.append(J_j)
            Phi_list.append(basis.phi_num(xj, indices, env.STATE_SCALE))

        # ---- ③ 学习更新：岭回归闭式解 + 部分更新 ----
        Phi_all = np.array(Phi_list)                # (n_valid, p)
        J_all   = np.array(J_list)                  # (n_valid,)
        W_fit   = np.linalg.solve(
            Phi_all.T @ Phi_all + RIDGE * np.eye(P_DIM), Phi_all.T @ J_all)
        W_new   = W + LR * (W_fit - W)              # 部分吸收拟合解（有效学习率）
        resid   = np.sqrt(np.mean((Phi_all @ W_fit - J_all) ** 2))

        # ---- ④ 外层停机检查：‖W^{t+1} − W^t‖ ≤ ε 或到达停学步数 ----
        delta = np.linalg.norm(W_new - W_old)
        W = W_new
        W_hist.append(W.copy())
        if (delta <= EPS or (STOP_AT is not None and k >= STOP_AT)) and not flag_converged:
            flag_converged = True
            t_conv = STOP_AT if STOP_AT is not None and k >= STOP_AT else k

        fail_info = ""
        if n_fail:
            top_status, top_cnt = fail_statuses.most_common(1)[0]
            fail_info = f"   失败原因: {top_status} x{top_cnt}"
        print(f"[步骤 {k:2d}] ACC = {acc:8.3f}   ‖W‖ = {np.linalg.norm(W):10.4f}"
              f"   ΔW = {delta:9.4f}   resid = {resid:8.4f}   J∈[{J_all.min():.1f}, {J_all.max():.1f}]"
              f"   失败样本 = {n_fail}{fail_info}"
              + (f"   ★ 收敛于第 {t_conv} 次迭代" if t_conv == k else ""))
    else:
        W_hist.append(W.copy())
        print(f"[步骤 {k:2d}] ACC = {acc:8.3f}   ‖W‖ = {np.linalg.norm(W):10.4f}   (学习已停止)")

# =================== 结果汇总与保存 ===================
x_hist   = np.array(x_hist)          # (T+1,3)
u_hist   = np.array(u_hist)          # (T,2)
acc_hist = np.array(acc_hist)        # (T,)
W_hist   = np.array(W_hist)          # (T+1,p)

print("\n========== Epi.1 结果 ==========")
print(f"收敛迭代数：{t_conv if t_conv else '未触发停机（60 步学满，W 温和演化）'}")
print(f"最终 ACC：{acc:.4f}（论文基准 625.0301）")
print(f"终点状态：{np.round(x_hist[-1], 4)}")
print(f"输入是否越界：v ∈ [{u_hist[:,0].min():.3f}, {u_hist[:,0].max():.3f}], "
      f"ω ∈ [{u_hist[:,1].min():.3f}, {u_hist[:,1].max():.3f}]")
print(f"状态 x 是否越界：[{x_hist[:,0].min():.3f}, {x_hist[:,0].max():.3f}]")

# 保存学到的价值权重（供后续 Epi.2 / 图表复现使用）
np.save(os.path.join(SCRIPT_DIR, "W_final.npy"), W)
np.save(os.path.join(SCRIPT_DIR, "basis_indices.npy"), np.array(indices, dtype=int))
np.savez(os.path.join(SCRIPT_DIR, "epi1_data.npz"),
         x=x_hist, u=u_hist, acc=acc_hist, w=W_hist)
print(f"W_final 与 epi1_data 已保存至 {SCRIPT_DIR}")

# =================== 可视化（英文标签） ===================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(x_hist[:, 0], x_hist[:, 1], "b-o", markersize=3, label="RLMPC Epi.1")
axes[0].axvline(env.X_MIN, color="r", linestyle="--", label="Constraint x=0")
axes[0].axvline(env.X_MAX, color="r", linestyle="--", label="Constraint x=2")
axes[0].scatter([env.X0[0]], [env.X0[1]], color="k", marker="o", s=60, zorder=5, label="Initial state")
axes[0].scatter([0], [0], color="k", marker="*", s=150, zorder=5, label="Target origin")
axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
axes[0].set_title("Closed-loop Trajectory (Epi.1)"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(np.arange(len(acc_hist)) + 1, acc_hist, "r-o", markersize=3)
axes[1].axhline(625.0301, color="g", linestyle="--", label="Paper ACC=625.0301")
axes[1].set_xlabel("Time step k"); axes[1].set_ylabel("ACC")
axes[1].set_title("Accumulated Cost (Epi.1)"); axes[1].legend(); axes[1].grid(True)

axes[2].plot(np.arange(len(W_hist)), W_hist)
axes[2].set_xlabel("Time step k / Iteration t"); axes[2].set_ylabel("Weight W_i")
axes[2].set_title("Evolution of Weight Vector W"); axes[2].grid(True)

plt.tight_layout()
plt.show()
