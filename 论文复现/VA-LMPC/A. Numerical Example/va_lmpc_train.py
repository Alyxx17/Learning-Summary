# =================== VA-LMPC 训练主程序（论文 Algorithm 1） ===================
"""
论文 Algorithm 1 对应关系：
  1 行   初始化数据集 D、价值近似器、x_0、最大迭代 m、最大步数 n   → 本文件 ① 初始化
  2-3 行 for i = 1..m, for k = 1..n                              → 主循环
  4-5 行 解问题(16) 并施加最优序列首元素 u_{0|k}^{*,i}（式 18）    → vs.rollout_va
  6-7 行 测量状态、收集历史状态与输入                              → X_ep / U_ep
  8 行   ||x_k^i|| <= 10^-2 则跳出本轮循环                         → STOP_TOL
  10 行  计算沿轨迹的 cost-to-go（式(12) 的数据标签）              → env.cost_to_go
  11 行  用本轮轨迹的输入-目标对更新价值近似器（式(14)）           → basis.fit_weights

初始化（论文）："We initialize the value approximator by using 100 sampled states from
the feasible trajectory and their corresponding cost data."
"""

import os
import time
import numpy as np
import double_integrator_env as env
import va_basis as basis
import va_lmpc_solver as vs

# =================== 超参数 ===================
MAX_ITER    = 5         # m：最大迭代轮数（算法第 1 轮即达最优，设 5 轮用于观察收敛行为）
MAX_STEPS   = 100       # n：每轮闭环最大步数
RIDGE       = 1e-8      # 最小二乘岭正则（仅数值稳定，远小于数据尺度）
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(precision=4, suppress=True)
#输出的浮点数保留 4 位小数，并且禁止使用科学计数法

# =================== ① 初始化：用初始可行轨迹的数据拟合 Q_v^0 ===================
X_f, U_f = env.generate_feasible_trajectory()    # 可行轨迹 X^0（开环滑行段 + OCP 尾巴）
J_f = env.cost_to_go(X_f, U_f)                                  # 各状态 cost-to-go
sel = np.arange(1, env.TRAJ_STEPS + 1)                          # 式(12) 中 t >= 1 的采样状态
W = basis.fit_weights(X_f[sel], J_f[sel], RIDGE)                # 价值权重 W^0

feasible_ok = all(env.in_constraints(X_f[k]) for k in range(len(X_f)))
print("=" * 92)
print("VA-LMPC 数值例子复现（论文 IV-A：双积分器，式(25)）")
print(f"系统：x_{{k+1}} = [[1,1],[0,1]]x_k + [0,-1]^T u_k，Q = 100 I_2，R = 0.5")
print(f"约束：-2 <= u <= 2，[-15,-1]^T <= x <= [0,6]^T，x_o = [-14,2]^T，N = {env.N_PRED}")
print(f"基函数：φ(x) = [1, X^2, XY, Y^2, X^4, Y^4]^T（n_v = {basis.P_DIM}）")
print(f"初始可行轨迹：{len(U_f)} 步，约束满足 = {feasible_ok}，"
      f"末状态 = {np.round(X_f[-1], 6)}，cost-to-go = {J_f[0]:.4f}")
psd_ok, eig = basis.psd_check(W)
lam_min, conv_ok = basis.convex_check_domain(W, env.X_MIN[0], env.X_MAX[0],
                                             env.X_MIN[1], env.X_MAX[1])
print(f"初始权重 W^0 = {np.round(W, 6)}")
print(f"  二次部分特征值 {np.round(eig, 4)}（全局半正定 = {psd_ok}）；"
      f"Ω 域内 Hessian 最小特征值 = {lam_min:.3f}（Ω 上凸 = {conv_ok}）")
print("=" * 92)
print(f"{'迭代':>4} {'闭环代价':>12} {'累计改进':>10} {'步数':>5} {'ΔW':>10} "
      f"{'残差RMSE':>10} {'耗时(s)':>9}  备注")
print(f"{'0':>4} {J_f[0]:>12.4f} {'-':>10} {len(U_f):>5} {'-':>10} {'-':>10} {'-':>9}  初始可行轨迹")
print("-" * 92)

# =================== ② 迭代学习主循环（Algorithm 1） ===================
solver, lbx, ubx, lbg, ubg = vs.build_va_solver(env.N_PRED)

cost_hist = [J_f[0]]                         # 每轮闭环代价（第 0 轮 = 初始可行轨迹）
W_hist    = [W.copy()]                       # 权重演化
time_hist = [0.0]

for it in range(1, MAX_ITER + 1):
    t_start = time.perf_counter()
    W_old = W.copy()

    # ---- 滚动执行：每步解问题(16)，施加首元素（式 18）----
    X_ep, U_ep, ok = vs.rollout_va(solver, lbx, ubx, lbg, ubg, W_old, env.X0, MAX_STEPS)
    J_ep_episode = env.trajectory_cost(X_ep, U_ep)       # 本轮闭环实际代价

    # ---- 用本轮轨迹的输入-目标对更新价值近似器（Alg.1 第 10-11 行，式(12)(14)）----
    # 与官方 VA_MPC.m 一致：Q_c = nlinfit(x_LMPC', IterationCost', @Q_F, Q_c)
    # —— 都只用「本轮闭环轨迹」的状态及其 cost-to-go 更新，不累积历轮数据
    J_new = env.cost_to_go(X_ep, U_ep)                   # 沿轨迹的 cost-to-go 标签
    W = basis.fit_weights(X_ep, J_new, RIDGE)            # 最小二乘拟合 W^i

    dW = float(np.linalg.norm(W - W_old))
    resid = float(np.sqrt(np.mean((basis.phi_matrix(X_ep) @ W - J_new) ** 2)))
    t_used = time.perf_counter() - t_start

    cost_hist.append(J_ep_episode)
    W_hist.append(W.copy())
    time_hist.append(t_used)

    note = f"收敛：{len(U_ep)} 步内 ‖x‖ <= {env.STOP_TOL:g}" if ok else "求解失败"
    print(f"{it:>4} {J_ep_episode:>12.4f} {cost_hist[0] - J_ep_episode:>10.4f} "
          f"{len(U_ep):>5} {dW:>10.4f} {resid:>10.4f} {t_used:>9.3f}  {note}")

print("-" * 92)
psd_ok, eig = basis.psd_check(W)
lam_min, conv_ok = basis.convex_check_domain(W, env.X_MIN[0], env.X_MAX[0],
                                             env.X_MIN[1], env.X_MAX[1])
print(f"最终权重 W* = {np.round(W, 6)}")
print(f"  二次部分特征值 {np.round(eig, 4)}（全局半正定 = {psd_ok}）；"
      f"Ω 域内 Hessian 最小特征值 = {lam_min:.3f}（Ω 上凸 = {conv_ok}）")
dcost = np.diff(cost_hist)
rel = float(np.max(np.maximum(dcost, 0.0)) / cost_hist[0]) if len(cost_hist) > 1 else 0.0
print(f"迭代代价单调不增 = {bool(np.all(dcost <= 1e-3))}"
      f"（最大上升 {float(np.max(np.maximum(dcost, 0.0))):.4f}，相对 {rel:.2e}；"
      f"论文：iteration cost does not increase with iteration count）")

# =================== ③ 保存结果 ===================
np.save(os.path.join(SCRIPT_DIR, "W_final.npy"), W)
np.savez(os.path.join(SCRIPT_DIR, "va_data.npz"),
         cost_hist=np.array(cost_hist),
         time_hist=np.array(time_hist),
         W_hist=np.array(W_hist),
         X_feasible=X_f, U_feasible=U_f, J_feasible=J_f,
         allow_pickle=False)
print(f"\n已保存：W_final.npy、va_data.npz -> {SCRIPT_DIR}")
