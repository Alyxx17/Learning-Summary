# =================== VA-LMPC 数值例子：结果验证与绘图 ===================
# 对应论文 IV-A 的图与表（仅复现本文方法自身的结果）：
#   Fig. 1  价值近似器对系统状态代价的预测（在紧集 Ω 上检验凸性）
#   Fig. 2  闭环状态轨迹被调节到原点
#   Table II 的 Ours 行：Objective（相对 OLPS）、存储代价 = 参数个数 6、CPU 时间
# 说明：论文用开环最优解 OLPS（直接法，N = 100）来验证本文方法结果的最优性，
#       OLPS 是论文自身的评价基准，因此保留；论文中与 S-LMPC / PI-ADP / ADHDP
#       等方法的对比不在本次复现范围内。

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # 只存图，不弹窗
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import double_integrator_env as env
import va_basis as basis
import va_lmpc_solver as vs
import olps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=4, suppress=True)

# =================== ① 载入训练结果 ===================
W = np.load(os.path.join(SCRIPT_DIR, "W_final.npy"))
data = np.load(os.path.join(SCRIPT_DIR, "va_data.npz"))
cost_hist = data["cost_hist"]                  # 每轮闭环代价（第 0 轮 = 初始可行轨迹）
time_hist = data["time_hist"]
X_f, U_f, J_f = data["X_feasible"], data["U_feasible"], data["J_feasible"]

# =================== ② 用最终权重闭环评价（式 16 + 式 18） ===================
solver, lbx, ubx, lbg, ubg = vs.build_va_solver(env.N_PRED)
X_va, U_va, _ = vs.rollout_va(solver, lbx, ubx, lbg, ubg, W, env.X0, max_steps=100)
J_va = env.trajectory_cost(X_va, U_va)         # 本文方法的实际闭环代价（Objective）

# =================== ③ OLPS 最优性基准 ===================
solver_ol, lbx_ol, ubx_ol, lbg_ol, ubg_ol = olps.build_olps_solver()
U_ol, X_ol, _, _, status_ol = olps.solve_olps(solver_ol, lbx_ol, ubx_ol, lbg_ol, ubg_ol)
J_ol = env.trajectory_cost(X_ol, U_ol)
ol_feas = all(env.in_constraints(X_ol[k]) for k in range(len(X_ol)))

# =================== ④ 结果汇总 ===================
print("=" * 88)
print("VA-LMPC 数值例子复现结果（论文 IV-A，双积分器）")
print("=" * 88)
print(f"{'迭代':>4} {'闭环代价':>14} {'相对 OLPS':>12} {'步数':>6} {'耗时(s)':>10}")
for i, (c, t_) in enumerate(zip(cost_hist, time_hist)):
    tag = "初始可行轨迹" if i == 0 else f"第 {i} 轮"
    print(f"{i:>4} {c:>14.4f} {c / J_ol:>12.6f} {'-':>6} {t_:>10.3f}   {tag}")
print("-" * 88)
print(f"OLPS（论文 Table I：N = 100，开环最优参考解）：J = {J_ol:.4f}，"
      f"约束满足 = {ol_feas}，状态 {status_ol}")
print("-" * 88)
print("论文 Table II 的 'Ours' 行（相对指标）：")
print(f"  CPU 时间（每轮平均）  = {np.mean(time_hist[1:]):.4f} s（本文 1.0000 = 基准）")
print(f"  Objective（相对 OLPS）= {J_va / J_ol:.6f}   （论文：Ours = 1.0000）")
print(f"  Storage cost（参数个数）= {basis.P_DIM}        （论文：Ours = 1.0000 = 6 个参数）")
print(f"  末状态 = {np.round(X_va[-1], 6)}，收敛步数 = {len(U_va)}，"
      f"输入范围 = [{U_va.min():.4f}, {U_va.max():.4f}]")
print(f"  状态是否满足约束：VA-LMPC = {all(env.in_constraints(x) for x in X_va)}")

# =================== ⑤ 图 1：价值近似器曲面（作图范围按论文 Fig.1） ===================
X_PLOT = (-15.0, 10.0)         # 论文 Fig.1 的 x 轴范围（含约束之外的 x > 0 区域）
Y_PLOT = (-6.0, 6.0)           # 论文 Fig.1 的 y 轴范围
xs = np.linspace(X_PLOT[0], X_PLOT[1], 161)
ys = np.linspace(Y_PLOT[0], Y_PLOT[1], 161)
Xg, Yg = np.meshgrid(xs, ys)
Qg = basis.value_grid(W, Xg, Yg)
# 凸性按论文的说法在约束集 Ω 上检验（Ω 之外不保证）
lam_min, conv_ok = basis.convex_check_domain(W, env.X_MIN[0], env.X_MAX[0],
                                             env.X_MIN[1], env.X_MAX[1])
W0 = data["W_hist"][0]                                    # 初始权重 W^0（由初始可行轨迹拟合）
lam0, conv0 = basis.convex_check_domain(W0, env.X_MIN[0], env.X_MAX[0],
                                        env.X_MIN[1], env.X_MAX[1])
print(f"\n价值近似器凸性（论文只声称\"在紧集 Ω 上为凸\"）：\n"
      f"  Ω = {{x | [-15,-1]^T <= x <= [0,6]^T}} 上 Hessian 最小特征值："
      f"初始 W^0 = {lam0:.2f}（凸 = {conv0}）；最终 W* = {lam_min:.2f}（凸 = {conv_ok}）\n"
      f"  作图范围 = 论文 Fig.1 的 x∈{X_PLOT}, y∈{Y_PLOT}")

fig = plt.figure(figsize=(8.0, 5.6))
ax1 = fig.add_subplot(1, 1, 1, projection="3d")
ax1.plot_surface(Xg, Yg, Qg, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("cost")
ax1.set_zticks([0.0, 5e4, 1e5, 1.5e5])           # z 轴刻度按论文：0/5/10/15（×10^4）
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((4, 4))                      # 固定用 10^4 这一档（与论文一致）
ax1.zaxis.set_major_formatter(fmt)
ax1.view_init(elev=25, azim=-135)                # 与论文 Fig.1 一致的视角
ax1.set_title("Value approximator prediction of state cost")
fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "fig_value_approximator.png"), dpi=150)
plt.close(fig)

# =================== ⑥ 图 2：闭环轨迹（x-y 相图，横轴 x、纵轴 y，论文 Fig.2 的画法） ===================
fig, ax = plt.subplots(figsize=(6.6, 5.0))
n_f_show = min(15, len(U_f))
n_ol_show = min(12, len(U_ol))
ax.plot(X_f[:n_f_show + 1, 0], X_f[:n_f_show + 1, 1], "s--", color="tab:cyan", markersize=5,
        label="First Feasible Solution")
ax.plot(X_va[:, 0], X_va[:, 1], "-o", color="navy", markersize=5, label="VA-LMPC")
ax.plot(X_ol[:n_ol_show + 1, 0], X_ol[:n_ol_show + 1, 1], "--o", color="red", markerfacecolor="none",
        markersize=5, label="OLPS")
ax.plot(env.X0[0], env.X0[1], "k*", markersize=11, label="Initial state $x_o$")
ax.plot(env.XE[0], env.XE[1], "k^", markersize=7, label="Equilibrium $x_e$")
ax.axvline(env.X_MAX[0], color="gray", linestyle=":", label="Constraint x = 0")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("State trajectory in x-y plane")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "fig_trajectory.png"), dpi=150)
plt.close(fig)

# =================== ⑦ 图 3：迭代代价收敛曲线 ===================
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(np.arange(len(cost_hist)), cost_hist, "b-o", markersize=5, label="VA-LMPC iteration cost")
ax.axhline(J_ol, color="r", linestyle="--", label=f"OLPS optimum = {J_ol:.2f}")
ax.set_xlabel("iteration i"); ax.set_ylabel("closed-loop cost")
ax.set_title("Iteration cost (iteration cost does not increase)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "fig_iteration_cost.png"), dpi=150)
plt.close(fig)

print(f"\n已保存图：fig_value_approximator.png、fig_trajectory.png、fig_iteration_cost.png")
print(f"目录：{SCRIPT_DIR}")
