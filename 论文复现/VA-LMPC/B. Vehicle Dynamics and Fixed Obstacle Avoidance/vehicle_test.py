# =================== 车辆避障：结果验证与绘图（论文 IV-B） ===================
# 对应论文 IV-B 的图与表（仅复现本文方法自身的部分）：
#   Fig. 3  价值近似器在 z ∈ [0,40]×[0,5] 上的预测（凸性）
#   Fig. 4/5 第 1 轮的速度/偏航角输入 vs 开环最优（OLPS）输入
#   Fig. 6  第 1 轮轨迹 + 初始可行轨迹 + 最优轨迹（椭圆障碍）
#   Table III/IV/V 的 VA-LMPC 列：迭代代价、计算时间、存储代价（= 6 个参数）

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Ellipse
import vehicle_env as env
import va_basis as basis
import vehicle_va_mpc as vs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=4, suppress=True)

# =================== ① 载入训练结果 ===================
W = np.load(os.path.join(SCRIPT_DIR, "vehicle_W_final.npy"))
data = np.load(os.path.join(SCRIPT_DIR, "vehicle_data.npz"))
cost_hist = data["cost_hist"]
time_hist = data["time_hist"]
Z_f, U_f, J_f = data["Z_feasible"], data["U_feasible"], data["J_feasible"]

# =================== ② 用最终权重闭环评价（式 16 + 式 18） ===================
solver, lbx, ubx, lbg, ubg = vs.build_va_solver(env.N_PRED)
Z_va, U_va, _ = vs.rollout_va(solver, lbx, ubx, lbg, ubg, W, env.Z0, max_steps=60)

# =================== ③ OLPS 最优性基准（小椭圆、长时域） ===================
Z_ol, U_ol = env.solve_ocp(60, env.OBS_R, env.Z0, terminal=True)
J_ol = env.cost_to_go(Z_ol)[0]

# =================== ④ 结果汇总 ===================
print("=" * 92)
print("VA-LMPC 车辆避障复现结果（论文 IV-B）")
print("=" * 92)
print(f"{'迭代':>4} {'闭环代价':>13} {'相对 OLPS':>11} {'步数':>5} {'耗时(s)':>9}")
for i, (c, t_) in enumerate(zip(cost_hist, time_hist)):
    tag = "初始可行轨迹（大椭圆）" if i == 0 else f"第 {i} 轮"
    print(f"{i:>4} {c:>13.4f} {c / J_ol:>11.6f} {'-':>5} {t_:>9.2f}   {tag}")
print("-" * 92)
print(f"OLPS（小椭圆、长时域开环最优）：J = {J_ol:.4f}（论文 S-LMPC 收敛值 14944.3660）")
print("-" * 92)
print("论文 Table III/IV/V 的 VA-LMPC 列（相对指标）：")
print(f"  Objective（第 1 轮相对 OLPS）= {cost_hist[1] / J_ol:.6f}   （论文：VA-LMPC 第 1 轮 14944.7131，"
      f"相对 S-LMPC 收敛值 14944.3660 为 {14944.7131 / 14944.3660:.6f}）")
print(f"  Storage cost（参数个数）     = {basis.P_DIM}          （论文：VA-LMPC = 6）")
print(f"  CPU 时间（每轮）             = {np.mean(time_hist[1:]):.2f} s（论文：VA-LMPC 33.42 s / 轮，MATLAB）")
ell_va = ((Z_va[:, 0] - env.OBS_CENTER[0]) / env.OBS_R[0]) ** 2 + \
         ((Z_va[:, 1] - env.OBS_CENTER[1]) / env.OBS_R[1]) ** 2
dev_y = np.abs(np.interp(Z_va[:, 0], Z_ol[:, 0], Z_ol[:, 1]) - Z_va[:, 1]).max()
print(f"  第 1 轮轨迹：{len(U_va)} 步到达目标（判据 J* <= {env.J_STOP:g}），"
      f"末位置 {np.round(Z_va[-1], 4)}，约束满足 = {all(env.in_constraints(z, tol=1e-6) for z in Z_va)}"
      f"（椭圆约束最小值 = {ell_va.min():.6f} >= 1）")
print(f"  与 OLPS 轨迹：按 x 对齐的 y 最大偏差 = {dev_y:.4f}（y_max：VA-LMPC {Z_va[:, 1].max():.4f} vs OLPS {Z_ol[:, 1].max():.4f}）")

# =================== ⑤ 图 3：价值近似器曲面（z ∈ [0,40]×[0,5]） ===================
xs = np.linspace(0.0, 40.0, 161)
ys = np.linspace(0.0, 5.0, 121)
Xg, Yg = np.meshgrid(xs, ys)
Sg = np.stack([Xg - env.ZE[0], Yg - env.ZE[1]], axis=-1)        # 相对坐标 z - z_e
Qg = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    Qg[i, :] = basis.phi_matrix(Sg[i]) @ W

fig = plt.figure(figsize=(9.4, 5.5))
ax1 = fig.add_subplot(1, 1, 1, projection="3d")
surf = ax1.plot_surface(Xg, Yg, Qg, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("Cost")
ax1.set_xticks([0, 10, 20, 30, 40])             # 与论文 Fig.3 一样只标 4~5 个刻度
fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((4, 4))                     # z 轴用 10^4 档（与论文一致）
ax1.zaxis.set_major_formatter(fmt)
# 视角：azim=-40、低仰角 elev=12（接近平视，用户选定）；正交投影 + 拉长 y 方向的盒子，
# z 轴挪到左侧、右侧加 colorbar，整体布局对齐论文 Fig.3
ax1.set_proj_type("ortho")
ax1.view_init(elev=12, azim=-40)
# 盒子比例：a=-40°, e=12° 时单位长度在屏幕上的投影系数为 x:0.662、y:0.778，
# 取 (1.35, 1.15, 1.05) 使 x、y 两轴屏幕等长（1.15/1.35 ≈ 0.85 ✓），底边同时拉长（不至于太瘦）
ax1.set_box_aspect((1.35, 1.15, 1.05), zoom=1.18)
ax1.zaxis._axinfo["juggled"] = (1, 2, 0)        # matplotlib 私有接口：把 z 轴画到左侧
ax1.set_position([0.01, 0.075, 0.80, 0.855])    # 让 3D 盒子尽量填满画布（上下留出标题/轴标签）
cax = fig.add_axes([0.84, 0.26, 0.022, 0.50])   # colorbar 固定位置（避免自动布局挤动盒子）
fig.colorbar(surf, cax=cax)
ax1.set_title("Value approximator prediction of state cost (vehicle)")
fig.savefig(os.path.join(SCRIPT_DIR, "fig_vehicle_value.png"), dpi=150)
plt.close(fig)

# =================== ⑥ 图 4/5：输入对比（第 1 轮 vs OLPS） ===================
k_va = np.arange(len(U_va))
k_ol = np.arange(len(U_ol))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
axes[0].plot(k_va, U_va[:, 1], "-o", color="navy", markersize=4, label="VA-LMPC (1st iteration)")
axes[0].plot(k_ol, U_ol[:, 1], "--", color="red", label="OLPS")
axes[0].axhline(env.V_MAX, color="gray", linestyle=":"); axes[0].axhline(env.V_MIN, color="gray", linestyle=":")
axes[0].set_xlabel("time step k"); axes[0].set_ylabel("v"); axes[0].set_title("Speed input")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(k_va, U_va[:, 0], "-o", color="navy", markersize=4, label="VA-LMPC (1st iteration)")
axes[1].plot(k_ol, U_ol[:, 0], "--", color="red", label="OLPS")
axes[1].axhline(env.PHI_MAX, color="gray", linestyle=":"); axes[1].axhline(env.PHI_MIN, color="gray", linestyle=":")
axes[1].set_xlabel("time step k"); axes[1].set_ylabel(r"$\varphi$"); axes[1].set_title("Yaw angle input")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "fig_vehicle_inputs.png"), dpi=150)
plt.close(fig)

# =================== ⑦ 图 6：轨迹 + 障碍 ===================
fig, ax = plt.subplots(figsize=(9.0, 4.6))
ax.add_patch(Ellipse(env.OBS_CENTER, 2 * env.OBS_R[0], 2 * env.OBS_R[1],
                     facecolor="lightcoral", edgecolor="firebrick", alpha=0.55, label="Obstacle"))
ax.plot(Z_f[:40, 0], Z_f[:40, 1], "s--", color="tab:cyan", markersize=5,
        label="First feasible solution (large ellipse)")
ax.plot(Z_ol[:, 0], Z_ol[:, 1], "--o", color="red", markerfacecolor="none", markersize=5, label="OLPS")
ax.plot(Z_va[:, 0], Z_va[:, 1], "-o", color="navy", markersize=5, label="VA-LMPC (1st iteration)")
ax.plot(*env.Z0, "k*", markersize=12, label="Initial state $z_o$")
ax.plot(*env.ZE, "k^", markersize=9, label="Equilibrium $z_e$")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_xlim(-2, 44); ax.set_ylim(-1.5, 6.5)
ax.set_title("Vehicle trajectory with fixed obstacle (N = 8)")
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "fig_vehicle_traj.png"), dpi=150)
plt.close(fig)

print(f"\n已保存图：fig_vehicle_value.png、fig_vehicle_inputs.png、fig_vehicle_traj.png")
print(f"目录：{SCRIPT_DIR}")
