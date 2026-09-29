# =================== 车辆 VA-LMPC 训练（论文 IV-B，Algorithm 1） ===================
"""
与数值例子（IV-A）的差异：
  - 动力学/代价：z_{k+1} = z_k + [cos φ; sin φ]·v·Δt，l(z) = ||z - z_e||_Q^2（不含输入代价）
  - 每轮终止判据：J_N^{t,*}(z_k) <= 1e-4（论文 IV-B），而不是 ||x|| <= 1e-2
  - 障碍：初始可行轨迹用大椭圆 [a_x,a_y] = [10,5] 生成；VA-LMPC 运行用小椭圆 [2,1]
  - 初始化：用初始可行轨迹上 150 个采样状态（论文："150 sampled states"）
  - 价值近似器自变量取相对平衡点的坐标 z - z_e（代价为 ||z - z_e||^2，故价值函数自然只依赖 z - z_e）
  - 权重拟合用 **带结构约束** 的最小二乘（见 va_basis.fit_weights_convex）：
    普通最小二乘会在"跟踪误差方向"（数据未覆盖）外推出负值区，使 MPC 目标函数出现伪极小，
    导致 J* <= 1e-4 终止判据在离目标 ~10 m 处被误触发；加约束后 Q_v >= 0 且凸（同论文 Fig.3）。
"""

import os
import time
import numpy as np
import vehicle_env as env
import va_basis as basis
import vehicle_va_mpc as vs

# =================== 超参数 ===================
MAX_ITER  = 3           # m：最多迭代轮数（论文 IV-B 中 VA-LMPC 一轮即达最优）
MAX_STEPS = 60          # n：每轮闭环最大步数（安全上限；实际约 30 步）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(precision=4, suppress=True)

# =================== ① 初始化：用初始可行轨迹（大椭圆）的数据拟合 Q_v^0 ===================
t_feas = time.perf_counter()
Z_f, U_f = env.generate_feasible_trajectory()                    # 大椭圆 [10,5]，N = 150
J_f = env.cost_to_go(Z_f)
t_feas = time.perf_counter() - t_feas
sel = np.arange(1, env.N_SAMPLES_INIT + 1)                       # 前 150 个采样状态
W = basis.fit_weights_convex(Z_f[sel] - env.ZE, J_f[sel])        # 价值权重 W^0（相对坐标，带结构约束）

print("=" * 96)
print("VA-LMPC 车辆避障复现（论文 IV-B：式(27)(28)）")
print(f"模型：z_{{k+1}} = z_k + [cosφ; sinφ]·v·Δt，Δt = {env.DT}，l(z) = ||z - z_e||_Q^2（不含输入项）")
print(f"约束：|φ| <= π/3，|v| <= 5，z ∈ [0,0]-[50,20]；椭圆障碍中心 {env.OBS_CENTER}，"
      f"半径：可行轨迹用 {env.OBS_R_FEAS}、VA-LMPC 用 {env.OBS_R}")
print(f"起点 z_o = {env.Z0}，目标 z_e = {env.ZE}，预测时域 N = {env.N_PRED}，"
      f"终止判据 J* <= {env.J_STOP:g}")
print(f"初始可行轨迹：{len(U_f)} 步（{len(U_f)*env.DT:.1f} s，OCP 用时 {t_feas:.1f} s），"
      f"cost-to-go = {J_f[0]:.4f}（论文 Table III 迭代 0 = 15668.9409）")
print(f"初始权重 W^0 = {np.round(W, 6)}")
print("=" * 96)
print(f"{'迭代':>4} {'闭环代价':>13} {'累计改进':>11} {'步数':>5} {'ΔW':>10} "
      f"{'残差RMSE':>10} {'耗时(s)':>9}  备注")
print(f"{'0':>4} {J_f[0]:>13.4f} {'-':>11} {len(U_f):>5} {'-':>10} {'-':>10} {t_feas:>9.2f}  初始可行轨迹（大椭圆）")
print("-" * 96)

# =================== ② 迭代学习主循环（Algorithm 1） ===================
solver, lbx, ubx, lbg, ubg = vs.build_va_solver(env.N_PRED)

cost_hist = [J_f[0]]
W_hist    = [W.copy()]
time_hist = [t_feas]

for it in range(1, MAX_ITER + 1):
    t_start = time.perf_counter()
    W_old = W.copy()

    # ---- 滚动执行（式 16 + 式 18；终止判据 J* <= 1e-4）----
    Z_ep, U_ep, ok = vs.rollout_va(solver, lbx, ubx, lbg, ubg, W_old,
                                  env.Z0, max_steps=MAX_STEPS)
    J_ep = env.cost_to_go(Z_ep)
    J_ep_cost = J_ep[0]                                  # 本轮闭环代价（Σ l，含末状态）

    # ---- 用本轮轨迹的输入-目标对更新价值近似器（Alg.1 第 10-11 行）----
    W = basis.fit_weights_convex(Z_ep - env.ZE, J_ep)
    dW = float(np.linalg.norm(W - W_old))
    resid = float(np.sqrt(np.mean((basis.phi_matrix(Z_ep - env.ZE) @ W - J_ep) ** 2)))
    t_used = time.perf_counter() - t_start

    cost_hist.append(J_ep_cost)
    W_hist.append(W.copy())
    time_hist.append(t_used)

    note = f"到达目标（{len(U_ep)} 步内 J* <= {env.J_STOP:g}）" if ok else "求解失败"
    print(f"{it:>4} {J_ep_cost:>13.4f} {cost_hist[0] - J_ep_cost:>11.4f} "
          f"{len(U_ep):>5} {dW:>10.4f} {resid:>10.4f} {t_used:>9.2f}  {note}")

print("-" * 96)
_, eig = basis.psd_check(W)
lam_min, _ = basis.convex_check_domain(W, -40.0, 0.0, 0.0, 5.0, n=151)
tol = 1e-8                                            # 约束拟合允许半正定（特征值 = 0）
psd_ok = bool(eig.min() > -tol and W[4] >= -tol and W[5] >= -tol)
conv_ok = bool(lam_min > -tol)
print(f"最终权重 W* = {np.round(W, 6)}")
print(f"  二次部分特征值 {np.round(eig, 4)}（半正定 = {psd_ok}）；"
      f"相对坐标域 [-40,0]×[0,5] 内 Hessian 最小特征值 = {lam_min:.6f}（凸 = {conv_ok}）")
dcost = np.diff(cost_hist)
print(f"迭代代价单调不增 = {bool(np.all(dcost <= 1e-3))}（最大上升 {float(np.max(np.maximum(dcost, 0.0))):.4f}）")

# =================== ③ 保存结果 ===================
np.save(os.path.join(SCRIPT_DIR, "vehicle_W_final.npy"), W)
np.savez(os.path.join(SCRIPT_DIR, "vehicle_data.npz"),
         cost_hist=np.array(cost_hist),
         time_hist=np.array(time_hist),
         W_hist=np.array(W_hist),
         Z_feasible=Z_f, U_feasible=U_f, J_feasible=J_f,
         allow_pickle=False)
print(f"\n已保存：vehicle_W_final.npy、vehicle_data.npz -> {SCRIPT_DIR}")
