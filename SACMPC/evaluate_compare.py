# =================== 对比实验: 固定权重基线 vs SAC ===================
# （学习用最小版：只保留两组——手动固定权重 与 SAC 在线学权重）
#   · 基线: 固定为"单位权重"（归一化坐标下全 1——无先验设定下的中性默认）
#   · 主角: SAC 确定性策略（加载 sac_diag_model.pth，输出 5 个对角权重）
#   · 评分: 由 unicycle_env.SCORE_MODE 选择（balanced/error/control）
#           —— 评分也是训练奖励口径，换口径后需重训 SAC 再对比
#   · 同一批 30 个评测场景（EVAL_SEED=123，与训练种子隔离，防"背题"）
# 指标: 等效总代价（= -回合回报 × C0）、RMSE、尾段误差、平均权重、求解失败
# 产出: 打印对比表 + 结果图（保存为 compare_result.png）
#
# 说明: RMSE/尾段误差在脚本外部读 env.state / env.X_ref 计算（不改环境）

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import unicycle_env as env_lib
from mpc_env import MPCEnv
from sac_agent import load_actor, DEVICE

# =================== 配置 ===================
EVAL_SEED   = 123                            # 评测种子（与训练隔离）
N_SCENARIOS = 30                             # 评测场景数
SIM_STEPS   = 80                             # 回合步数（与 MPCEnv 默认一致）
N_PRED      = 5                              # MPC 预测时域（仅用于重建场景画图）
BASELINE_W  = np.ones(5)                                # 基线权重 = 全 1（归一化坐标）
BASELINE_ACTION = env_lib.norm_action_of_weights(BASELINE_W)  # 反解成固定动作（恒用同一值）
# 对应公式: decode_action(a) = [q1,q2,q3,r1,r2]，a∈[-1,1]^5 → [0.1,20]^5（对数均匀映射）
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "sac_diag_model_error.pth")        # 最终模型
BEST_PATH   = os.path.join(SCRIPT_DIR, "sac_diag_model_best_error.pth")   # 训练中"评估最优"模型
# 选模开关: 训练会震荡，"最后一个模型"往往不是"最好的模型"，默认优先评估最优模型
#   True  = 若 sac_diag_model_best.pth 存在则评估它; False = 评估最终模型
USE_BEST_MODEL = True
if USE_BEST_MODEL and os.path.exists(BEST_PATH):
    MODEL_PATH = BEST_PATH
FIG_PATH    = os.path.join(SCRIPT_DIR, "compare_result.png")


# =================== 工具函数 ===================
def sac_policy_factory(actor):
    """确定性 SAC 策略: action = tanh(μ(obs))"""
    def policy(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, _ = actor(obs_t)
            return torch.tanh(mean).squeeze(0).cpu().numpy()
    return policy


def rollout(env, policy_fn):
    """跑一个回合，返回 (指标 dict, 状态轨迹 (T+1,3))"""
    obs, _ = env.reset()
    x_hist = [env.state.copy()]
    e_pos = [np.linalg.norm(env_lib.tracking_error(env.state, env.X_ref[0])[:2])]
    ep_return, n_fail, w_sum, w_n = 0.0, 0, np.zeros(5), 0
    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        n_fail += int(not info["solve_ok"])
        w_sum += info["weights"]
        w_n += 1
        x_hist.append(env.state.copy())
        # 外部读状态算位置误差（env.k 已 +1，X_ref[k] 有效）
        e_pos.append(np.linalg.norm(env_lib.tracking_error(env.state, env.X_ref[env.k])[:2]))
        if terminated or truncated:
            break
    e_pos = np.array(e_pos)
    k0 = int(len(e_pos) * 0.75)                  # 尾段 = 最后 25%
    rec = {"cost": -ep_return * env.c0,
           "rmse_pos": float(np.sqrt(np.mean(e_pos ** 2))),
           "tail_pos": float(np.mean(e_pos[k0:])),
           "mean_w": w_sum / w_n,
           "n_fail": n_fail}
    return rec, np.array(x_hist)


def run_method(policy_fn):
    """在 30 个固定评测场景上运行某策略（重建 env → 两组的场景流逐条一致）"""
    env = MPCEnv(seed=EVAL_SEED)
    recs, xs = [], []
    for _ in range(N_SCENARIOS):
        rec, x = rollout(env, policy_fn)
        recs.append(rec)
        xs.append(x)
    return recs, xs


# =================== 运行: 基线 + SAC ===================
print("===== 对比实验: 固定权重（单位阵，全 1）基线 vs SAC（30 个固定评测场景, EVAL_SEED=123）=====")
print(f"SAC 模型文件: {os.path.basename(MODEL_PATH)}")
print(f"评分口径: {env_lib.SCORE_MODE} | SCORE_W = {env_lib.SCORE_W}")

base_recs, base_xs = run_method(lambda obs: BASELINE_ACTION)
actor, _ = load_actor(MODEL_PATH)
sac_recs, sac_xs = run_method(sac_policy_factory(actor))

cb = np.array([r["cost"] for r in base_recs])
cs = np.array([r["cost"] for r in sac_recs])

# =================== 对比总表 ===================
print("\n===== 对比总表（30 场景平均）=====")
print(f"{'方法':<14}{'平均代价':>10}{'RMSE(m)':>10}{'尾段(m)':>10}{'失败':>6}")

def summary_line(name, recs, costs):
    print(f"{name:<14}{costs.mean():>10.4f}"
          f"{np.mean([r['rmse_pos'] for r in recs]):>10.4f}"
          f"{np.mean([r['tail_pos'] for r in recs]):>10.4f}"
          f"{sum(r['n_fail'] for r in recs):>6}")

summary_line("固定权重(单位阵)", base_recs, cb)
summary_line("SAC", sac_recs, cs)

print(f"\nSAC vs 固定基线: {cs.mean():.4f} vs {cb.mean():.4f}"
      f"（Δ = {(cs.mean() - cb.mean()) / cb.mean() * 100:+.2f}%）"
      f"  逐场景 胜 {np.sum(cs < cb)} / 负 {np.sum(cs > cb)} / 共 {N_SCENARIOS}")

# ---- 权重对照（归一化坐标 + 折算回物理单位）----
w_sac  = np.mean([r["mean_w"] for r in sac_recs], axis=0)
w_base = np.mean([r["mean_w"] for r in base_recs], axis=0)
print("\n权重对照（归一化坐标，q1..q3 | r1,r2）:")
print(f"  固定基线(全1): [{w_base[0]:.4f} {w_base[1]:.4f} {w_base[2]:.4f}"
      f" | {w_base[3]:.4f} {w_base[4]:.4f}]")
print(f"  SAC        : [{w_sac[0]:.4f} {w_sac[1]:.4f} {w_sac[2]:.4f}"
      f" | {w_sac[3]:.4f} {w_sac[4]:.4f}]")
phy = env_lib.physical_weights_of(w_sac)
print(f"SAC 折算回物理单位: [{phy[0]:.3f} {phy[1]:.3f} {phy[2]:.3f}"
      f" | {phy[3]:.5f} {phy[4]:.5f}]")

# =================== 可视化（图注英文；结尾保存结果图） ===================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图 1: 均值柱状图（误差棒 = ±1 SEM）
labels = ["Fixed (identity)", "SAC"]
means = np.array([cb.mean(), cs.mean()])
sems = np.array([cb.std(), cs.std()]) / np.sqrt(N_SCENARIOS)
axes[0].bar(labels, means, yerr=sems, capsize=6, color=["tab:gray", "tab:red"], alpha=0.85)
for i, m in enumerate(means):
    axes[0].text(i, m + sems[i] + 0.002, f"{m:.4f}", ha="center", fontsize=9)
axes[0].set_ylabel("Mean equivalent cost")
axes[0].set_ylim(bottom=0.9 * means.min(), top=means.max() * 1.08)
axes[0].set_title("Mean cost (30 eval scenarios, +/-1 SEM)")
axes[0].grid(True, axis="y")

# 图 2: 逐场景散点（SAC vs 固定基线）
axes[1].scatter(cb, cs, alpha=0.75, s=28, color="tab:red")
lims = [min(cb.min(), cs.min()) * 0.95, max(cb.max(), cs.max()) * 1.05]
axes[1].plot(lims, lims, "k--", lw=1, label="y = x")
axes[1].set_xlim(lims)
axes[1].set_ylim(lims)
axes[1].set_xlabel("Fixed (identity) cost")
axes[1].set_ylabel("SAC cost")
axes[1].set_title("Per-scenario cost: SAC vs baseline")
axes[1].legend()
axes[1].grid(True)

# 图 3: 差异最大的场景——轨迹对比（重建该场景参考序列，与环境场景流同序）
idx = int(np.argmax(np.abs(cs - cb)))
rng = np.random.default_rng(EVAL_SEED)
for _ in range(idx + 1):
    ref, X_ref, U_ref = env_lib.sample_reference(rng, SIM_STEPS + N_PRED)
    x0 = env_lib.sample_initial_state(rng, ref)

axes[2].plot(X_ref[:, 0], X_ref[:, 1], "k--", lw=1.2, label="Reference")
axes[2].plot(base_xs[idx][:, 0], base_xs[idx][:, 1], "b-", lw=1.5, label="Fixed (identity)")
axes[2].plot(sac_xs[idx][:, 0], sac_xs[idx][:, 1], "r-", lw=1.5, label="SAC")
axes[2].scatter(base_xs[idx][0, 0], base_xs[idx][0, 1], c="g", marker="o", s=50,
                zorder=5, label="Initial state")
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("y (m)")
axes[2].set_title(f"Scenario #{idx} (largest cost gap): trajectories")
axes[2].legend()
axes[2].grid(True)
axes[2].axis("equal")

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)
plt.show()

# =================== 保存结果图 ===================
print(f"\n结果图已保存: {FIG_PATH}")
