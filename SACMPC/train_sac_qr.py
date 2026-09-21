# =================== SAC × MPC 训练: 在线学习 Q/R 对角权重 ===================
# 干什么: SAC 每步输出 5 维归一化动作 → MPCEnv 解码为权重向量 w=[q1,q2,q3,r1,r2]
#         （即 Q、R 的对角线本身，作用在归一化误差/控制偏差上）→ 解 MPC → 闭环
# 结构沿用 RL_py/SAC/SAC_train.py: START_STEPS 预热 + 每步 update + 周期评估
# 验收: 学习曲线不崩; balanced 口径下与单位默认基线"相当"即可（默认与评分同源、无赢面）；
#       error/control 口径（错配设定）下 SAC 应优于单位默认基线

import os
import numpy as np
import matplotlib.pyplot as plt
from set_seed import set_seed
import unicycle_env as env_lib
from mpc_env import MPCEnv
from sac_agent import SACAgent, DEVICE

# =================== 配置 ===================
SEED           = 42          # 训练随机源（场景采样 / 网络初始化 / 预热随机动作）
EVAL_SEED      = 123         # 评估场景专用种子（与训练隔离，防"背题"）
EPISODES       = 2000         # 训练回合数（80 步/回合 → 共 160000 步）
START_STEPS    = 10000      # 前 10000步用均匀随机动作预热
EVAL_INTERVAL  = 100         # 每多少回合评估一次
EVAL_SCENARIOS = 30          # 每次评估用多少个固定评测场景
LOG_INTERVAL   = 25          # 每多少回合打印训练信息

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#dirname是获取当前脚本所在的目录，abspath是获取当前脚本的绝对路径，os.path.abspath(__file__)返回当前脚本的绝对路径，os.path.dirname()返回该路径的目录部分。最终SCRIPT_DIR就是当前脚本所在的目录路径。
MODEL_PATH = os.path.join(SCRIPT_DIR, "sac_diag_model_error.pth")        # 最终模型（训练结束时保存）
BEST_PATH  = os.path.join(SCRIPT_DIR, "sac_diag_model_best_error.pth")   # 训练中"评估最优"模型（防"最后跑偏"）
FIG_PATH   = os.path.join(SCRIPT_DIR, "sac_diag_curve_error.png")        # 训练结果图（结尾保存）

set_seed(SEED)

# =================== 环境与智能体 ===================
env   = MPCEnv(seed=SEED)                       # 训练环境（场景随机流在环境内部）
agent = SACAgent(env.obs_dim, env.action_dim)
rng   = np.random.default_rng(SEED)             # 预热随机动作专用流（可复现）

print(f"设备: {DEVICE} | 训练 {EPISODES} 回合 × {env.sim_steps} 步 | 预热 {START_STEPS} 步")
print(f"评分口径: {env_lib.SCORE_MODE} | SCORE_W = {env_lib.SCORE_W}")
print("===== SAC 开始训练（在线学 Q/R 对角权重）=====")


# =================== 评估函数 ===================
def evaluate(agent):
    """确定性策略在固定评测场景集上的表现（每次评估场景完全相同）
    返回: (平均等效总代价, 平均权重向量 (5,), 累计求解失败数)
    """
    eval_env = MPCEnv(seed=EVAL_SEED)   # 每次重建 → 场景序列与上次一致（防背题验证）
    cost_sum, w_sum, w_n, n_fail = 0.0, np.zeros(5), 0, 0
    for _ in range(EVAL_SCENARIOS):
        obs, _ = eval_env.reset()
        ep_return = 0.0
        while True:
            action = agent.select_action(obs, eval_mode=True)   # 确定性动作 tanh(μ)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += reward
            w_sum += info["weights"]
            w_n += 1
            n_fail += int(not info["solve_ok"])
            if terminated or truncated:
                break
        cost_sum += -ep_return * eval_env.c0    # 回合回报 → 等效总评价代价
    return cost_sum / EVAL_SCENARIOS, w_sum / w_n, n_fail


# =================== 训练循环 ===================
episode_returns  = []    # 每回合回报
episode_weights  = []    # 每回合平均权重向量 (5,)（观察 SAC 学到的 Q/R 对角）
eval_history     = []    # (回合数, 平均代价, 平均权重×5)
total_steps, total_fails, total_solve_time = 0, 0, 0.0
best_eval_cost   = float("inf")   # 训练中最好的评估代价（用于保存"最优模型"）

for ep in range(EPISODES):
    obs, _ = env.reset()
    ep_return, ep_w_sum, ep_steps = 0.0, np.zeros(5), 0
    done = False

    while not done:
        # ---------- 1. 选择动作 ----------
        if total_steps < START_STEPS:
            action = rng.uniform(-1.0, 1.0, size=env.action_dim)   # 预热: 均匀随机旋钮
        else:
            action = agent.select_action(obs)                      # 策略采样（探索来自高斯）

        # ---------- 2. 执行动作 ----------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---------- 3. 存经验（off-policy: 先存后学） ----------
        agent.store_transition(obs, action, reward, next_obs, done)

        # ---------- 4. 学一步（池不满批大小自动跳过） ----------
        agent.update()

        obs = next_obs
        ep_return += reward
        ep_w_sum += info["weights"]
        ep_steps += 1
        total_steps += 1
        total_fails += int(not info["solve_ok"])
        total_solve_time += info["solve_time"]

    episode_returns.append(ep_return)
    episode_weights.append(ep_w_sum / ep_steps)

    # ---------- 打印训练进度 ----------
    if (ep + 1) % LOG_INTERVAL == 0:
        avg   = np.mean(episode_returns[-LOG_INTERVAL:])
        avg_w = np.mean(episode_weights[-LOG_INTERVAL:], axis=0)
        print(f"回合 {ep+1:4d} | 步数 {total_steps:6d} | 最近{LOG_INTERVAL}回合平均回报 {avg:8.2f}"
              f" | 平均等效总代价 {-avg * env.c0:.4f}"
              f" | 平均权重 [{avg_w[0]:.3f} {avg_w[1]:.3f} {avg_w[2]:.3f} |"
              f" {avg_w[3]:.3f} {avg_w[4]:.3f}] | α = {agent.alpha.item():.3f}")

    # ---------- 定期评估 ----------
    if (ep + 1) % EVAL_INTERVAL == 0:
        eval_cost, eval_w, n_fail = evaluate(agent)
        eval_history.append([ep + 1, eval_cost, *eval_w])
        print(f"           [评估] {EVAL_SCENARIOS} 个固定场景平均等效总代价 {eval_cost:.4f}"
              f" | 平均权重 [{eval_w[0]:.3f} {eval_w[1]:.3f} {eval_w[2]:.3f} |"
              f" {eval_w[3]:.3f} {eval_w[4]:.3f}] | 求解失败 {n_fail}")

        # 评估新低 → 保存"最优模型"（训练后期会震荡，"最后一个"往往不是"最好的"）
        if eval_cost < best_eval_cost:
            best_eval_cost = eval_cost
            agent.save(BEST_PATH)
            print(f"           [最优模型] 评估代价新低 {eval_cost:.4f} → 已保存 {os.path.basename(BEST_PATH)}")

print("===== 训练完成 =====")

# =================== 保存模型 ===================
agent.save(MODEL_PATH)
print(f"模型已保存: {MODEL_PATH}")
print(f"全程统计: 平均单步 MPC 求解 {total_solve_time / total_steps * 1e3:.2f} ms"
      f" | 求解失败 {total_fails} 次")

# =================== 学习曲线（图注英文） ===================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(episode_returns, alpha=0.35, label="Episode Return")
window = LOG_INTERVAL
if len(episode_returns) >= window:
    smooth = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(episode_returns)), smooth,
            label=f"Moving Average ({window})")
if eval_history:
    ev = np.array(eval_history)
    ax.plot(ev[:, 0] - 1, [-c / env.c0 for c in ev[:, 1]], "r-o", markersize=4,
            label="Eval (deterministic)")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("SAC Training Curve (online cost-weight learning)")
ax.legend()
ax.grid(True)

ax = axes[1]
w_arr = np.array(episode_weights)                 # (N, 5)
names = ["q1 (x)", "q2 (y)", "q3 (theta)", "r1 (v)", "r2 (omega)"]
for i, nm in enumerate(names):
    ax.plot(w_arr[:, i], alpha=0.35, lw=1.0, label=nm)
    if len(w_arr) >= window:
        smooth = np.convolve(w_arr[:, i], np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(w_arr)), smooth, lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("Episode")
ax.set_ylabel("Learned weight (normalized coords)")
ax.set_title("Learned Q/R Diagonal Weights (normalized coords)")
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)
plt.show()
print(f"训练结果图已保存: {FIG_PATH}")
