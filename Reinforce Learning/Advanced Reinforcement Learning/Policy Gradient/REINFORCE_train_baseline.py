# ============================================================
# REINFORCE Rainbow（基线 + 熵正则 + 回报标准化）训练代码
# 在保持 REINFORCE 核心框架（采样轨迹 → G_t 加权更新策略）不变的前提下，
# 叠加三个正交组件（类似 Rainbow DQN 的做法，对应笔记 1.5 / 1.6 / 1.7 节）：
#
#   改动① baseline（笔记 1.5）：损失用 (G_t - 基线)，减方差
#   改动② 回报标准化（笔记 1.7）：G_t 先减均值再除标准差，稳定梯度尺度
#   改动③ 熵正则（笔记 1.6）：损失加 -β·H(π(·|s))，防策略坍缩、鼓励探索
#
# 三者都只改"update 里的损失/优势计算"，不碰网络结构、采样、训练循环。
# 测试直接沿用 REINFORCE_test.py（网络结构未变，无需新测试文件）。
# ============================================================

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from set_seed import set_seed   # 从 set_seed 模块导入 set_seed 函数

# =================== 环境配置 ===================
env = gym.make("CartPole-v1")         # 经典平衡车环境（与 DQN 同环境）
state_dim  = env.observation_space.shape[0]  # 状态维度：4
action_dim = env.action_space.n              # 动作维度：2

# =================== 超参数 ===================
EPISODES = 1000       # 训练回合数
GAMMA    = 0.99       # 折扣因子
LR       = 1e-3       # 学习率
SEED     = 42         # 随机种子
ENTROPY_COEF = 0.01   # 熵正则系数 β（改动③：鼓励策略保持随机，防止坍缩）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =================== 策略网络定义（与无基线版完全一致） ===================
class PolicyNet(nn.Module):
    """输入状态，输出每个动作的概率（离散动作 → softmax）"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)   # 输出 logits
        )

    def forward(self, x):
        return self.fc(x)                # 输出形状: [batch, action_dim]

# =================== 计算折扣回报 G_t ===================
def compute_returns(rewards, gamma):
    """输入一回合的奖励列表，返回每个时刻的折扣回报 G_t
    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...   （反向迭代）
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32).to(DEVICE)

# =================== REINFORCE with Baseline 智能体 ===================
class REINFORCEAgent:
    def __init__(self):
        self.policy_net = PolicyNet(state_dim, action_dim).to(DEVICE)
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)

    def select_action(self, state):
        """按概率采样动作（随机策略），返回动作、log π(a|s)、熵 H(π(·|s))
        多返回一个熵：熵正则（改动③）需要在每一步记录分布熵，必须在采样时算
        注意：这里【不能】加 no_grad！loss 直接使用这份 log_probs 反向传播，
        no_grad 会切断梯度路径导致整个训练失效（与 PPO 不同）
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # [1, 4]
        logits = self.policy_net(state_t)                           # [1, 2]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()   # 改动③：多返回熵

    def update(self, log_probs, entropies, rewards):
        """整条轨迹的一次更新（Rainbow 组合版：baseline + 标准化 + 熵正则）
        三个改动全在这一段里，核心框架（轨迹 → 梯度上升）不变：
        """
        returns   = compute_returns(rewards, GAMMA)                 # [T]

        # ---- 改动① + ②：回报标准化（含基线） ----
        # ① baseline（笔记 1.5）：减均值，让评估围绕 0 波动（动态基线）
        # ② 标准化（笔记 1.7）：再除标准差，让梯度尺度稳定（1e-8 防除零）
        # 这 1 行同时实现了 1.5 和 1.7 两个组件
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)   # [T]

        log_probs  = torch.cat(log_probs)                           # [T]

        # ---- ③ 熵正则（笔记 1.6）：损失加 -β·H ----
        # 熵 H = -Σ_a π(a|s)logπ(a|s) 越大 → 策略越随机
        # 加负熵项 = 鼓励随机 → 防止 CartPole 奖励恒正导致策略坍缩
        entropies = torch.cat(entropies)                            # [T]
        loss = -(log_probs * advantages).mean() - ENTROPY_COEF * entropies.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

# =================== 训练循环 ===================
agent = REINFORCEAgent()
episode_rewards = []

print("===== REINFORCE with Baseline 开始训练 =====")
env.reset(seed=SEED)
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    log_probs  = []   # 存每步的 log π(a_t|s_t)
    rewards    = []   # 存每步的奖励 r_t
    entropies  = []   # 存每步的分布熵（改动③：熵正则需要）

    # ---------- 1. 用当前策略采样一整条轨迹 ----------
    while not done:
        action, log_prob, entropy = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(reward)
        state = next_state
        total_reward += reward

    # ---------- 2. 回合结束：用整条轨迹更新一次 ----------
    agent.update(log_probs, entropies, rewards)

    episode_rewards.append(total_reward)

    if (ep + 1) % 100 == 0:
        avg = np.mean(episode_rewards[-100:])
        print(f"回合 {ep+1:4d} | 平均奖励(近100回合): {avg:.2f}")
print("===== 训练完成 =====")

#===================保存模型 ===================
# 保存为 reinforce_cartpole.pth：与 REINFORCE_test.py 的加载名一致，
# 因此测试代码可直接沿用 REINFORCE_test.py，无需为组合版单独写测试文件
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reinforce_cartpole.pth")
torch.save(agent.policy_net.state_dict(), SAVE_PATH)
print(f"模型已保存为 {SAVE_PATH}")

# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("REINFORCE with Baseline Training Curve")
plt.show()
env.close()

# ============================================================================
# 【三个改动的位置汇总】（全部在 update() 里，其余代码与无基线版一致）
#
# 改动① baseline（笔记 1.5）：advantages = returns - 基线
#        本文件用"减均值"实现动态基线（等价于按回合回报均值中心化）
# 改动② 回报标准化（笔记 1.7）：advantages 再除以标准差，稳定梯度尺度
#        （①+② 合并成一行：advantages = (returns - returns.mean()) / (returns.std() + 1e-8)）
# 改动③ 熵正则（笔记 1.6）：loss 加 -β·H(π(·|s))，防策略坍缩、鼓励探索
#
# 测试：网络结构未变，直接运行 REINFORCE_test.py 即可（加载 reinforce_cartpole.pth）
# 下一步（1.8 节 A2C）：把"常数/统计基线"换成 V(s) 神经网络（critic），
# 基线随状态变化、更精准，这就是 Actor-Critic 的雏形。
# ============================================================================
