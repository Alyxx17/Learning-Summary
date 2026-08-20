# ============================================================
# PPO（Proximal Policy Optimization）训练代码 —— LunarLander-v3
# 相比 A2C 的三个新东西（对应笔记章节三）：
#   1. 采样时额外存 log_probs_old（旧策略的对数概率）
#   2. 同一批数据做 EPOCHS 个 epoch × mini-batch 更新
#      （重要性采样让"旧数据可复用"，解决样本效率）
#   3. clip 目标：min(r·A, clip(r, 1±ε)·A) 夹住更新幅度
#      （近似信任区域，解决"步长难调/震荡"）
# 其余（共享网络 / GAE / 熵 / 优势标准化 / 并行环境）与 A2C 完全一致
# 结尾有整体框架的运行流程
# ============================================================

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from set_seed import set_seed

# =================== 环境配置（并行 N 个） ===================
N_ENVS = 8     # 并行环境数
# gymnasium 批量环境：一次 reset/step 同时操作 N 个环境
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("LunarLander-v3") for _ in range(N_ENVS)]
)
# LunarLander：状态 8 维，动作 4 个（与 A2C 相同）
state_dim  = envs.single_observation_space.shape[0]   # 8
action_dim = envs.single_action_space.n               # 4

# =================== 超参数 ===================
ROLLOUT_STEPS   = 256        # T：每次更新前滚动收集多少步（每批共 N×T=2048 条样本）
UPDATES         = 250        # 总更新次数（每批 2048 步 → 共约 51 万步；没学会可加大）
MINI_BATCH_SIZE = 64         # mini-batch 大小（2048/64 = 32 个 mini-batch）
EPOCHS          = 4          # 同一批数据重复使用的次数（PPO 的核心：多 epoch 复用）
GAMMA           = 0.99       # 折扣因子
LAM             = 0.95       # GAE 的 λ
LR              = 3e-4       # 学习率
CLIP_EPS        = 0.2        # clip 的 ε：概率比只允许在 [0.8, 1.2] 内自由优化（笔记 3.3）
ENTROPY_COEF    = 0.01       # 熵正则系数 β（笔记 1.6，防策略坍缩）
VALUE_COEF      = 0.5        # critic 损失权重 c1
EVAL_INTERVAL   = 20         # 每多少次更新评估一次
EVAL_EPISODES   = 10         # 每次评估跑几局
SEED            = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =================== 共享网络（actor + critic，与 A2C 完全一致） ===================
class ActorCriticNet(nn.Module):
    """actor 和 critic 共享底层特征提取层（类比 Dueling 的共享底层）"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(256, action_dim)   # → 各动作 logits
        self.critic_head = nn.Linear(256, 1)            # → V(s)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor_head(features)              # [batch, action_dim]
        value  = self.critic_head(features).squeeze(-1) # [batch]
        return logits, value

# =================== GAE：广义优势估计（与 A2C 完全一致，笔记 2.6） ===================
def compute_gae(rewards, values, dones, last_values, last_dones, gamma, lam):
    """输入：rewards/values/dones 形状 [T, N]（时间在前），
            last_values [N]、last_dones [N]（窗口末尾 bootstrap）
    输出：advantages [T, N]（GAE），returns [T, N]（供 critic 训练）
    反向迭代：A_t = δ_t + γλ·(1-done)·A_{t+1}，δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            # 窗口末尾：用"窗口末状态"的价值做 bootstrap（若未结束）
            next_value = last_values              # [N]
            next_non_terminal = 1.0 - last_dones  # [N]
        else:
            next_value = values[t + 1]            # [N]
            next_non_terminal = 1.0 - dones[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

# =================== PPO 智能体 ===================
class PPOAgent:
    def __init__(self):
        self.net = ActorCriticNet(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

    def select_actions(self, states):
        """对 N 个状态同时采样动作（批量）
        返回：actions [N]、log_probs [N]（记作 log π_old，更新时要存）、entropies [N]
        注意：必须用 no_grad 包裹——log_probs_old 是"旧策略的常数"，不带计算图，
        否则 PPO 多 epoch 复用同一批数据时会多次 backward 经过它而报错
        """
        states_t = torch.FloatTensor(states).to(DEVICE)   # [N, 8]
        with torch.no_grad():
            logits, _ = self.net(states_t)                # [N, 4]
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
        return actions, log_probs, entropies

    def update(self, states, actions, log_probs_old, advantages, returns):
        """PPO 更新：同一批数据做 EPOCHS 个 epoch 的 mini-batch 更新
        核心（笔记 3.3）：
          r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))     # 概率比
          loss = -min(r·A, clip(r, 1±ε)·A).mean() + c1·MSE(V, return) - c2·H
        """
        # ---- 打平成 [N*T, ...]（顺序：时间在前，与收集顺序一致）----
        states        = torch.FloatTensor(np.array(states)).reshape(-1, state_dim).to(DEVICE)  # [N*T, 8]
        actions       = torch.cat(actions)          # [N*T]
        log_probs_old = torch.cat(log_probs_old)    # [N*T]（旧策略的对数概率，保持不变）
        advantages    = advantages.reshape(-1)      # [N*T]
        returns       = returns.reshape(-1)         # [N*T]

        # ---- 多 epoch × mini-batch（同一批数据反复用！）----
        for _ in range(EPOCHS):
            perm = torch.randperm(states.size(0))   # 每个 epoch 重新打乱顺序
            for i in range(0, states.size(0), MINI_BATCH_SIZE):
                idx = perm[i:i + MINI_BATCH_SIZE]
                mb_states, mb_actions = states[idx], actions[idx]
                mb_log_probs_old, mb_adv = log_probs_old[idx], advantages[idx]
                mb_returns = returns[idx]

                # 用【当前】θ 重算 log π_θ 和 V(s)（每次 mini-batch 都在变）
                logits, values = self.net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(mb_actions)
                entropy   = dist.entropy()

                # ---- clip 目标（笔记 3.3）：好的更新有上限、坏的更新有下限 ----
                ratio = torch.exp(log_probs - mb_log_probs_old)                    # r_t(θ)
                surr1 = ratio * mb_adv                                            # 未裁剪
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv    # 裁剪后
                actor_loss  = -torch.min(surr1, surr2).mean()                      # 取 min → 保守

                critic_loss  = ((values - mb_returns) ** 2).mean()                # 同 A2C
                entropy_loss = entropy.mean()                                      # 同 A2C（多 epoch 时用当前 θ 重算）

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

# =================== 评估函数（与 A2C 相同，用真实奖励） ===================
def evaluate(agent):
    """用单个环境（无并行、无渲染）评估当前策略（argmax 贪婪），返回平均回报"""
    eval_env = gym.make("LunarLander-v3")
    eval_env.reset(seed=SEED)
    total = 0.0
    for _ in range(EVAL_EPISODES):
        state, _ = eval_env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                logits, _ = agent.net(state_t)
                action = logits.argmax().item()   # 评估用贪婪决策，表现更稳定
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total += reward
    eval_env.close()
    return total / EVAL_EPISODES

# =================== 训练循环 ===================
agent = PPOAgent()
print(f"===== PPO 开始训练（LunarLander-v3, N={N_ENVS}, T={ROLLOUT_STEPS}, EPOCHS={EPOCHS}）=====")
envs.reset(seed=SEED)

states, _ = envs.reset()   # 初始状态 [N, 8]

for update in range(UPDATES):
    # ---------- 1. 采样一批数据（额外存 log_probs_old，这是 PPO 的关键） ----------
    batch_states, batch_actions, batch_log_probs = [], [], []
    batch_rewards, batch_values, batch_dones = [], [], []

    for _ in range(ROLLOUT_STEPS):
        actions, log_probs, _ = agent.select_actions(states)      # log_probs = log π_old
        with torch.no_grad():
            _, values = agent.net(torch.FloatTensor(states).to(DEVICE))

        next_states, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
        dones = terminated | truncated

        batch_states.append(states)
        batch_actions.append(actions)
        batch_log_probs.append(log_probs)     # 存旧策略的对数概率（PPO 专用）
        batch_rewards.append(rewards)
        batch_values.append(values)
        batch_dones.append(dones)

        states = next_states   # 结束的环境由 vector env 自动 reset

    last_dones = dones

    # ---------- 2. 算 GAE + returns（与 A2C 相同） ----------
    rewards = torch.stack([torch.FloatTensor(r).to(DEVICE) for r in batch_rewards])  # [T, N]
    values  = torch.stack(batch_values)                                              # [T, N]
    dones   = torch.stack([torch.FloatTensor(d).to(DEVICE) for d in batch_dones])    # [T, N]
    with torch.no_grad():
        _, last_values = agent.net(torch.FloatTensor(states).to(DEVICE))             # [N]
    last_dones_t = torch.FloatTensor(last_dones).to(DEVICE)                          # [N]

    advantages, returns = compute_gae(rewards, values, dones, last_values,
                                      last_dones_t, GAMMA, LAM)                      # [T, N]
    # 优势标准化（笔记 2.9）：稳定梯度尺度
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ---------- 3. PPO 更新（多 epoch × mini-batch + clip） ----------
    agent.update(batch_states, batch_actions, batch_log_probs, advantages, returns)

    # ---------- 4. 定期评估 ----------
    if (update + 1) % EVAL_INTERVAL == 0:
        avg_reward = evaluate(agent)
        print(f"更新 {update+1:5d} | 评估平均回报: {avg_reward:7.2f} （≥200 可认为学会）")

print("===== 训练完成 =====")

#===================保存模型 ===================
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_lunarlander.pth")
torch.save({
    'model_state_dict': agent.net.state_dict(),
    # state_dim/action_dim 需 int() 转成 Python 整数，否则 PyTorch 2.6+ weights_only 报错
    'hyperparams': {'state_dim': int(state_dim), 'action_dim': int(action_dim), 'HIDDEN': 256},
}, SAVE_PATH)
print(f"模型已保存为 {SAVE_PATH}")

envs.close()


# ============================================================================
# 【理解笔记】PPO 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体与环境（与 A2C 完全相同）
#   agent = PPOAgent()：共享网络（actor 头 + critic 头）+ 优化器
#   envs  = SyncVectorEnv(...)：并行 N 个 LunarLander
#
# 第二阶段：更新循环（外层 for update in range(UPDATES)）
#   1. 采样一批数据（内层 for _ in range(ROLLOUT_STEPS)）：
#      - 批量采样动作，【额外存 log_probs_old】← PPO 和 A2C 的第一个区别
#      - critic 批量估计 V(s)，envs.step 批量推进，记录 reward/value/done
#   2. 算 GAE 优势 + 优势标准化（与 A2C 相同）
#   3. PPO 更新（agent.update）：
#      a. 同一批数据做 EPOCHS 个 epoch，每个 epoch 打乱成若干 mini-batch
#      b. 每个 mini-batch 用【当前】θ 重算 log π_θ，得概率比
#         r_t(θ) = exp(log π_θ - log π_old)
#      c. clip 目标：loss_actor = -min(r·A, clip(r,1±ε)·A).mean()
#      d. 加上 critic 损失与熵正则，反向传播更新
#   4. 定期用单环境评估（argmax 贪婪）并打印平均回报
#
# 【与 A2C 的三个区别】（其余全部相同）
#   1. 采样时多存一份 log_probs_old
#   2. 同一批数据用 EPOCHS×mini-batch 次（而不是 1 次）——样本效率↑
#   3. 损失用 clip 目标（而不是 -log π·A）——更新幅度被夹住，训练稳定↑
# ============================================================================
