# ============================================================
# Double Dueling DQN (Dueling + Double) 训练代码 —— 第二阶段改进
# 相对 DDQN 的唯一改动：QNetwork 结构改为 Dueling 架构
#   Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
#   - 共享特征层 → 价值分支 V(s)：状态本身有多好（与动作无关）
#   -             优势分支 A(s,a)：该动作相对平均水平的优势
#   - 网络输出接口不变（仍是每个动作的 Q 值），因此 select_action / update
#     中的 Double 逻辑、经验池、ε贪婪、目标网络同步、训练循环全部不用动
# ============================================================
import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from set_seed import set_seed

# =================== 环境配置 ===================
env = gym.make("Acrobot-v1")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# =================== 超参数 ===================
EPISODES        = 1500
BUFFER_SIZE     = 10000
BATCH_SIZE      = 128
GAMMA           = 0.99
LR              = 1e-3
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.995
TARGET_UPDATE   = 10
SEED            = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

# =================== Q 网络定义（Dueling 结构，本阶段唯一改动） ===================
class QNetwork(nn.Module):
    """Dueling 网络：把 Q 值拆成 状态价值 V(s) + 动作优势 A(s,a)"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享特征层：先把状态压成特征向量（结构与基准前两层一致）
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # 价值分支：输出标量 V(s)，衡量"这个状态本身有多好"（与动作无关）
        self.value_stream = nn.Linear(128, 1)
        # 优势分支：输出每个动作的 A(s,a)，衡量"该动作相对平均水平的优势"
        self.advantage_stream = nn.Linear(128, action_dim)

    def forward(self, x):
        features = self.feature(x)               # [B, 128] 共享特征
        V = self.value_stream(features)          # [B, 1]  状态价值
        A = self.advantage_stream(features)      # [B, action_dim] 动作优势
        # 关键组合：Q = V + A - mean(A)
        # 减均值是为了"可辨识性"：若 Q = V + A，则 V 加任意常数 c、A 减 c 后 Q 不变，
        # 网络无法唯一确定 V 和 A；减去优势均值后强制 mean(A)≈0，分解才唯一、训练才稳定
        Q = V + A - A.mean(dim=1, keepdim=True)  # [B, action_dim]
        return Q

# =================== 经验回放池 ===================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# =================== DQN 智能体 ===================
class DQNAgent:
    def __init__(self):
        self.q_net   = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer  = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state, eval_mode=False):
        if eval_mode:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return self.q_net(state_t).argmax().item()
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return self.q_net(state_t).argmax().item()

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # ---- 当前 Q 值 (选用实际执行的动作) ----
        q_values = self.q_net(states).gather(1, actions)

        # ---- 目标 Q 值 (沿用 Double：当前网络选动作，目标网络评价值) ----
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q   = self.target_net(next_states).gather(1, best_actions)
            q_target = rewards + GAMMA * max_next_q * (1 - dones)

        # ---- 均方误差损失 + 反向传播 ----
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- 探索率衰减 ----
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def sync_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# =================== 训练循环 ===================
agent = DQNAgent()
episode_rewards = []

print("===== Double Dueling DQN 开始训练 =====")
env.reset(seed=SEED)
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    if ep % TARGET_UPDATE == 0:
        agent.sync_target_net()

    episode_rewards.append(total_reward)

    if (ep + 1) % 50 == 0:
        avg = np.mean(episode_rewards[-50:])
        print(f"回合 {ep+1:4d} | 平均奖励(近50回合): {avg:.2f} | epsilon: {agent.epsilon:.3f}")
print("===== 训练完成 =====")

# =================== 保存模型 ===================
torch.save(agent.q_net.state_dict(), "dqn_Acrobot_duelingddqn.pth")
print("模型已保存为 dqn_Acrobot_duelingddqn.pth")

# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Double Dueling DQN Training Curve")
plt.show()
env.close()
