# ============================================================
# Double DQN (DDQN) 训练代码 —— 在基准 DQN 上的第一个改进
# 唯一改动：update() 里目标 Q 值的计算方式（解除"选动作"与"评估价值"的耦合）
# 其余部分（网络、经验池、ε贪婪、目标网络同步、训练循环）与基准 DQN_train.py 完全一致
# ============================================================
import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from set_seed import set_seed   # 从 set_seed 模块导入 set_seed 函数（直接调用函数，而不是调用模块）
# =================== 环境配置 ===================
env = gym.make("CartPole-v1")         # 经典平衡车环境
state_dim  = env.observation_space.shape[0]  # 状态维度：4 (位置,速度,角度,角速度)
action_dim = env.action_space.n              # 动作维度：2 (左/右)

# =================== 超参数（与基准 DQN 完全一致） ===================
EPISODES        = 600           # 训练回合数
BUFFER_SIZE     = 10000         # 经验池容量
BATCH_SIZE      = 128           # 小批量大小
GAMMA           = 0.99          # 折扣因子
LR              = 1e-3          # 学习率
EPSILON_START   = 1.0           # 探索率起始值
EPSILON_END     = 0.01          # 探索率最小值
EPSILON_DECAY   = 0.995         # 探索率衰减
TARGET_UPDATE   = 10            # 目标网络更新间隔 (回合数)
SEED            = 42            # 随机种子（固定训练过程，保证可复现）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)   #全局随机种子

# =================== Q 网络定义（与基准完全一致） ===================
class QNetwork(nn.Module):
    """简单的全连接网络，输入状态，输出各动作的 Q 值"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)           # 输出形状: [batch, action_dim]

# =================== 经验回放池（与基准完全一致） ===================
class ReplayBuffer:
    """存储 (s, a, r, s', done) 并支持随机采样"""
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

# =================== DDQN 智能体 ===================
class DQNAgent:
    def __init__(self):
        self.q_net   = QNetwork(state_dim, action_dim).to(DEVICE)       # 当前 Q 网络
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)    # 目标 Q 网络
        self.target_net.load_state_dict(self.q_net.state_dict())        # 初始化时同步参数
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer  = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state, eval_mode=False):
        """ε-贪婪策略选择动作。eval_mode=True 时关闭探索 (仅测试用)"""
        if eval_mode:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return self.q_net(state_t).argmax().item()
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()                        # 随机探索
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                return self.q_net(state_t).argmax().item()          # 贪婪选择

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        """从经验池采样并更新 Q 网络"""
        if len(self.buffer) < BATCH_SIZE:
            return

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)   # [B, 1]
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)  # [B, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)    # [B, 1]

        # ---- 当前 Q 值 (选用实际执行的动作) ----
        q_values = self.q_net(states).gather(1, actions)                  # [B, 1]

        # ---- 目标 Q 值 (Double DQN：当前网络选动作，目标网络评估价值) ----
        with torch.no_grad():
            # 基准 DQN：target_net 自己选自己评 → max 高估
            #   max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            # DDQN：q_net 选动作，target_net 评价值 → 解除耦合、降低高估
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)  # 当前网络选择最优动作
            max_next_q   = self.target_net(next_states).gather(1, best_actions) # 目标网络评估该动作的价值
            q_target = rewards + GAMMA * max_next_q * (1 - dones)               # [B, 1]

        # ---- 均方误差损失 + 反向传播 ----
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，提升稳定性 (可选)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- 探索率衰减 ----
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def sync_target_net(self):
        """将当前 Q 网络的权重复制到目标网络 (硬更新)"""
        self.target_net.load_state_dict(self.q_net.state_dict())

# =================== 训练循环（与基准完全一致） ===================
agent = DQNAgent()
episode_rewards = []           # 记录每个回合的总奖励 (用于绘图)

print("===== Double DQN 开始训练 =====")
env.reset(seed=SEED)        # 给环境的随机数发生器播种（只播种一次）
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # 1. 选择动作
        action = agent.select_action(state)

        # 2. 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 3. 存入经验池
        agent.store_transition(state, action, reward, next_state, done)

        # 4. 更新网络 (在线更新，每步都学)
        agent.update()

        state = next_state
        total_reward += reward

    # 每隔一定回合同步目标网络
    if ep % TARGET_UPDATE == 0:
        agent.sync_target_net()

    episode_rewards.append(total_reward)

    # 打印训练进度
    if (ep + 1) % 50 == 0:
        avg = np.mean(episode_rewards[-50:])
        print(f"回合 {ep+1:4d} | 平均奖励(近50回合): {avg:.2f} | epsilon: {agent.epsilon:.3f}")
print("===== 训练完成 =====")

#===================保存模型 ===================
torch.save(agent.q_net.state_dict(), "dqn_cartpole_ddqn.pth")
print("模型已保存为 dqn_cartpole_ddqn.pth")


# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Double DQN Training Curve")
plt.show()
env.close()
