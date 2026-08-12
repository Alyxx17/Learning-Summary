# ============================================================
# N-step DQN 训练代码 —— 第三阶段改进（在 Double + Dueling 之上）
# 唯一改动：把"1 步目标值"升级为"N 步目标值"
#   target = G_t + γ^N * max Q(s_{t+N}, a*)      （G_t 是 N 步累计折扣回报）
#   涉及 3 处：
#   ① 超参数新增 N_STEP = 3
#   ② ReplayBuffer 改为"滚动窗口攒 N 步"再产出经验（存的是 N 步转换）
#   ③ update() 里折扣因子 GAMMA 换成 GAMMA ** N_STEP
#   其余（Dueling 网络、Double 目标值、ε贪婪、目标网络同步、训练循环骨架）不变
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
env = gym.make("CartPole-v1")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# =================== 超参数 ===================
EPISODES        = 600
BUFFER_SIZE     = 10000
BATCH_SIZE      = 128
GAMMA           = 0.99
N_STEP          = 3             # 新增：N 步学习，目标值看未来 N 步的真实回报
LR              = 1e-3
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.995
TARGET_UPDATE   = 10
SEED            = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

# =================== Q 网络定义（Dueling，与上一阶段相同） ===================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_dim)

    def forward(self, x):
        features = self.feature(x)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q

# =================== 经验回放池（N 步版本，本阶段核心改动） ===================
class ReplayBuffer:
    """存储 N 步转换 (s, a, G, s_N, done)，G 是 N 步累计折扣回报"""
    def __init__(self, capacity, n_step=N_STEP, gamma=GAMMA):
        self.buffer = deque(maxlen=capacity)#经验池的队列
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque()   # 存n步队列：暂存最近 N 步，攒满才产出
        #与经验池的队列独立，是用于存n步的队列

    def push(self, state, action, reward, next_state, terminated):
        # 先把当前一步放入存n步队列（terminated 表示真正的回合结束）
        self.n_step_buffer.append((state, action, reward, next_state, terminated))

        if terminated:
            # 回合真正结束：把窗口里剩余的经验补成一条（可能不足 N 步），并清空窗口
            # G = r_0 + γr_1 + ... + γ^{len-1} r_{len-1}（len 为窗口内实际步数）
            G = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(len(self.n_step_buffer)))#实际步数的奖励
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            self.buffer.append((s0, a0, G, next_state, True))   # done=True：未来价值清零
            self.n_step_buffer.clear()
        elif len(self.n_step_buffer) == self.n_step:
            # 攒满 N 步：生成一条 N 步转换，窗口滑动一步
            # G = r_0 + γr_1 + ... + γ^{N-1} r_{N-1}；s_N 是第 N 步到达的状态
            G = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(self.n_step))#n步的奖励
            #从存n步队列的最旧一步出发，连续 N 步的折扣奖励之和
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            #最旧一步的s,a
            sN = self.n_step_buffer[-1][3]
            self.buffer.append((s0, a0, G, sN, False))
            #存n步队列的最旧一步的s,a,G,next_state(即sN，是存n步队列最新的一步内的nextstate)
            #使其进入经验池
            self.n_step_buffer.popleft()#滑动
        # 既没结束也没攒满：继续等，不产出

    def reset_n_step(self):
        """回合结束（terminated 或 truncated）后清空窗口，避免跨回合拼接"""
        self.n_step_buffer.clear()

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

        # ---- 目标 Q 值 (N 步 + Double：当前网络选动作，目标网络评价值) ----
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q   = self.target_net(next_states).gather(1, best_actions)
            # N 步目标：G + γ^N * max Q(s_{t+N}, a*)
            # rewards 已经是 N 步累计回报 G，所以折扣从 GAMMA 换成 GAMMA ** N_STEP
            q_target = rewards + (GAMMA ** N_STEP) * max_next_q * (1 - dones)

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

print("===== N-step DQN 开始训练 =====")
env.reset(seed=SEED)
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # 关键：传给 buffer 的是 terminated（真正结束）而不是合并后的 done
        # 因为 truncated（超时）不是真终止，目标里的未来价值不应被清零
        #terminated:杆子倾角超过阈值，或者小车滑出边界
        #truncated：不满足上述的条件下，称到第500步
        agent.store_transition(state, action, reward, next_state, terminated)
        if truncated:
            agent.buffer.reset_n_step()   # 超时截断：环境回合已结束，清空窗口避免跨回合拼接
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
torch.save(agent.q_net.state_dict(), "dqn_cartpole_nstep.pth")
print("模型已保存为 dqn_cartpole_nstep.pth")

# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("N-step DQN Training Curve")
plt.show()
env.close()
