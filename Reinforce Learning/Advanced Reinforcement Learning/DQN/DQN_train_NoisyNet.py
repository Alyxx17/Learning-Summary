# ============================================================
# Noisy Net DQN 训练代码 —— 第四阶段改进（在 Double + Dueling + N-step 之上）
# 核心改动：用"参数噪声"取代 ε-贪婪探索
#   ① 新增 NoisyLinear 层：y = (μ_W + σ_W⊙ε_W)x + (μ_b + σ_b⊙ε_b)
#   ② 网络所有线性层换成 NoisyLinear
#   ③ 删除 ε 相关的一切（超参数、ε-贪婪分支、ε 衰减）
#   其余（Dueling 结构、Double 目标值、N 步回放、目标网络同步、训练循环骨架）不变
# ============================================================
import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from set_seed import set_seed

# =================== 环境配置 ===================
env = gym.make("LunarLander-v3",continuous=False)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# =================== 超参数（已删除所有 EPSILON_*） ===================
EPISODES        = 600
BUFFER_SIZE     = 50000
BATCH_SIZE      = 128
GAMMA           = 0.99
N_STEP          = 3
LR              = 1e-3
TARGET_UPDATE   = 10
SEED            = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

# =================== Noisy 线性层（本阶段核心改动 1） ===================
class NoisyLinear(nn.Module):
    """带可学习噪声的线性层：y = (μ_W + σ_W⊙ε_W)x + (μ_b + σ_b⊙ε_b)
    训练时：每个前向都采样噪声 ε，等价于每次用"带噪声的权重"计算
    测试时：不采样噪声，只用均值 μ（确定性推理）"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #y = (μ_W + σ_W⊙ε_W)x + (μ_b + σ_b⊙ε_b)
        #μ_W ，σ_W，ε_W这三个分别是权重均值，权重方差，权重噪声，相当于原来的W，因此都为2维矩阵
        #后面3个都是偏置，因此都是一维向量
        #torch.randn：不仅申请内存，还会花时间把每个位置填上符合正态分布的随机数（计算量大）。
        #torch.empty：只申请内存，绝不花一秒钟去清理或赋值。
        # 内存里原来残留着什么数字（可能是 0、极大极小值、或者 NaN），它就原封不动地保留。
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))#μ_W
        self.bias_mu   = nn.Parameter(torch.empty(out_features))#μ_b
        # 可学习参数：噪声尺度 σ（网络自己学"每个权重该加多少噪声" = 探索强度）
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))#σ_W
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))#σ_b
        #权重是（输入维数，输出维数）的矩阵，而偏置是x*W^T算完后，经过权重矩阵后加上的，因此只是输出维数
        #权重参数中，输出在前，是因为Pytorch中定义权重矩阵是W^T，有一个转置
        self.reset_parameters()

    def reset_parameters(self):
        # 均值用均匀初始化（与 nn.Linear 类似，只是类似）可以参考何凯明初始化等内容，用于防止梯度爆炸
        std = 1.0 / (self.in_features ** 0.5)
        #uniform是均匀分布，即在均值参数内随机均匀填入区间的数字
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)
        # 噪声尺度初始化为小常数（探索强度起步适中）#fill是填入固定的数字，填入固定数（0.5*std)
        self.weight_sigma.data.fill_(0.5 * std)
        self.bias_sigma.data.fill_(0.5 * std)
        #均值要随机，打破对称性，否则反向传播回来时算出来梯度一样
        #噪声固定无所谓，因为还要乘ε

    def forward(self, x):
        if self.training:#self.training继承nn.Module，由model.eval()控制关闭。
            # 训练：采样标准正态噪声ε ~ N(0, 1)，叠加到均值上（ε 不参与梯度，但 σ 会）
            noise_w = torch.randn_like(self.weight_mu)
            noise_b = torch.randn_like(self.bias_mu)
            #randn（n 代表 normal），而不是 rand（均匀分布）
            weight = self.weight_mu + self.weight_sigma * noise_w
            bias   = self.bias_mu   + self.bias_sigma   * noise_b
        else:
            # 测试：只用均值（去掉噪声，确定性决策）
            weight = self.weight_mu
            bias   = self.bias_mu
            #self.weight_mu 和self.bias_mu 就是原始DQN的W和b
        return F.linear(x, weight, bias)
    #当 loss.backward() 执行时，PyTorch 沿着计算图反推。对于这一行：
    #weight = weight_mu + weight_sigma * noise_w
    #因为 noise_w 是固定的常量（在这一次前向中），weight_sigma 乘在它前面。
    #反向传播会算出：∂Loss / ∂weight_sigma = (∂Loss / ∂weight) × noise_w
    #也就是说，noise_w 虽然没梯度，但它作为 “见证者”，记录了这次抖动的方向和大小。
    # 如果这次抖动（noise_w 为正）恰好让损失变小了，
    # 梯度就会告诉优化器 “把 σ 调大，下次多抖点”；反之则调小。

# =================== Q 网络定义（Dueling + Noisy，本阶段核心改动 2） ===================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            NoisyLinear(state_dim, 256),
            nn.ReLU(),
            NoisyLinear(256, 256),
            nn.ReLU()
        )
        self.value_stream = NoisyLinear(256, 1)
        self.advantage_stream = NoisyLinear(256, action_dim)

    def forward(self, x):
        features = self.feature(x)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q

# =================== 经验回放池（N 步，与上一阶段相同） ===================
class ReplayBuffer:
    """存储 N 步转换 (s, a, G, s_N, done)，G 是 N 步累计折扣回报"""
    def __init__(self, capacity, n_step=N_STEP, gamma=GAMMA):
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque()   # 滚动窗口：暂存最近 N 步，攒满才产出

    def push(self, state, action, reward, next_state, terminated):
        self.n_step_buffer.append((state, action, reward, next_state, terminated))

        if terminated:
            G = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(len(self.n_step_buffer)))
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            self.buffer.append((s0, a0, G, next_state, True))
            self.n_step_buffer.clear()
        elif len(self.n_step_buffer) == self.n_step:
            G = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(self.n_step))
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            sN = self.n_step_buffer[-1][3]
            self.buffer.append((s0, a0, G, sN, False))
            self.n_step_buffer.popleft()

    def reset_n_step(self):
        self.n_step_buffer.clear()

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# =================== DQN 智能体（已删除 ε 相关） ===================
class DQNAgent:
    def __init__(self):
        self.q_net   = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer  = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state):
        # 不再有 ε-贪婪：探索由网络内部的参数噪声完成（训练时 forward 自动采样）
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
            q_target = rewards + (GAMMA ** N_STEP) * max_next_q * (1 - dones)

        # ---- 均方误差损失 + 反向传播 ----
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        # （无 ε 衰减：探索强度由网络自己学）

    def sync_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# =================== 训练循环 ===================
agent = DQNAgent()
episode_rewards = []

print("===== Noisy Net DQN 开始训练 =====")
env.reset(seed=SEED)
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(state, action, reward, next_state, terminated)
        if truncated:
            agent.buffer.reset_n_step()
        agent.update()
        state = next_state
        total_reward += reward

    if ep % TARGET_UPDATE == 0:
        agent.sync_target_net()

    episode_rewards.append(total_reward)

    if (ep + 1) % 50 == 0:
        avg = np.mean(episode_rewards[-50:])
        print(f"回合 {ep+1:4d} | 平均奖励(近50回合): {avg:.2f}")
print("===== 训练完成 =====")

# =================== 保存模型 ===================
torch.save(agent.q_net.state_dict(), "dqn_LunarLander_noisynet.pth")
print("模型已保存为 dqn_LunarLander_noisynet.pth")

# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Noisy Net DQN Training Curve")
plt.show()
env.close()
