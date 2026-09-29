# =================== SAC 智能体（连续动作版） ===================
# 移植自 RL_py/SAC/SAC_train.py，结构保持一致（你熟悉它的写法）:
#   双 Q + 目标网络软更新 + 重参数化 tanh 策略 + 自动温度 α + numpy 环形经验池
# 本项目差异:
#   · state_dim = 5（误差 3 + 参考输入 2）、action_dim = 2（权重倍数 q、r 的"旋钮"）
#   · 动作不再乘 ACTION_SCALE：MPCEnv 内部自己解码成 (q, r)
#   · 经验池存【归一化动作】∈[-1,1]（与 SAC_train.py 的约定一致）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =================== 超参数（默认沿用 SAC_train.py） ===================
HIDDEN      = 256        # 隐层宽度
GAMMA       = 0.99       # 折扣因子
TAU         = 0.005      # 目标网络软更新系数（每步 θ' ← τθ + (1-τ)θ'）
LR          = 1e-4       # actor 与双 Q 的学习率
LR_ALPHA    = 3e-4       # 温度 α 的学习率
BATCH_SIZE  = 256        # 小批量大小
BUFFER_SIZE = 100000     # 经验池容量（off-policy 要存很多旧数据）
LOG_STD_MIN = -20        # log σ 下限（防方差坍缩到 0）
LOG_STD_MAX = 2          # log σ 上限（防方差爆炸）—— log σ 限幅是 SAC 稳定的关键小技巧

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================== 策略网络（Actor）===================
class ActorNet(nn.Module):
    """输出高斯分布的两个参数: μ(s) 与 log σ(s)（连续动作版策略）"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(HIDDEN, action_dim)   # → μ
        self.log_std_head = nn.Linear(HIDDEN, action_dim)   # → log σ

    def forward(self, x):
        features = self.fc(x)
        mean    = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)   # 限幅
        return mean, log_std

    def sample(self, x):
        """采样动作 + 计算 log π(a|s)
        ① 重参数化采样: u = μ + σ·ξ（rsample 才能让梯度穿过采样回传）
        ② tanh 压缩: a = tanh(u) ∈ (-1,1)
        ③ log π 修正（tanh 换元，SAC 论文附录 C）: log π = log N(u) - Σlog(1-tanh²(u))
        返回: (a [batch, action_dim] 归一化动作, log π [batch, 1])
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)   # 1e-6 防 log(0)
        log_prob = log_prob.sum(dim=-1, keepdim=True)   # 各维求和 → [batch, 1]
        return action, log_prob


# =================== 双 Q 网络（Critic）===================
class QNet(nn.Module):
    """Q(s,a): 状态与动作拼接后输入；双 Q 取 min 抑制自举高估（同 TD3 思路）"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)                       # [batch, 1]


# =================== 经验回放池 ===================
class ReplayBuffer:
    """预分配 numpy 数组 + 环形缓冲（比 deque 随机访问快，可整批向量化取出）"""

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr  = 0      # 下一个写入位置（环形指针）
        self.size = 0      # 当前已存条数
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done
        self.ptr = (self.ptr + 1) % self.capacity      # 环形前进，存满覆盖最旧
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """随机抽 batch_size 条（一次性向量化取出）"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size


# =================== SAC 智能体 ===================
class SACAgent:
    """三张要训练的网络（actor、q1、q2）+ 两张目标网络 + 自动温度 α + 经验池"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ---- 要训练的网络 ----
        self.actor = ActorNet(state_dim, action_dim).to(DEVICE)
        self.q1    = QNet(state_dim, action_dim).to(DEVICE)
        self.q2    = QNet(state_dim, action_dim).to(DEVICE)
        # ---- 目标网络（只用来算目标 y，软更新慢慢跟随）----
        self.q1_target = QNet(state_dim, action_dim).to(DEVICE)
        self.q2_target = QNet(state_dim, action_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.q_optimizer     = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)

        # ---- 自动温度 α（SAC v2）: 用 log α 作可训练参数（α = exp(log α) 恒正）----
        self.target_entropy = -float(action_dim)   # 目标熵 H̄ = -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)   # 初始 α = 1
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)

    @property
    def alpha(self):
        """温度系数 α（恒正）"""
        return self.log_alpha.exp()

    def select_action(self, state, eval_mode=False):
        """选择动作（返回归一化动作 ∈[-1,1]，直接喂 MPCEnv）
        eval_mode=True 时用确定性动作 tanh(μ)（评估用）
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)   # (5,) → [1, 5]
        with torch.no_grad():
            if eval_mode:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy()     # [1, 2] → (2,)

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        """一步学习: 采样 mini-batch，依次更新 双 Q → actor → α → 软更新目标网络
        Q 目标   : y = r + γ(1-done)·[ min(Q1',Q2')(s',a') - α·log π(a'|s') ]
        actor 目标: min E[ α·log π(a|s) - min(Q1,Q2)(s,a) ]（a 为重参数化采样）
        α 目标    : 让策略随机程度逼近目标熵 H̄ = -dim(A)
        """
        if len(self.buffer) < BATCH_SIZE:
            return   # 池子不够批大小: 先不学

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)
        actions     = torch.FloatTensor(actions).to(DEVICE)
        rewards     = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones       = torch.FloatTensor(dones).to(DEVICE)

        # ============ ① 更新双 Q ============
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_probs   # 熵项
            backup  = rewards + GAMMA * (1.0 - dones) * q_next
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ============ ② 更新策略网络 ============
        pi, log_pi = self.actor.sample(states)
        q_pi_min = torch.min(self.q1(states, pi), self.q2(states, pi))
        actor_loss = (self.alpha.detach() * log_pi - q_pi_min).mean()
        # α.detach(): α 在策略损失里视为常数，防止梯度串扰
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============ ③ 自动调节温度 α ============
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ============ ④ 软更新目标网络 ============
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)

    def save(self, path):
        """保存 actor 与必要超参（第 4 步评估脚本直接加载）"""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            # state_dim/action_dim 用 Python int（避免 torch.load 默认 weights_only
            # 对 numpy 标量的不兼容——SAC_train.py 里踩过的坑）
            "hyperparams": {"state_dim": int(self.state_dim),
                            "action_dim": int(self.action_dim),
                            "HIDDEN": HIDDEN},
        }, path)


def load_actor(path, device=DEVICE):
    """加载训练好的 actor（第 4 步评估用）
    返回: (actor: ActorNet, hyperparams: dict)"""
    ckpt = torch.load(path, map_location=device)
    hp = ckpt["hyperparams"]
    actor = ActorNet(int(hp["state_dim"]), int(hp["action_dim"])).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor, hp
