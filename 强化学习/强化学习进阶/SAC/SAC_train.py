# ============================================================
# SAC（Soft Actor-Critic）训练代码 —— Pendulum-v1（连续控制）
# SAC = 三者的融合体：【连续动作版 Actor-Critic + off-policy 经验回放 + 最大熵】
#   1. 像 A2C/PPO 一样同时学"策略 + 价值"，但策略输出【连续高斯分布】
#      （A2C/PPO 是离散 Categorical，这里换成 Normal + tanh 压缩）
#   2. 像 DQN 一样是 【off-policy】：经验回放池 + 目标网络，每步都能学、旧数据反复用
#   3. 把 A2C/PPO 的"熵正则项"升级成【最大熵目标】：Q 目标与策略损失里
#      处处带 -α·log π(a|s)，且温度 α 【自动调节】（SAC v2 的关键改进）
#
# 五个核心组件（看代码时逐个对照）：
#   ① 双 Q 网络（Twin Q）：目标里取 min(Q1', Q2')，抑制自举带来的 Q 值高估（借鉴 TD3）
#   ② 目标网络【软更新】：每步 θ' ← τθ + (1-τ)θ'（对比 DQN 每 10 回合的硬拷贝）
#   ③ 重参数化采样：a = tanh(μ(s) + σ(s)·ξ)，ξ~N(0,I)，让梯度穿过"采样"回传
#   ④ tanh 压缩：无界高斯 → 归一化动作 a ∈ [-1,1]（送环境前乘 2 → 真实扭矩）
#   ⑤ 自动温度 α：α = exp(log_alpha) 恒正，用梯度下降让策略的随机程度逼近目标
#
# 环境说明：Pendulum 状态 3 维 (cosθ, sinθ, 角速度)，动作 1 维连续扭矩 ∈ [-2,2]，
#   每回合固定 200 步；回报范围约 [-1200（乱转）, -150（稳定在竖直附近）]。
# 结尾有整体框架的运行流程
# ============================================================

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from set_seed import set_seed   # 从 set_seed 模块导入 set_seed 函数

# =================== 环境配置 ===================
env = gym.make("Pendulum-v1")     # 倒立摆：把杆从任意角度摆到竖直向上并稳住
state_dim  = env.observation_space.shape[0]     # 3 (cosθ, sinθ, 角速度)
action_dim = env.action_space.shape[0]          # 1（连续动作用 .shape[0]，离散用 .n）
ACTION_SCALE = float(env.action_space.high[0])  # 动作上限 2.0（真实扭矩 ∈ [-2,2]）

# 全代码统一约定：
#   策略网络输出的"归一化动作" ∈ [-1,1]（tanh 压缩的结果）；
#   送进 env.step 前乘 ACTION_SCALE 还原成真实扭矩；
#   经验池与 Q 网络统一使用【归一化动作】——这样 log π 就是论文里 [-1,1] 动作域上的密度。

# =================== 超参数 ===================
EPISODES      = 100       # 训练回合数（Pendulum 每回合固定 200 步 → 共 20 万步；想先快速看趋势可减到 300）
BUFFER_SIZE   = 100000     # 经验池容量（off-policy 要存很多旧数据）
BATCH_SIZE    = 256        # 小批量大小（每步从池中采一批来学）
GAMMA         = 0.99       # 折扣因子
TAU           = 0.005      # 目标网络软更新系数（每步 θ' ← τθ + (1-τ)θ'）
LR            = 3e-4       # actor 与双 Q 的学习率
LR_ALPHA      = 3e-4       # 温度 α 的学习率
START_STEPS   = 10000      # 前 1 万步用均匀随机动作（预热经验池，类似 DQN 一开始 ε=1 的纯探索阶段）
LOG_STD_MIN   = -20        # log σ 下限（防方差坍缩到 0）
LOG_STD_MAX   = 2          # log σ 上限（防方差爆炸）—— log σ 限幅是 SAC 稳定的关键小技巧
EVAL_INTERVAL = 100        # 每多少回合评估一次
EVAL_EPISODES = 5          # 每次评估跑几局（确定性动作）
LOG_INTERVAL  = 20         # 每多少回合打印一次训练信息
SEED          = 42         # 随机种子（固定训练过程，保证可复现）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =================== 策略网络（Actor）===================
class ActorNet(nn.Module):
    """连续动作版策略网络：不再输出"每个动作的分数"（logits），
    而是输出【高斯分布的两个参数】：
      - μ(s)     ：动作均值，形状 [batch, action_dim]
      - log σ(s) ：动作标准差的对数（用"对数"是工程习惯：σ = exp(log σ) 恒正，好优化）
    对比 A2C/PPO：那边是 Categorical(logits)（离散概率）→ 这里是 Normal(μ, σ)（连续分布）
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, action_dim)   # → μ
        self.log_std_head = nn.Linear(256, action_dim)   # → log σ

    def forward(self, x):
        features = self.fc(x)
        mean    = self.mean_head(features)
        log_std = self.log_std_head(features)
        #对数标准差，作用是：1. 保证标准差 σ 恒正（σ = exp(log σ)，即使对数标准差为负数，指数也可以为正）；2. log σ 的梯度更平滑，便于优化
        # 限幅：把 log σ 夹在 [LOG_STD_MIN, LOG_STD_MAX]，防止训练早期方差过大或坍缩到 0
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    #SAC 的 Actor 就是“输出高斯参数 → 重参数化采样 → tanh 压缩 → 缩放平移 → 得到真实动作”
    #关于为什么修正项 log π(a|s) 要减去 Σ log(1 - tanh²(u))，
    # 简单来说，tanh 是一个非线性变换，它会改变概率密度的分布，因此需要对 log π(a|s) 进行修正，以确保策略梯度的正确性。
    def sample(self, x):#输入状态 x，输出归一化动作 a 和对应的 log π(a|s)
        """采样动作 + 计算 log π(a|s)：
        ① 重参数化采样：u = μ + σ·ξ，ξ~N(0,I)（用 rsample() 才能让梯度穿过采样回传）
        ② tanh 压缩：a = tanh(u) ∈ (-1,1)，得到归一化动作
        ③ log π 修正（tanh 变换的概率密度换元，SAC 论文附录 C）：
             log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))
        返回：动作 a [batch, action_dim]（归一化）、log π(a|s) [batch, 1]
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)# 定义正态分布
        u = dist.rsample()                     # 重参数化采样（可导），u = μ + σ·ξ
        action = torch.tanh(u)                 # 压缩到 [-1,1]

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        # 1e-6 防止 tanh 饱和（|u| 大）时 log(0) = -inf
        log_prob = log_prob.sum(dim=-1, keepdim=True)   # 多维动作各维求和 → [batch, 1]
        #就是把每个动作维度的 log 概率相加，得到整个动作向量的 log 概率，总和为 log π(a|s)
        #总和不是1，是因为我们在计算 log π(a|s) 时，考虑的是整个动作向量的联合概率密度，而不是单个动作维度的概率密度。
        # 对于多维连续动作空间，联合概率密度是各个维度概率密度的乘积，因此在对数空间中，它们的 log 概率是相加的。
        #dim=-1 表示在最后一个维度上求和，也就是对动作维度求和，得到每个样本的 log π(a|s)。
        return action, log_prob#返回归一化动作和对应的 log π(a|s)

# =================== 双 Q 网络（Critic）===================
class QNet(nn.Module):
    """Q(s,a) 网络：把状态和动作【拼接】后输入（对比 DQN 的 Q 网络只输入状态）
    SAC 用两个结构相同、参数独立的 Q 网络（Q1、Q2），目标值取二者较小者，
    抑制自举带来的 Q 值高估（与 TD3 相同的思路，代价是计算量翻倍）
    """
    #这个网络的作用是：给定一个状态 s 和一个动作 a，
    # 输出该状态-动作对的 Q 值，即该动作在该状态下的预期回报。
    # 它通过将状态和动作拼接在一起作为输入，然后经过几层全连接网络，最终输出一个标量 Q 值。
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),   # 注意：输入维度 = 状态 + 动作
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),                        # 输出一个 Q 值（标量）
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)   
        #按列拼接状态和动作，得到一个新的张量 x，形状为 [batch, state_dim + action_dim]。
        return self.fc(x)                       # [batch, 1]

# =================== 经验回放池 ===================
class ReplayBuffer:
    """预分配 numpy 数组 + 环形缓冲（对比 DQN 里 deque 的写法）
    为什么换写法：SAC 池子更大（10 万）、采样更频繁（每步一次）。
    deque 按索引随机访问是 O(n)（要沿着链表走）；numpy 数组索引 O(1)，
    而且能一次向量化取出整批样本，效率高得多。
    """
    #该采样池与DQN的队列的区别在于，它使用了预分配的numpy数组和环形缓冲区的方式来存储经验数据。这样做的好处是：
    #1. 提高了采样效率：使用numpy数组可以一次性向量化地取出整批样本，而不是像deque那样按索引随机访问，效率更高。
    #2. 节省内存：预分配的numpy数组可以避免频繁的内存分配和释放，从而节省内存开销。
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.ptr  = 0      # 下一个写入位置（环形指针）
        self.size = 0      # 当前已存条数（未满之前 < capacity）
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)
        # rewards/dones 存成 [n,1] 是为了后面直接和 [B,1] 的 Q 值做运算，不用反复 unsqueeze

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done
        self.ptr = (self.ptr + 1) % self.capacity      # 环形前进，存满后覆盖最旧的
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
    def __init__(self):
        # ---- 三个"学生"网络（要被训练）----
        self.actor = ActorNet(state_dim, action_dim).to(DEVICE)   # 策略（决策 + 被更新）
        self.q1    = QNet(state_dim, action_dim).to(DEVICE)       # 双 Q 之一（被更新）
        self.q2    = QNet(state_dim, action_dim).to(DEVICE)       # 双 Q 之二（被更新）
        # ---- 两个"老师"目标网络（只用来算目标 y，不直接被训练；每步软更新慢慢跟住学生）----
        self.q1_target = QNet(state_dim, action_dim).to(DEVICE)
        self.q2_target = QNet(state_dim, action_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())   # 初始化时先同步一次
        self.q2_target.load_state_dict(self.q2.state_dict())   # （对比 DQN：同样的初始化思路）

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.q_optimizer     = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)

        # ---- 自动温度 α：用 log α 作为可训练参数（α = exp(log α) 恒正，不用管符号）---
        #温度系数是目标函数中熵前面的系数，用于平衡奖励和熵的权重。较高的温度系数会鼓励更多的探索，而较低的温度系数会使策略更确定性。
        self.target_entropy = -float(action_dim)   # 目标熵 H̄ = -dim(A)（SAC 论文的启发式默认值，这里 = -1）
        #SAC 论文作者通过大量实验发现，把目标熵设为 -动作维度，能让策略在“探索”和“利用”之间达到一个很好的平衡，不需要手动调参。
        #-1：可以把它理解为：每个动作维度，平均分到 1 个单位的熵

        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)   # 初始 α = exp(0) = 1
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)   # 新建一个容量 10 万的经验池

    @property#@property 装饰器把一个方法变成属性调用（不用加括号），方便外部访问
    def alpha(self):
        """温度系数 α（恒正）"""
        return self.log_alpha.exp()

    def select_action(self, state, eval_mode=False):
        """选择动作（返回【归一化动作】∈[-1,1]；乘 ACTION_SCALE 后才是真实扭矩）
        eval_mode=True 时用确定性动作 tanh(μ)（评估/测试用，表现更稳定）
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)   # (3,) → [1, 3]
        with torch.no_grad():
            if eval_mode:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)              # 确定性：直接用均值过 tanh
                #它直接把 mean 当作确定性动作 u
            else:
                action, _ = self.actor.sample(state_t) # 随机采样（训练时的探索来源）
                #“和环境交互、收集数据”，根本不需要梯度，因此不用rsample()，直接用 sample() 采样就行
        return action.squeeze(0).cpu().numpy()         # [1, 1] → (1,)（gymnasium 接受 numpy 数组）

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        """一次学习：从池中采 BATCH_SIZE 条经验，依次更新 Q1/Q2 → actor → α → 软更新目标网络
        三个目标（最大熵 off-policy）：
          Q 目标   ：y = r + γ(1-done)·[ min(Q1',Q2')(s',a') - α·log π(a'|s') ]
          actor 目标：min E[ α·log π(a|s) - min(Q1,Q2)(s,a) ]（a 由重参数化采样得到）
          α 目标   ：让策略随机程度逼近目标熵 H̄ = -dim(A)
        """
        if len(self.buffer) < BATCH_SIZE:
            return   # 池子不够 256 条：先不学（与 DQN 的同一个套路）

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)       # [B, 3]
        actions     = torch.FloatTensor(actions).to(DEVICE)      # [B, 1]（归一化动作）
        rewards     = torch.FloatTensor(rewards).to(DEVICE)      # [B, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)  # [B, 3]
        dones       = torch.FloatTensor(dones).to(DEVICE)        # [B, 1]

        # ============ ① 更新双 Q 网络 ============
        # 目标 y 里所有"估计部分"都用 no_grad 冻结（同 DQN 的目标网络思想）：
        #   next_a, next_log_pi ← 用【当前策略】在 s' 上采样（SAC 论文的做法，不是用目标策略）
        #   q_next = min(Q1', Q2')(s', next_a) - α·next_log_pi
        #   ↑ 那个 -α·log π 就是"熵项"：把未来的"探索红利"也算进价值里
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            backup  = rewards + GAMMA * (1.0 - dones) * q_next    # [B, 1]
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)  # 两条 MSE（同 DQN 的 MSELoss，回归目标是 y）
        self.q_optimizer.zero_grad()#清空上次的梯度
        q_loss.backward()#反向传播，自动算出 loss 对每个参数的偏导数
        self.q_optimizer.step()#更新权重

        # ============ ② 更新策略网络（actor）============
        # 目标：最大化 E[ min(Q1,Q2)(s,a) - α·log π(a|s) ]（既追求高 Q 值，又保持探索熵）
        # 写成最小化（优化器只会梯度下降）：min E[ α·log π(a|s) - min(Q1,Q2)(s,a) ]
        # 关键：a 是重参数化采样的，梯度可以从 Q 一路传回 actor 参数（这是 SAC 可训练的核心）
        pi, log_pi = self.actor.sample(states)
        q1_pi = self.q1(states, pi)
        q2_pi = self.q2(states, pi)
        q_pi_min = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - q_pi_min).mean()
        # α.detach()：α 在策略损失里视为常数（它由自己的优化器更新），detach 防止梯度串扰
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ============ ③ 自动调节温度 α（SAC v2）============
        # 目标：min E[ -α·(log π(a|s) + H̄) ]，H̄ = -dim(A)
        # 直觉：当前 log π + H̄ 为负（比目标更随机）→ 梯度下降会调小 α（少给熵加权）；
        #       为正（比目标更确定）→ 调大 α（多给熵加权、逼策略更随机）。α 自己找平衡点。
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        # log_pi 用 detach：α 的梯度不能被策略网络"借走"，log_alpha 是唯一被更新的参数
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ============ ④ 软更新目标网络 ============
        # θ' ← τ·θ + (1-τ)·θ'（每步都做一点点，比 DQN 每 10 回合硬拷贝更平滑）
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)

# =================== 评估函数 ===================
def evaluate(agent):
    """用单个环境（无并行、无渲染）评估当前策略（确定性动作 tanh(μ)），返回平均回报
    Pendulum 回报参考：随机策略约 -1200，优秀策略约 -150（越接近 -150 越好）
    """
    eval_env = gym.make("Pendulum-v1")
    eval_env.reset(seed=SEED)
    total = 0.0
    for _ in range(EVAL_EPISODES):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)   # 确定性动作（归一化）
            state, reward, terminated, truncated, _ = eval_env.step(action * ACTION_SCALE)
            done = terminated or truncated
            total += reward
    eval_env.close()
    return total / EVAL_EPISODES

# =================== 训练循环 ===================
agent = SACAgent()     # 创建智能体（actor、双 Q、目标网络、α、三个优化器、经验池全在里面）
episode_rewards = []   # 记录每个回合的总回报（用于绘图）

print("===== SAC 开始训练（Pendulum-v1）=====")
env.reset(seed=SEED)   # 给环境播种一次：之后每次 reset 按确定序列给初始状态，保证可复现

total_steps = 0        # 全局步数（用于判断随机预热期）
for ep in range(EPISODES):
    state, _ = env.reset()   # Pendulum 初始角度随机
    ep_reward = 0.0          # 本回合累计回报清零
    done = False

    while not done:          # Pendulum 每回合固定 200 步（到了就 truncated）
        # ---------- 1. 选择动作 ----------
        if total_steps < START_STEPS:
            # 预热阶段：环境动作空间均匀随机采样 → 除以 2 归一化到 [-1,1]
            # （与策略输出同一空间，之后可以混在同一个经验池里正常训练）
            action = env.action_space.sample() / ACTION_SCALE
        else:
            # 正常阶段：策略网络采样（随机性来自高斯分布本身，不需要 DQN 的 ε）
            action = agent.select_action(state)

        # ---------- 2. 执行动作 ----------
        # 归一化动作 × 2 → 真实扭矩 ∈ [-2,2]（环境内部还会自己 clip 一次）
        next_state, reward, terminated, truncated, _ = env.step(action * ACTION_SCALE)
        done = terminated or truncated   # 任一为 True 即认为本回合结束

        # ---------- 3. 存入经验池（与 DQN 相同：off-policy，先存后学） ----------
        agent.store_transition(state, action, reward, next_state, done)

        # ---------- 4. 学一步（池不满 BATCH_SIZE 条时自动跳过） ----------
        agent.update()

        state = next_state
        ep_reward += reward
        total_steps += 1

    episode_rewards.append(ep_reward)

    # ---------- 打印训练进度（含 α，观察自动温度的调节过程） ----------
    if (ep + 1) % LOG_INTERVAL == 0:
        avg = np.mean(episode_rewards[-LOG_INTERVAL:])
        print(f"回合 {ep+1:4d} | 步数 {total_steps:6d} | 最近{LOG_INTERVAL}回合平均回报 {avg:8.2f} | α = {agent.alpha.item():.3f}")

    # ---------- 定期评估（确定性动作） ----------
    if (ep + 1) % EVAL_INTERVAL == 0:
        avg_eval = evaluate(agent)
        print(f"           [评估] 确定性策略 {EVAL_EPISODES} 局平均回报: {avg_eval:8.2f}（Pendulum 最优约 -150，接近即为学会）")

print("===== 训练完成 =====")

#===================保存模型 ===================
# 测试只需要 actor（策略网络）；Q 网络、目标网络、α 都只是训练时的"脚手架"
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sac_pendulum.pth")
torch.save({
    'actor_state_dict': agent.actor.state_dict(),
    # state_dim/action_dim 是 numpy 整数（shape[0] 返回 np.int64），
    # 必须用 int() 转成 Python 整数，否则 PyTorch 2.6+ 的 torch.load 默认
    # weights_only=True 会拒绝加载含 numpy 标量的文件（报 UnpicklingError）
    'hyperparams': {'state_dim': int(state_dim), 'action_dim': int(action_dim), 'HIDDEN': 256},
}, SAVE_PATH)
print(f"模型已保存为 {SAVE_PATH}")

# =================== 绘制学习曲线 ===================
# 每回合回报波动大，叠一条 20 回合滑动平均看趋势
plt.plot(episode_rewards, alpha=0.35, label="Episode Reward")
if len(episode_rewards) >= LOG_INTERVAL:
    window = LOG_INTERVAL
    smooth = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    plt.plot(range(window - 1, len(episode_rewards)), smooth, label=f"Moving Average ({window})")
plt.axhline(y=-200, color="gray", linestyle="--", label="Reference (-200)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SAC Training Curve (Pendulum-v1)")
plt.legend()
plt.show()
env.close()

# ============================================================================
# 【理解笔记】SAC 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体（对象准备）
#   agent = SACAgent()，它自带以下固有属性：
#     - actor      ：策略网络，输出 μ、log σ → 重参数化采样 → tanh 得归一化动作
#     - q1 / q2    ：两个独立的 Q 网络（双 Q）；输入 (s, a)，输出标量 Q 值
#     - q1_target / q2_target：目标 Q 网络，只用来算备份目标 y，每步被软更新
#     - log_alpha  ：温度参数（α = exp(log_alpha) 恒正），它本身也是可训练的
#     - buffer     ：经验回放池（容量 10 万，存归一化动作）
#     - 三个优化器：actor_optimizer / q_optimizer(q1+q2) / alpha_optimizer
#   提供的方法：选择动作（采样/确定性）、存放经验、更新（一次更新含 4 个子步骤）
#
# 第二阶段：回合循环（外层 for ep in range(1000)）
#   1. env.reset() → 初始状态（Pendulum 初始角度随机）
#   2. 步循环（内层 while not done，每回合固定 200 步）每步做四件事：
#      ① 选动作：总步数 < 1 万 → 均匀随机（归一化后存池）；之后 actor 采样
#      ② 执行动作：归一化动作 × 2 → 真实扭矩 → env.step
#      ③ 存经验：push(s, a, r, s', done)
#      ④ 学一步：agent.update()（池不满 256 条时跳过），顺序为
#         a. 更新 Q1、Q2：朝 y = r + γ(1-done)·[min(Q1',Q2') - α·log π(a'|s')] 回归（MSE）
#         b. 更新 actor：最小化 α·log π - min(Q1,Q2)(s,a)（重参数化让梯度直通）
#         c. 更新 α：让策略的随机程度逼近 H̄ = -dim(A)（太随机→调小，太确定→调大）
#         d. 软更新目标网络：θ' ← τθ + (1-τ)θ'
#   3. 回合收尾：记录总回报；每 20 回合打印（含 α）；每 100 回合评估
#
# 第三阶段：保存 actor 权重 + 画学习曲线
#
# 【和前面三个算法的对照】
#   - 抽样方式：A2C/PPO 是 on-policy（数据用完即弃）；SAC 是 off-policy（池里反复用，样本效率高）
#   - 策略输出：A2C/PPO 离散（Categorical softmax）；SAC 连续（高斯 + tanh 压缩，可微采样）
#   - 探索方式：DQN 靠 ε-贪婪（动作层面加噪声）；SAC 靠"最大熵"（策略自带方差 + α 自动调节）
#   - 目标网络：DQN 每 10 回合硬拷贝；SAC 每步 τ=0.005 软更新
#   - 价值高估：DQN 单 Q（或 DDQN 缓解）；SAC 双 Q 取 min
#   - 更新时机：PPO 攒一批数据更新 EPOCHS 轮；SAC 每走一步就学一次
# ============================================================================
