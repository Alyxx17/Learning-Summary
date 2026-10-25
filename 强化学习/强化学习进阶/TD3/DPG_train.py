# ============================================================
# DPG（Deterministic Policy Gradient，确定性策略梯度）训练代码 —— Pendulum-v1（连续控制）
# DPG 是 DDPG / TD3 的地基：它只回答一个问题——
#   “策略如果是【确定性的】a = μ(s)，梯度该怎么求？”
#
# 三个关键词（看代码时逐个对照）：
#   ① 确定性策略：策略网络直接吐出【唯一动作】a = μ(s)，不再输出分布、不再采样
#   ② off-policy 行为策略：为了探索，收集数据时用 a = μ(s) + 高斯噪声；
#      但学习的目标始终是【不带噪声的确定性策略 μ】——这就是“确定性策略 + off-policy”的组合
#   ③ 只用两条公式：Critic 学 Q(s,a)，Actor 沿 ∇a Q 的方向挪动自己的动作（DPG 定理）
#
# 两个核心公式（全文只围绕这两行展开）：
#   Critic：δ = r + γ·Q(s', μ(s')) - Q(s, a)          → 回归 TD 目标（结构同 DQN）
#   Actor ：∇θ J = E[ ∇θ μ(s) · ∇a Q(s,a)|_{a=μ(s)} ] → 沿“Q 值上升方向”挪动作（DPG 定理）
#
# 与已学算法的关系：
#   相比 REINFORCE / A2C / PPO：策略从“随机分布”变成“确定性函数”。这时 log π(a|s) 恒为 -∞，
#     【似然比（log 技巧）整条路直接作废】——必须换一条路，这就是 DPG 定理存在的理由；
#   相比 DQN：把“先学 Q 再 argmax”换成“策略网络直接输出动作”，于是能处理连续动作；
#   相比 SAC：没有最大熵、没有双 Q、【没有目标网络】——这些正是后面 DDPG / TD3 要逐步补上的。
#     （所以本文件故意留着这些短板，用来当 DDPG 的“对照组”，见文末理解笔记）
#
# 【标准性说明】本文件 = 论文《Deterministic Policy Gradient Algorithms》(Silver 等, 2014)
#   的 “Simple Deterministic Actor-Critic” + 两处工程化改造，也就是学界公认“能跑起来的 DPG”：
#     改造① 行为策略加高斯噪声 —— 论文原版用 a = μ(s) 自身采样，那根本无法探索；
#     改造② 经验回放池        —— 论文 4.2 节已证明 off-policy 梯度【不需要重要性采样权重】
#                                 （这是确定性策略相对随机策略的一大便宜，见学习笔记 2.4）
#   刻意【不包含】以下内容（它们属于后面两级台阶，不属于 DPG 本身）：
#     目标网络（DDPG）、双 Q 取 min（TD3）、延迟更新（TD3）、最大熵与自动温度（SAC）
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

# 全代码统一约定（与 SAC 完全一致，方便两个文件对照着看）：
#   策略网络输出的“归一化动作” ∈ [-1,1]（tanh 压缩的结果）；
#   送进 env.step 前乘 ACTION_SCALE 还原成真实扭矩；
#   经验池与 Q 网络统一使用【归一化动作】。

# =================== 超参数 ===================
EPISODES      = 500        # 训练回合数（Pendulum 每回合固定 200 步 → 共 10 万步；想先快速看趋势可以减小）
BUFFER_SIZE   = 100000     # 经验池容量（off-policy 要存很多旧数据）
BATCH_SIZE    = 128        # 小批量大小（每步从池中采一批来学）
GAMMA         = 0.99       # 折扣因子
LR_ACTOR      = 1e-4       # actor 学习率（比 critic 小一个量级：策略是“被 Q 牵着走”的，步子要小）
LR_CRITIC     = 1e-3       # critic 学习率
NOISE_SIGMA   = 0.1        # 探索噪声标准差（归一化动作空间上的高斯噪声，σ=0.1 即 10% 满量程）
START_STEPS   = 5000       # 前 5000 步用均匀随机动作预热经验池（类似 DQN 一开始 ε=1 的阶段）
EVAL_INTERVAL = 50         # 每多少回合评估一次
EVAL_EPISODES = 5          # 每次评估跑几局（确定性动作）
LOG_INTERVAL  = 20         # 每多少回合打印一次训练信息
SEED          = 42         # 随机种子（固定训练过程，保证可复现）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)
###先看看要补什么学过的算法基础
###先看看要补什么学过的算法基础
###先看看要补什么学过的算法基础
###先看看要补什么学过的算法基础
###先看看要补什么学过的算法基础
###先看看要补什么学过的算法基础
# =================== 策略网络（Actor）===================
class ActorNet(nn.Module):
    """确定性策略网络 μ(s; θ)：输入状态，直接输出【唯一的一个动作】
    对比 A2C/PPO/SAC：那边输出的是分布参数（Categorical 的 logits / 高斯 μ、log σ），
    决策 = 从分布里采样；这里输出的是一个确定的数值，决策 = 前向传播本身（没有任何随机性）。

    最后过 tanh 压到 (-1,1)：与 SAC 同款做法，全代码统一使用【归一化动作】，
    只在 env.step 那一行乘 ACTION_SCALE 还原成真实扭矩。
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),   # 输出层：一个动作维度一个标量
        )

    def forward(self, x):
        return torch.tanh(self.fc(x))   # [batch, action_dim]，取值范围 (-1,1)

    def predict(self, state):#与环境交互时
        """单条状态（numpy）→ 归一化动作（numpy），供环境交互使用"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)   # (3,) → [1, 3]
        with torch.no_grad():
            action = self.forward(state_t)
        return action.squeeze(0).cpu().numpy()   # [1, 1] → (1,)

# =================== 动作价值网络（Critic）===================
class CriticNet(nn.Module):
    """Q(s,a) 网络：把状态和动作【拼接】后输入（对比 DQN 的 Q 网络只输入状态）
    这是 DPG 能把动作变成连续实数的关键：Q 不再是一张“动作表”，
    而是一个关于 a 的【光滑函数】——于是可以对 a 求导，用 ∇a Q 指导 Actor。
    对比 SAC：SAC 用两个 Q 取 min；这里只用【一个】Q（先留个短板，留给 TD3 补）
    """
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
        x = torch.cat([state, action], dim=1)   # 按列拼接 → [batch, state_dim + action_dim]
        return self.fc(x)                       # [batch, 1]

# =================== 经验回放池 ===================
class ReplayBuffer:
    """预分配 numpy 数组 + 环形缓冲（与 SAC 里的写法完全相同）
    DPG 是 off-policy 算法，池子照用：数据由“行为策略 μ(s)+噪声”产生，
    但学的是确定性策略 μ —— 这种“数据来源 ≠ 学习目标”正是 off-policy 的定义。
    """
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

# =================== DPG 智能体 ===================
class DPGAgent:
    def __init__(self):
        # ---- 两个网络（要被训练）----
        self.actor  = ActorNet(state_dim, action_dim).to(DEVICE)    # 策略 μ(s)：决定“做什么”
        self.critic = CriticNet(state_dim, action_dim).to(DEVICE)   # 价值 Q(s,a)：评价“有多好”
        # 注意：这里【没有】target 网络。TD 目标 y = r + γQ(s', μ(s')) 里的 Q 和 μ
        # 用的就是正在被更新的自己 —— 目标本身在动，这是 DPG 不稳定的根源之一（DDPG 才修）
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)   # 新建一个容量 10 万的经验池

    def select_action(self, state, noise_std=0.0):
        """行为策略 π_b：a = μ(s) + N(0, σ)，再 clip 回 [-1,1]
        返回【归一化动作】∈[-1,1]（乘 ACTION_SCALE 后才是真实扭矩）。
        - noise_std > 0：训练时收集数据用（确定性策略自己不会探索，只能外挂噪声）
        - noise_std = 0：评估/测试用（此时行为策略 = 目标策略，动作完全确定）
        """
        action = self.actor.predict(state)                       # μ(s)（确定性输出）
        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)   # 加高斯噪声
            action = np.clip(action, -1.0, 1.0)                  # 噪声可能把动作顶出界，夹回来
        return action

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        """一次学习：从池中采 BATCH_SIZE 条经验，先更新 Critic，再更新 Actor
        Critic 目标（普通的贝尔曼方程，没有熵项）：
            y = r + γ(1-done)·Q(s', μ(s'))       ← 下一步的动作由【确定性策略】给出，不加噪声
        Actor 目标（DPG 定理，写成“最大化”形式）：
            max E[ Q(s, μ(s)) ]                  ← 让 μ 往 Q 值高的方向长
        """
        if len(self.buffer) < BATCH_SIZE:
            return   # 池子不够 128 条：先不学（与 DQN/SAC 的同一个套路）

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)       # [B, 3]
        actions     = torch.FloatTensor(actions).to(DEVICE)      # [B, 1]（归一化动作，含探索噪声）
        rewards     = torch.FloatTensor(rewards).to(DEVICE)      # [B, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)  # [B, 3]
        dones       = torch.FloatTensor(dones).to(DEVICE)        # [B, 1]

        # ============ ① 更新 Critic：回归 TD 目标 ============
        # 与 DQN 的唯一区别在箭头处：DQN 要 argmax 挑动作，这里直接用 μ(s') 给出唯一动作
        # （能这么干的前提：μ 就是那个“当前要被改进的策略”，这正是 DPG 定理给的权利）
        with torch.no_grad():   # 目标 y 里的所有估计部分都冻结（同 DQN 的目标网络思想）
            next_actions = self.actor(next_states)              # μ(s')，【不带探索噪声】
            q_next  = self.critic(next_states, next_actions)     # Q(s', μ(s')) → [B, 1]
            backup  = rewards + GAMMA * (1.0 - dones) * q_next   # [B, 1]
        q = self.critic(states, actions)                         # Q(s, a)，a 是池里存的行为动作
        # 论文第 3.2 节写的是逐样本的 w ← w + αw·δ·∇w Q(s,a)（δ 是 TD 误差），
        # 这里用 F.mse_loss 是它的“小批量版”：对 δ² 求导正好得到 -2δ·∇w Q，只差一个常数 2
        # （会被学习率吸收）。所以 MSE 回归与论文的 δ·∇Q 是同一件事，同 DQN 的写法。
        critic_loss = F.mse_loss(q, backup)
        self.critic_optimizer.zero_grad()   # 清空上次的梯度
        critic_loss.backward()              # 反向传播，自动算出 loss 对每个参数的偏导数
        self.critic_optimizer.step()        # 更新权重
        #只更新 Critic

        # ============ ② 更新 Actor：DPG 定理 ============
        # 数学上的梯度是 ∇θ μ(s)·∇a Q(s,a)；代码里【不需要手写这个乘积】——
        # 把 a = μ(s) 直接喂给 Q，再对 actor_loss 调用 backward()，
        # 链式法则会自动算出 ∂Q/∂a（从 Q 网络那一侧）乘以 ∂μ/∂θ（从 actor 这一侧）。
        # 这也是 DPG 最巧妙的地方：Q 网络当了“可微的评价器”，直接告诉策略“动作往哪挪”
        pred_actions = self.actor(states)                        # a = μ(s)，带梯度
        actor_loss   = -self.critic(states, pred_actions).mean()  # 取负号 → 让优化器“最小化”= 最大化 Q
        self.actor_optimizer.zero_grad()   # 只清 actor 的梯度
        actor_loss.backward()              # 反传：梯度穿过 Q 网络（但不更新它）直到 actor 参数
        self.actor_optimizer.step()
        # 小提醒：这一步 backward 也会在 Critic 的参数上留下梯度，但这里【不会】调用
        # critic_optimizer.step()，所以 Critic 不受影响；那些残留梯度会在下一次
        # critic_optimizer.zero_grad() 时被清掉（下一次 ① 里）。
        #算出∇θ μ(s)·∇a Q(s,a)

# =================== 评估函数 ===================
def evaluate(agent):
    """用单个环境（无并行、无渲染）评估当前策略（noise_std=0，纯确定性动作），返回平均回报
    Pendulum 回报参考：随机策略约 -1200，优秀策略约 -150（越接近 -150 越好）
    """
    eval_env = gym.make("Pendulum-v1")
    eval_env.reset(seed=SEED)
    total = 0.0
    for _ in range(EVAL_EPISODES):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(state)     # 不加噪声 → 纯确定性策略
            state, reward, terminated, truncated, _ = eval_env.step(action * ACTION_SCALE)
            done = terminated or truncated
            total += reward
    eval_env.close()
    return total / EVAL_EPISODES

# =================== 训练循环 ===================
agent = DPGAgent()     # 创建智能体（actor、critic、两个优化器、经验池全在里面）
episode_rewards = []   # 记录每个回合的总回报（用于绘图）

print("===== DPG 开始训练（Pendulum-v1）=====")
env.reset(seed=SEED)   # 给环境播种一次：之后每次 reset 按确定顺序给出初始状态
total_steps = 0        # 全局步数（用于判断随机预热期）
for ep in range(EPISODES):
    state, _ = env.reset()   # Pendulum 初始角度随机
    ep_reward = 0.0          # 本回合累计回报清零
    done = False

    while not done:          # Pendulum 每回合固定 200 步（到了就 truncated）
        # ---------- 1. 选择动作（行为策略） ----------
        if total_steps < START_STEPS:
            # 预热阶段：环境动作空间均匀随机采样 → 除以 2 归一化到 [-1,1]
            # （与策略输出同一空间，之后可以混在同一个经验池里正常训练）
            action = env.action_space.sample() / ACTION_SCALE
        else:
            # 正常阶段：a = μ(s) + 高斯噪声（确定性策略的探索只能靠“外挂噪声”）
            action = agent.select_action(state, noise_std=NOISE_SIGMA)

        # ---------- 2. 执行动作 ----------
        # 归一化动作 × 2 → 真实扭矩 ∈ [-2,2]（环境内部还会自己 clip 一次）
        next_state, reward, terminated, truncated, _ = env.step(action * ACTION_SCALE)
        done = terminated or truncated   # 任一为 True 即认为本回合结束

        # ---------- 3. 存入经验池（存的是【加过噪声的行为动作】，不是 μ(s)） ----------
        # 因为 Critic 要学的 Q(s,a) 是“在这套行为分布下的价值”，必须存实际执行的动作
        agent.store_transition(state, action, reward, next_state, done)

        # ---------- 4. 学一步（池不满 BATCH_SIZE 条时自动跳过） ----------
        agent.update()

        state = next_state
        ep_reward += reward
        total_steps += 1

    episode_rewards.append(ep_reward)

    # ---------- 打印训练进度 ----------
    if (ep + 1) % LOG_INTERVAL == 0:
        avg = np.mean(episode_rewards[-LOG_INTERVAL:])
        print(f"回合 {ep+1:4d} | 步数 {total_steps:6d} | 最近{LOG_INTERVAL}回合平均回报 {avg:8.2f}")

    # ---------- 定期评估（确定性动作） ----------
    if (ep + 1) % EVAL_INTERVAL == 0:
        avg_eval = evaluate(agent)
        print(f"           [评估] 确定性策略 {EVAL_EPISODES} 局平均回报: {avg_eval:8.2f}（Pendulum 最优约 -150，接近即为学会）")

print("===== 训练完成 =====")

# =================== 保存模型 ===================
# 测试只需要 actor（策略网络）；Critic 只是训练时的“脚手架”
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpg_pendulum.pth")
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
plt.title("DPG Training Curve (Pendulum-v1)")
plt.legend()
plt.show()
env.close()

# ============================================================================
# 【理解笔记】DPG 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体（对象准备）
#   agent = DPGAgent()，它自带以下固有属性：
#     - actor  ：确定性策略网络 μ(s)，输出归一化动作 tanh(...) ∈ (-1,1)
#     - critic ：Q(s,a) 网络，输入 (s, a)，输出标量 —— 注意它是【可微的】，actor 靠它给方向
#     - buffer ：经验回放池（容量 10 万，存【加过噪声】的行为动作）
#     - 两个优化器：actor_optimizer / critic_optimizer
#     - 没有 target 网络！（这是刻意的，见下面“三个短板”）
#   提供的方法：选择动作（可选噪声）、存放经验、更新（一次 update 含 2 个子步骤）
#
# 第二阶段：回合循环（外层 for ep in range(500)）
#   1. env.reset() → 初始状态（Pendulum 初始角度随机）
#   2. 步循环（内层 while not done，每回合固定 200 步）每步做四件事：
#      ① 选动作：总步数 < 5000 → 均匀随机；之后 a = μ(s) + N(0, 0.1)，clip 到 [-1,1]
#      ② 执行动作：归一化动作 × 2 → 真实扭矩 → env.step
#      ③ 存经验：push(s, a, r, s', done)（a 是加噪后的行为动作）
#      ④ 学一步：agent.update()（池不满 128 条时跳过），顺序为
#         a. 更新 Critic：朝 y = r + γ(1-done)·Q(s', μ(s')) 回归（MSE），μ(s') 不加噪声
#         b. 更新 Actor ：最小化 -Q(s, μ(s))，梯度经 ∂Q/∂a · ∂μ/∂θ 自动回传到 μ
#   3. 回合收尾：记录总回报；每 20 回合打印；每 50 回合评估
#
# 第三阶段：保存 actor 权重 + 画学习曲线
#
# ============================================================================
# 【和已学算法的对照】
#   - 策略参数化：A2C/PPO/SAC 输出分布（要采样）；DPG 输出一个确定的动作（没有采样）
#   - 学习信号  ：REINFORCE/A2C/PPO 用 log π·A（似然比，事后加权，方差大）；
#                 DPG 用 ∇a Q（路径导数，Q 光滑 → 直接指出“往哪挪”，方差小、样本效率高）
#   - 探索方式  ：DQN 用 ε-贪婪；PPO 用熵正则；SAC 靠最大熵；DPG 只能【外挂高斯噪声】
#   - 动作处理  ：DQN 靠 argmax（离散）；DPG 靠“策略网络直接给动作”（连续）
#   - 目标网络  ：DQN 每 10 回合硬拷贝；SAC 每步软更新；DPG 【完全没有】
#
# 【DPG 的三个致命短板 → 也就是我们的学习路线图】
#   短板①：没有目标网络 → TD 目标里的 Q 和 μ 每步都在变，目标在“追自己”，
#          表现为 Q 值震荡、策略反复横跳，训练后期容易退化（这是 DPG 公认的稳定性问题，
#          也是 DDPG 第一个要补的东西）
#          → 【DDPG】补上：复制出 Q' 和 μ'，每步软更新 θ' ← τθ + (1-τ)θ'
#   短板②：单个 Q 网络 + 自举 → Q 值系统性【高估】，Actor 被虚高的 Q 带偏、
#          容易在“自认为好”的动作上越走越远（和 DQN 笔记里讲过的高估问题是同一个）
#          → 【DDPG】没解决；【TD3】用“双 Q 取 min”（借鉴 DDQN）来压
#   短板③：确定性策略 + 固定噪声 → 探索全靠噪声大小；噪声太小容易卡在局部最优，
#          太大又破坏已学到的策略（Q 值会被噪声动作“污染”）
#          → 【TD3】把噪声加在【目标动作】上（target policy smoothing），
#            并延迟更新 Actor（每 2 步一次），让 Critic 先稳定下来
# ============================================================================
