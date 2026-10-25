# ============================================================
# DDPG（Deep Deterministic Policy Gradient）训练代码 —— Pendulum-v1（连续控制）
# DDPG = DPG + 【目标网络（软更新）】+ 一点工程细节
#
# 关键认知：DDPG 的两条更新公式和 DPG【一个字都没变】，它解决的不是“公式不会写”，
#   而是“同样的公式在深度网络上根本训不动”。所以本文件与 DPG_train.py 的差异
#   只有下面 4 处（其余代码逐行相同，方便直接 diff 对照）：
#     差异① 多造两个目标网络 actor_target / critic_target，初始化时拷贝一份参数
#     差异② 目标 y 里的 Q 和 μ 换成目标网络：y = r + γ·Q'(s', μ'(s'))
#     差异③ 每步结束后做软更新：θ' ← τθ + (1-τ)θ'（DPG 里完全没有这一步）
#     差异④ 超参：BATCH_SIZE 128（论文 64，Pendulum 状态少取大一点更稳）、
#                 TAU = 0.005（论文 0.001；“略调”是为了让目标在 10 万步内跟得上）
#
# 为什么这 4 处这么关键（三层放大器，见文末理解笔记）：
#   ① y 里用 Q 自己的估计 → 误差自我强化（自举高估）
#   ② ∇a Q 的误差原样传给 Actor（第七章：Actor 梯度的质量全押在 ∇a Q 上）
#   ③ Actor 一动，y 里的 μ(s') 立刻跟着动 → 目标追着自己跑
#   目标网络把 ①②③ 的回路【掐断】：y 里的东西变成“慢半拍、基本不动”的快照。
#   τ=0.005 意味着目标大约 200 步跟完一次（时间常数 1/τ），
#   而 Actor 每步只挪一点点，两者节奏错开，训练才收敛。
#
# 四个网络的分工（对比 DQN 只有 2 个，容易漏）：
#   μ  ：Actor（学生）—— 被训练，直接进环境
#   Q  ：Critic（学生）—— 被训练，给 Actor 提供 ∇a Q
#   μ' ：目标 Actor —— 只用来算 y（提供 s' 上的动作），不被梯度更新
#   Q' ：目标 Critic —— 只用来算 y（评估 s' 上的动作），不被梯度更新
#
# 【论文里还有、但本文件刻意省略的两样东西】（它们不属于算法骨架）：
#   · Batch Norm：论文给所有层加了 BN，是为了处理“观测各分量物理单位不同”
#     （像素 0~255 vs 位置 0.001）；Pendulum 状态只有 3 维且量级接近，加了反而添乱。
#     作者在后续实验中也承认低维任务不需要它。
#   · OU 噪声：论文用 Ornstein-Uhlenbeck 过程（有“惯性”的噪声）做探索，
#     但平稳高斯噪声在实践中通常一样好，TD3 论文直接用高斯噪声。这里用高斯。
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

# 全代码统一约定（与 DPG / SAC 完全一致，方便几个文件对照着看）：
#   策略网络输出的“归一化动作” ∈ [-1,1]（tanh 压缩的结果）；
#   送进 env.step 前乘 ACTION_SCALE 还原成真实扭矩；
#   经验池与 Q 网络统一使用【归一化动作】。

# =================== 超参数 ===================
EPISODES      = 100        # 训练回合数（Pendulum 每回合固定 200 步 → 共 2 万步；想先快速看趋势可以减小）
BUFFER_SIZE   = 100000     # 经验池容量（论文 100 万，Pendulum 10 万足够；off-policy 要存很多旧数据）
BATCH_SIZE    = 128        # 小批量大小（论文 64，这里取 128：状态维度低，多点样本估计更稳）
GAMMA         = 0.99       # 折扣因子
TAU           = 0.005      # 软更新系数（论文 0.001；这里略调大，让目标在 10 万步内跟得上）
LR_ACTOR      = 1e-4       # actor 学习率（论文原值；比 critic 小一个量级，策略是“被 Q 牵着走”的）
LR_CRITIC     = 1e-3       # critic 学习率（论文原值）
NOISE_SIGMA   = 0.1        # 探索噪声标准差（归一化动作空间上的高斯噪声，σ=0.1 即 10% 满量程）
START_STEPS   = 5000       # 前 5000 步用均匀随机动作预热经验池（类似 DQN 一开始 ε=1 的阶段）
EVAL_INTERVAL = 50         # 每多少回合评估一次
EVAL_EPISODES = 5          # 每次评估跑几局（确定性动作）
LOG_INTERVAL  = 20         # 每多少回合打印一次训练信息
SEED          = 42         # 随机种子（固定训练过程，保证可复现）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =================== 策略网络（Actor）===================
class ActorNet(nn.Module):
    """确定性策略网络 μ(s; θ)：输入状态，直接输出【唯一的一个动作】
    对比 A2C/PPO/SAC：那边输出的是分布参数（Categorical 的 logits / 高斯 μ、log σ），
    决策 = 从分布里采样；这里输出的是一个确定的数值，决策 = 前向传播本身（没有任何随机性）。

    最后过 tanh 压到 (-1,1)：与 DPG / SAC 同款做法，全代码统一使用【归一化动作】，
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

    def predict(self, state):
        """单条状态（numpy）→ 归一化动作（numpy），供环境交互使用"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)   # (3,) → [1, 3]
        with torch.no_grad():
            action = self.forward(state_t)
        return action.squeeze(0).cpu().numpy()   # [1, 1] → (1,)

# =================== 动作价值网络（Critic）===================
class CriticNet(nn.Module):
    """Q(s,a) 网络：把状态和动作【拼接】后输入（对比 DQN 的 Q 网络只输入状态）
    这是 DDPG 能把动作变成连续实数的关键：Q 不再是一张“动作表”，
    而是一个关于 a 的【光滑函数】——于是可以对 a 求导，用 ∇a Q 指导 Actor。
    对比 SAC / TD3：它们用两个 Q 取 min；DDPG 只用【一个】Q（高估问题留给 TD3 修）
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
    """预分配 numpy 数组 + 环形缓冲（与 DPG / SAC 里的写法完全相同）
    DDPG 是 off-policy 算法，池子照用：数据由“行为策略 μ(s)+噪声”产生，
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

# =================== DDPG 智能体 ===================
class DDPGAgent:
    def __init__(self):
        # ---- 两个“学生”网络（要被训练）----
        self.actor  = ActorNet(state_dim, action_dim).to(DEVICE)    # 策略 μ(s)：决定“做什么”
        self.critic = CriticNet(state_dim, action_dim).to(DEVICE)   # 价值 Q(s,a)：评价“有多好”
        # ---- 两个“老师”目标网络（只用来算目标 y，不直接被训练）----
        self.actor_target  = ActorNet(state_dim, action_dim).to(DEVICE)
        self.critic_target = CriticNet(state_dim, action_dim).to(DEVICE)
        # 初始化时先同步一次：目标网络一开始必须和学生一模一样
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 注意：目标网络【不需要】优化器——它们从不参与梯度更新，只靠软更新慢慢跟住学生

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

    def soft_update(self):
        """差异③：软更新 θ' ← τθ + (1-τ)θ'（每步都做一点点，而不是定期硬拷贝）
        对比 DQN：那边每 10 回合 θ' ← θ（硬拷贝，目标会“跳变”）；
        这里每步挪 τ=0.005，目标是一条平滑移动的曲线，Actor 能跟上。
        """
        #与Dueling DQN不同，这里并不会解决高估Q的问题，只是让高估的速度变慢而已。
        #目的主要是解决Actor的梯度问题，毕竟Actor的梯度是由Critic提供的，如果Critic的Q值估计不准，那么Actor的梯度也会不准。
        with torch.no_grad():   # 目标网络不是“被训练”的对象，只是被赋值，不需要梯度
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - TAU)          # θ' ← (1-τ)·θ'
                target_param.data.add_(TAU * param.data)  # θ' ← θ' + τ·θ
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
        # 这个更新没有显式的等号，因为 target_param.data.mul_(1 - TAU) 就相当于
        # target_param.data = target_param.data * (1 - TAU)，add_ 同理（下划线 = 原地操作）

    def update(self):
        """一次学习：从池中采 BATCH_SIZE 条经验，先更新 Critic，再更新 Actor，最后软更新目标网络
        Critic 目标（差异②：全部换成目标网络，其他与 DPG 一字不差）：
            y = r + γ(1-done)·Q'(s', μ'(s'))     ← 下一步的动作由【目标策略】给出，不加噪声
        Actor 目标（与 DPG 完全相同，用的是【当前】Q 和【当前】μ）：
            max E[ Q(s, μ(s)) ]                  ← 让 μ 往 Q 值高的方向长
        """
        if len(self.buffer) < BATCH_SIZE:
            return   # 池子不够 128 条：先不学（与 DQN / DPG / SAC 的同一个套路）

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)       # [B, 3]
        actions     = torch.FloatTensor(actions).to(DEVICE)      # [B, 1]（归一化动作，含探索噪声）
        rewards     = torch.FloatTensor(rewards).to(DEVICE)      # [B, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)  # [B, 3]
        dones       = torch.FloatTensor(dones).to(DEVICE)        # [B, 1]

        # ============ ① 更新 Critic：回归 TD 目标 ============
        # 与 DPG 的差别只有“目标网络”三个字：μ → μ'，Q → Q'
        # （这就是 DQN 笔记里的“原理二：为什么要有目标网络”在连续动作下的版本）
        with torch.no_grad():   # 目标 y 里的所有估计部分都冻结
            next_actions = self.actor_target(next_states)            # μ'(s')，【不带探索噪声】
            q_next  = self.critic_target(next_states, next_actions)   # Q'(s', μ'(s')) → [B, 1]
            backup  = rewards + GAMMA * (1.0 - dones) * q_next        # [B, 1]
        q = self.critic(states, actions)                              # Q(s, a)，用【当前】Q
        # 论文写成逐样本的 w ← w + αw·δ·∇w Q，δ 是 TD 误差；F.mse_loss 是它的小批量版
        # （对 δ² 求导得到 -2δ·∇w Q，常数 2 被学习率吸收）。
        critic_loss = F.mse_loss(q, backup)
        self.critic_optimizer.zero_grad()   # 清空上次的梯度
        critic_loss.backward()              # 反向传播，自动算出 loss 对每个参数的偏导数
        self.critic_optimizer.step()        # 更新权重

        # ============ ② 更新 Actor：与 DPG 完全相同 ============
        # 数学上的梯度是 ∇θ μ(s)·∇a Q(s,a)；代码里【不需要手写这个乘积】——
        # 把 a = μ(s) 直接喂给 Q，再对 actor_loss 调用 backward()，
        # 链式法则会自动算出 ∂Q/∂a（从 Q 网络那一侧）乘以 ∂μ/∂θ（从 actor 这一侧）。
        # 注意：这里用的是【当前】Q 和【当前】μ，不是目标网络
        # （目标网络只服务于“算 y”，不参与 Actor 的改进方向）
        pred_actions = self.actor(states)                        # a = μ(s)，带梯度
        actor_loss   = -self.critic(states, pred_actions).mean()  # 取负号 → 让优化器“最小化”= 最大化 Q
        self.actor_optimizer.zero_grad()   # 只清 actor 的梯度
        actor_loss.backward()              # 反传：梯度穿过 Q 网络（但不更新它）直到 actor 参数
        self.actor_optimizer.step()
        # 小提醒：这一步 backward 也会在 Critic 的参数上留下梯度，但这里【不会】调用
        # critic_optimizer.step()，所以 Critic 不受影响；那些残留梯度会在下一次
        # critic_optimizer.zero_grad() 时被清掉（下一次 ① 里）。

        # ============ ③ 软更新目标网络（DPG 完全没有这一步）============
        self.soft_update()

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
agent = DDPGAgent()    # 创建智能体（4 个网络、2 个优化器、经验池全在里面）
episode_rewards = []   # 记录每个回合的总回报（用于绘图）

print("===== DDPG 开始训练（Pendulum-v1）=====")
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
# 测试只需要 actor（策略网络）；Critic、两个目标网络都只是训练时的“脚手架”
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ddpg_pendulum.pth")
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
plt.title("DDPG Training Curve (Pendulum-v1)")
plt.legend()
plt.show()
env.close()

# ============================================================================
# 【理解笔记】DDPG 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体（对象准备）
#   agent = DDPGAgent()，它自带以下固有属性：
#     - actor / actor_target   ：策略网络 μ 与它的目标版本 μ'（结构相同，初始参数相同）
#     - critic / critic_target ：Q 网络与它的目标版本 Q'
#     - buffer                 ：经验回放池（容量 10 万，存【加过噪声】的行为动作）
#     - 两个优化器             ：actor_optimizer / critic_optimizer（目标网络【没有】优化器）
#   提供的方法：选择动作（可选噪声）、存放经验、更新（一次 update 含 3 个子步骤）、软更新
#
# 第二阶段：回合循环（外层 for ep in range(500)）
#   1. env.reset() → 初始状态（Pendulum 初始角度随机）
#   2. 步循环（内层 while not done，每回合固定 200 步）每步做四件事：
#      ① 选动作：总步数 < 5000 → 均匀随机；之后 a = μ(s) + N(0, 0.1)，clip 到 [-1,1]
#      ② 执行动作：归一化动作 × 2 → 真实扭矩 → env.step
#      ③ 存经验：push(s, a, r, s', done)（a 是加噪后的行为动作）
#      ④ 学一步：agent.update()（池不满 128 条时跳过），顺序为
#         a. 更新 Critic：朝 y = r + γ(1-done)·Q'(s', μ'(s')) 回归（MSE）← 用目标网络
#         b. 更新 Actor ：最小化 -Q(s, μ(s))，梯度经 ∂Q/∂a · ∂μ/∂θ 自动回传 ← 用当前网络
#         c. 软更新      ： critic_target ← τ·critic + (1-τ)·critic_target（actor_target 同理）
#   3. 回合收尾：记录总回报；每 20 回合打印；每 50 回合评估
#
# 第三阶段：保存 actor 权重 + 画学习曲线
#
# ============================================================================
# 【和 DPG 的逐行对照】（本文件与 DPG_train.py 的全部差异）
#   1. 类名 DPGAgent → DDPGAgent（无实质变化）
#   2. __init__ 里多 2 个网络 + load_state_dict 同步 + 一句话注释“目标网络没有优化器”
#   3. 新增 soft_update() 方法（约 8 行）
#   4. update() 的 ① 里：self.actor → self.actor_target，self.critic → self.critic_target
#   5. update() 末尾多一行 self.soft_update()
#   6. 超参：TAU=0.005（新增）、BATCH_SIZE 128
#   7. 保存文件名 dpg_ → ddpg_
#   —— 就这些。所以 DDPG 的“新知识”不是公式，而是【目标网络 + 软更新】这套稳定化机制。
#
# ============================================================================
# 【和已学算法的对照】
#   - 策略参数化：A2C/PPO/SAC 输出分布（要采样）；DDPG 输出一个确定的动作（没有采样）
#   - 学习信号  ：REINFORCE/A2C/PPO 用 log π·A（似然比，事后加权，方差大）；
#                 DDPG 用 ∇a Q（路径导数，Q 光滑 → 直接指出“往哪挪”，方差小、样本效率高）
#   - 探索方式  ：DQN 用 ε-贪婪；PPO 用熵正则；SAC 靠最大熵；DDPG/DPG 只能【外挂高斯噪声】
#   - 动作处理  ：DQN 靠 argmax（离散）；DDPG 靠“策略网络直接给动作”（连续）
#   - 目标网络  ：DQN 每 10 回合硬拷贝 θ'←θ；DDPG 每步软更新 θ'←τθ+(1-τ)θ'；DPG 没有
#   - 网络的个数：DQN 2 个（Q、Q'）；DDPG 4 个（μ、Q、μ'、Q'）
#
# 【DDPG 修了什么、没修什么】
#   修了：短板①（没有目标网络 → 目标追自己）。证据：同一套数据、同样的超参，
#         DPG 的曲线会突然爬升又突然回落，DDPG 的曲线基本单调上升（可自行对比两张图）
#   没修：短板②（单个 Q + 自举 → Q 值系统性【高估】）
#         症状：Q 的绝对值越来越大，而实际回报没涨；Actor 被虚高的 Q 带偏，
#               在“自认为好”的动作上越走越远
#         原因：y 里的 max/目标动作仍然来自同一个 Q，误差是【单向】的（永远偏大）
#         谁来修：TD3 的 clipped double Q（借鉴 DDQN 的“选评解耦”）
#   没修：短板③（确定性策略 + 固定噪声 → 探索与 Q 污染的两难）
#         谁来修：TD3 的 target policy smoothing（把噪声加在目标动作上，让 Q 变平滑）
#                 + 延迟更新 Actor（每 2 步一次，先让 Critic 稳下来）
#
# 【通向 TD3 的三个补丁（第九章预告）】
#   补丁① Clipped Double Q：多一个 Q2，y 里取 min(Q1', Q2')，压高估
#   补丁② Delayed Policy Update：Critic 每步都更新，Actor（和目标网络）每 2 步才更新一次
#   补丁③ Target Policy Smoothing：y 里的目标动作加截断噪声
#          a' = clip(μ'(s') + clip(ε, -c, c), -a_max, a_max)，ε ~ N(0, σ²)
#          作用：让 Q 在动作邻域内被“抹平”，避免 Actor 钻到 Q 的尖峰（错估）里
# ============================================================================
