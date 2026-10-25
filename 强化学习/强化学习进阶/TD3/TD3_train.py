# ============================================================
# TD3（Twin Delayed DDPG）训练代码 —— Pendulum-v1（连续控制）
# TD3 = DDPG + 三个补丁，三个补丁【全部】只针对一件事：让 ∇a Q 更可信
#
#   补丁① Clipped Double Q（双 Q 取 min）—— 修“Q 值系统性高估”
#      DDPG 只有一个 Q，y = r + γQ'(s',μ'(s')) 里的 Q' 是“自己猜自己”，
#      误差是单向的（永远偏大）。TD3 学两个独立初始化的 Q，
#      y 里取 min(Q1', Q2')：只要有一个估低了就会拉住结果，高估被压住。
#      —— 这就是 DDQN“选评解耦”思想在连续动作下的升级版
#
#   补丁② Delayed Policy Update（延迟策略更新）—— 让 Critic 先稳下来
#      Critic 每步都更新，但 Actor（和两个目标网络）每 POLICY_DELAY=2 步才更新一次。
#      理由：Actor 一旦跑在 Critic 前面，就会去追一个还没估准的 ∇a Q，
#      越追越偏；让 Critic 先多学一步，Actor 再动手。
#
#   补丁③ Target Policy Smoothing（目标策略平滑）—— 修“Q 的尖峰”
#      确定性策略的 Q 容易在某个动作上冲出尖峰（错估），Actor 会一头钻进去。
#      TD3 在【目标动作】上加截断噪声：
#          ã = clip( μ'(s') + clip(ε, -c, c), -a_max, a_max ),  ε ~ N(0, σ²)
#      相当于要求“动作附近一圈的 Q 都得好”，尖峰被抹平，Actor 自然不去钻。
#      注意：噪声加在【目标动作】上，不是加在探索动作上——这是两个完全不同的东西。
#
# 与 DDPG 的逐行差异（对照 DDPG_train.py 看最快）：
#     1. 网络 4 个 → 6 个（多一个 Q2 和 Q2'）
#     2. update() 的目标 y 里多两段：加噪声（补丁③）+ 取 min（补丁①）
#     3. update() 里 Actor 与软更新被包进 `if self.total_it % POLICY_DELAY == 0:`（补丁②）
#     4. 新增计数器 self.total_it
#     5. 超参：学习率改动（论文 TD3 用 lr=3e-4 同值，DDPG 论文是 actor 1e-4 / critic 1e-3）、
#        BATCH_SIZE 256、新增 TARGET_NOISE / NOISE_CLIP / POLICY_DELAY
#
# 为什么 Actor 损失只用 Q1（而不是 min(Q1,Q2)）？
#   论文原话：策略改进用 Q1 即可。因为 min 已经用在【目标 y】上了，
#   高估已经被压住；Actor 再用 min 会让梯度更保守、学得更慢。
#   （部分实现用 min，也能跑；这里按论文来，便于和原文对照）
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

# 全代码统一约定（与 DPG / DDPG / SAC 完全一致，方便几个文件对照着看）：
#   策略网络输出的“归一化动作” ∈ [-1,1]（tanh 压缩的结果）；
#   送进 env.step 前乘 ACTION_SCALE 还原成真实扭矩；
#   经验池与 Q 网络统一使用【归一化动作】。
#   → 所以“动作上下界”在归一化空间里就是 ±1，
#     补丁③里的 a_max = 1，clip 到 [-1, 1] 即可。

# =================== 超参数（TD3 论文 / 官方实现的值）===================
EPISODES      = 100        # 训练回合数（Pendulum 每回合固定 200 步 → 共 2 万步；想先快速看趋势可以减小）
BUFFER_SIZE   = 100000     # 经验池容量（论文 100 万，Pendulum 10 万足够）
BATCH_SIZE    = 256        # 小批量大小（TD3 官方实现用 256）
GAMMA         = 0.99       # 折扣因子
TAU           = 0.005      # 软更新系数（TD3 论文原值，比 DDPG 的 0.001 大）
LR_ACTOR      = 3e-4       # actor 学习率（TD3 论文：actor 与 critic 用【同一个】学习率，与 DDPG 不同）
LR_CRITIC     = 3e-4       # critic 学习率
POLICY_DELAY  = 2          # 补丁②：Critic 每步都更新，Actor 每 2 步更新一次
TARGET_NOISE  = 0.2        # 补丁③：加在【目标动作】上的高斯噪声标准差 σ
NOISE_CLIP    = 0.5        # 补丁③：目标噪声的截断半径 c（防止把动作推太远）
EXPL_NOISE    = 0.1        # 探索噪声标准差（加在【行为动作】上，与 TARGET_NOISE 是两码事）
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
    与 DPG / DDPG 完全相同——TD3 的三个补丁都不改策略网络的结构，
    只改“什么时候更新它”（补丁②）和“目标里的动作怎么来”（补丁③）。

    最后过 tanh 压到 (-1,1)：全代码统一使用【归一化动作】，
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
    这是能把动作变成连续实数的关键：Q 是关于 a 的【光滑函数】，
    于是可以对 a 求导，用 ∇a Q 指导 Actor。

    相比 DDPG：TD3 需要【两个】结构相同但独立初始化的 Q（补丁①）。
    这里沿用 SAC 的写法——实例化两次同一个类，而不是让一个网络输出两个头。
    为什么独立初始化很重要：如果两个 Q 一开始一模一样，
    它们的误差就完全同步，取 min 等于没取（学不到互补的“保守估计”）。
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
    """预分配 numpy 数组 + 环形缓冲（与 DPG / DDPG / SAC 里的写法完全相同）
    TD3 是 off-policy 算法：数据由“行为策略 μ(s)+噪声”产生，
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

# =================== TD3 智能体 ===================
class TD3Agent:
    def __init__(self):
        # ---- 学生网络（要被训练）：1 个 Actor + 【2 个】Critic（补丁①）----
        self.actor = ActorNet(state_dim, action_dim).to(DEVICE)
        self.q1    = CriticNet(state_dim, action_dim).to(DEVICE)
        self.q2    = CriticNet(state_dim, action_dim).to(DEVICE)

        # ---- 目标网络（只用来算 y）：1 个 Actor' + 【2 个】Critic'，共 3 个 ----
        self.actor_target = ActorNet(state_dim, action_dim).to(DEVICE)
        self.q1_target    = CriticNet(state_dim, action_dim).to(DEVICE)
        self.q2_target    = CriticNet(state_dim, action_dim).to(DEVICE)
        # 初始化时先同步一次；注意 q1 与 q2 是各自独立初始化的，不要互相拷贝，
        # 否则补丁①失效（两个 Q 误差同步，min 起不到保守作用）
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        # 两个 Q 共用一个优化器：一次 backward 同时更新 Q1、Q2
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)
        self.total_it = 0   # 补丁② 用的计数器：记录 update() 被调用了几次

    def select_action(self, state, noise_std=0.0):
        """行为策略 π_b：a = μ(s) + N(0, σ)，再 clip 回 [-1,1]
        注意这里的噪声是【探索噪声】（EXPL_NOISE），与补丁③的【目标平滑噪声】无关：
          · 探索噪声：加在送给环境的动作上，目的是“多走一些没走过的路”
          · 平滑噪声：加在目标动作上，目的是“让 Q 在动作附近变平滑”
        两者作用位置、目的都不同，别混在一起。
        """
        action = self.actor.predict(state)                       # μ(s)（确定性输出）
        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
            action = np.clip(action, -1.0, 1.0)                  # 噪声可能把动作顶出界，夹回来
        return action

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def soft_update(self):
        """软更新三组目标网络：θ' ← τθ + (1-τ)θ'（Actor'、Q1'、Q2'）
        与 DDPG 的 soft_update 相比只是多了一组 Q2'；
        调用时机变了：只在【Actor 更新的那一步】调用（补丁②），而不是每步都调。
        """
        with torch.no_grad():   # 目标网络不是“被训练”的对象，只是被赋值，不需要梯度
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1 - TAU)
                target_param.data.add_(TAU * param.data)

    def update(self):
        """一次学习：Critic【每步】都更新，Actor 和三个目标网络【每 POLICY_DELAY 步】更新一次
        Critic 目标（三处改动全都在这几行里）：
            y = r + γ(1-done)·min( Q1'(s', ã), Q2'(s', ã) )        ← 补丁①
            ã = clip( μ'(s') + clip(ε, -c, c), -1, 1 )             ← 补丁③
        Actor 目标（与 DDPG 相同，只是调用频率减半）：
            max E[ Q1(s, μ(s)) ]
        """
        # 计数器在函数最前面自增：这样第 0 次 update 就会触发一次 Actor 更新
        self.total_it += 1

        if len(self.buffer) < BATCH_SIZE:
            return   # 池子不够 256 条：先不学（与 DQN / DPG / DDPG / SAC 的同一个套路）

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)       # [B, 3]
        actions     = torch.FloatTensor(actions).to(DEVICE)      # [B, 1]（归一化动作，含探索噪声）
        rewards     = torch.FloatTensor(rewards).to(DEVICE)      # [B, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)  # [B, 3]
        dones       = torch.FloatTensor(dones).to(DEVICE)        # [B, 1]

        # ============ ① 更新 Critic（每步都做）============
        with torch.no_grad():   # 目标 y 里的所有估计部分都冻结
            # --- 补丁③ Target Policy Smoothing ---
            next_actions = self.actor_target(next_states)                 # μ'(s')，[B,1]
            noise = torch.randn_like(next_actions) * TARGET_NOISE         # ε ~ N(0, σ²)
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)           # clip(ε, -c, c)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)   # clip 回动作范围
            # 两层 clip 缺一不可：
            #   内层限制“噪声本身”不超过 c → 保证平滑窗口不会太宽
            #   外层保证“加噪后的动作”仍在合法动作域里 → 否则 Q' 会在没见过的动作上瞎估

            # --- 补丁① Clipped Double Q ---
            q1_next = self.q1_target(next_states, next_actions)   # Q1'(s', ã)
            q2_next = self.q2_target(next_states, next_actions)   # Q2'(s', ã)
            q_next  = torch.min(q1_next, q2_next)                 # 取较小者 → 压住高估

            backup = rewards + GAMMA * (1.0 - dones) * q_next     # [B, 1]

        q1 = self.q1(states, actions)    # 用【当前】Q1 预测
        q2 = self.q2(states, actions)    # 用【当前】Q2 预测
        # 两个 Q 回归【同一个】目标 y（这就是 "clipped" 的含义：
        # 目标被 min 截断过，两个 Q 都被拉向这个保守值）
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.q_optimizer.zero_grad()   # 一次清空 Q1、Q2 两个网络的梯度
        critic_loss.backward()         # 一次 backward 同时算出两个网络的梯度
        self.q_optimizer.step()        # 一次 step 同时更新两个网络

        # ============ ② 更新 Actor（补丁②：每 POLICY_DELAY 步才做一次）============
        if self.total_it % POLICY_DELAY == 0:
            # Actor 损失只用 Q1（论文做法）。min(Q1,Q2) 已经用在目标 y 上压高估了，
            # Actor 再用 min 会让梯度更保守、学得更慢。
            
            pred_actions = self.actor(states)                          # a = μ(s)，带梯度
            actor_loss   = -self.q1(states, pred_actions).mean()       # 取负号 → 最小化 = 最大化 Q
            self.actor_optimizer.zero_grad()
            actor_loss.backward()     # 梯度经 ∂Q1/∂a · ∂μ/∂θ 自动回传到 actor
            self.actor_optimizer.step()

            # ============ ③ 软更新目标网络（跟着 Actor 一起延迟更新）============
            # 为什么目标网络也要延迟：Actor' 变了 Q' 的输入就变了，
            # 让“Actor 改进”和“目标跟随”保持同一节奏，y 才不会一边稳一边动。
            #TD3 延迟 Actor 的目的是：Actor 不动的那一步，Critic 多学一次，让∇aQ 更稳；
    
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
agent = TD3Agent()     # 创建智能体（6 个网络、2 个优化器、经验池全在里面）
episode_rewards = []   # 记录每个回合的总回报（用于绘图）

print("===== TD3 开始训练（Pendulum-v1）=====")
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
            # 正常阶段：a = μ(s) + 探索噪声（这里用的是 EXPL_NOISE，不是补丁③的噪声）
            action = agent.select_action(state, noise_std=EXPL_NOISE)

        # ---------- 2. 执行动作 ----------
        # 归一化动作 × 2 → 真实扭矩 ∈ [-2,2]（环境内部还会自己 clip 一次）
        next_state, reward, terminated, truncated, _ = env.step(action * ACTION_SCALE)
        done = terminated or truncated   # 任一为 True 即认为本回合结束

        # ---------- 3. 存入经验池（存的是【加过噪声的行为动作】，不是 μ(s)） ----------
        agent.store_transition(state, action, reward, next_state, done)

        # ---------- 4. 学一步（池不满 BATCH_SIZE 条时自动跳过） ----------
        # Critic 每步都学；Actor 每 POLICY_DELAY 步才学一次（在 update() 内部判断）
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
# 测试只需要 actor（策略网络）；两个 Q、三个目标网络都只是训练时的“脚手架”
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "td3_pendulum.pth")
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
plt.title("TD3 Training Curve (Pendulum-v1)")
plt.legend()
plt.show()
env.close()

# ============================================================================
# 【理解笔记】TD3 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体（对象准备）
#   agent = TD3Agent()，它自带以下固有属性：
#     - actor / actor_target             ：策略网络 μ 与目标版本 μ'
#     - q1 / q2 / q1_target / q2_target  ：两个 Q 与它们的目标版本（补丁①的载体）
#     - buffer                           ：经验回放池（容量 10 万，存【加过噪声】的行为动作）
#     - 两个优化器                       ：actor_optimizer / q_optimizer（后者管 Q1+Q2）
#     - total_it                         ：补丁② 的计数器
#   一共 6 个网络。对比 DDPG 4 个、DQN 2 个。
#
# 第二阶段：回合循环（外层 for ep in range(500)）
#   1. env.reset() → 初始状态（Pendulum 初始角度随机）
#   2. 步循环（内层 while not done，每回合固定 200 步）每步做四件事：
#      ① 选动作：总步数 < 5000 → 均匀随机；之后 a = μ(s) + N(0, 0.1)，clip 到 [-1,1]
#      ② 执行动作：归一化动作 × 2 → 真实扭矩 → env.step
#      ③ 存经验：push(s, a, r, s', done)（a 是加噪后的行为动作）
#      ④ 学一步：agent.update()（池不满 256 条时跳过），内部顺序为
#         a. Critic（每步）：y = r + γ(1-d)·min(Q1',Q2')(s', ã)，ã = clip(μ'(s')+ε, -1, 1)
#            loss = MSE(Q1,y) + MSE(Q2,y)，两个 Q 一起更新
#         b. 若 total_it % 2 == 0 才继续：
#            · Actor：最小化 -Q1(s, μ(s))
#            · 软更新：三个目标网络各挪 τ=0.005
#   3. 回合收尾：记录总回报；每 20 回合打印；每 50 回合评估
#
# 第三阶段：保存 actor 权重 + 画学习曲线
#
# ============================================================================
# 【三个补丁分别对应哪个公式】（背下来这张表，TD3 就通了）
#   补丁① Clipped Double Q  → y 里的 min(Q1', Q2')
#   补丁② Delayed Update    → if self.total_it % POLICY_DELAY == 0
#   补丁③ Target Smoothing  → ã = clip(μ'(s') + clip(ε,-c,c), -a_max, a_max)
#   —— 三个补丁【没有一个是新公式】，全是在 DPG 定理那两行上做“数值保护”。
#
# ============================================================================
# 【和前三个算法的对照】
#   - 目标 y：DQN 用 max；DPG/DDPG 用 Q'(s',μ'(s'))；TD3 用 min(Q1',Q2')(s', ã)
#   - 网络数：DQN 2 个；DDPG 4 个；TD3 6 个（多一个 Q 和一个 Q'）
#   - 更新频率：都每步更新 Actor；TD3 每 2 步才更新 Actor 与目标网络
#   - 探索噪声：DQN ε-贪婪；DPG/DDPG/TD3 外挂高斯；SAC 最大熵
#   - 目标噪声：只有 TD3 有（补丁③）；它不出现在行为动作上，只出现在目标动作上
#   - 高估处理：DQN 单 Q；DDQN 选评解耦；SAC/TD3 双 Q 取 min
#
# 【TD3 修了什么、还留着什么】
#   修了：短板②（高估 → 双 Q 取 min）+ 短板③（Q 尖峰 → 目标平滑），
#         并用延迟更新防止 Actor 追还没估准的 ∇a Q
#   还留着：探索噪声仍需手调（EXPL_NOISE 是常数）；
#           目标噪声 σ、截断 c、延迟步数 POLICY_DELAY 也是手调超参。
#           → 这正是 SAC 要解决的问题：用最大熵 + 自动温度 α 把“探索强度”交给算法自己学
# ============================================================================
