# ============================================================
# Rainbow A2C（Actor-Critic）训练代码 —— LunarLander-v3
# 组合：GAE 优势 + 并行环境 + 共享网络 + 熵正则 + 优势标准化
#   （对应笔记章节二：2.3 ~ 2.10）
# 与 REINFORCE 的本质区别：
#   1. 多一个 critic 网络（学 V(s)），作为"每步可算"的精准基线
#   2. 优势用 GAE（λ 加权 TD 误差），不再用 MC 的 G_t
#   3. 并行 N 个环境同时采样，每 T 步更新一次（样本效率↑）
# 结尾有整体框架的运行流程
#问题：超参难以调整，训练难以收敛，模型倾向于次优，即悬浮而非着陆。
#可以考虑奖励重塑，但效果不大，奖励重塑部分可以作为别的项目参考。
#此外还有 potential-based shaping，是一种理论上有保证不改变最优策略的奖励塑形方法
#训练出来的模型是次优悬浮
# ============================================================

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from set_seed import set_seed

# =================== 环境配置（并行 N 个） ===================
N_ENVS = 16     # 并行环境数（同时跑 8 个 LunarLander，互不相关 → 梯度更稳）
# gymnasium 批量环境：一次 reset/step 同时操作 N 个环境
envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("LunarLander-v3") for _ in range(N_ENVS)]
)
#lambda: 等同于 def func():
# 延迟创建游戏环境，防止并行冲突。
## ❌ 错误写法
#envs = gym.vector.SyncVectorEnv([gym.make("LunarLander-v3") for _ in range(N_ENVS)])
#这会导致程序立刻创建出 N_ENVS 个环境实例，并将同一个或已初始化的实例直接传给向量化容器。
# 在多线程或并行环境中，这会引发资源冲突、内存报错或状态覆盖。
#lambda 是用来定义匿名函数（没有名字的函数）的关键字。
# 它能让你在只用一行代码的简短情况下，快速写出一个功能单一的函数。

# LunarLander：状态 8 维 (位置x,y、速度、角度、角速度、两脚接触)，动作 4 个 (无/左喷/主喷/右喷)
state_dim  = envs.single_observation_space.shape[0]   # 8
#envs 是一个向量化环境（SyncVectorEnv），它同时管理 N 个相同的环境
#single_observation_space 表示单个环境的观察空间（而不是整个向量化环境的观察空间）
#对于 LunarLander-v3，单个环境的观察空间是一个 Box，表示一个连续的 8 维向量（例如位置、速度等）
#.shape 返回这个 Box 的形状，是一个元组 (8,)
#[0] 取元组的第一个元素，也就是 8。
action_dim = envs.single_action_space.n               # 4

# =================== 超参数 ===================
ROLLOUT_STEPS = 64     # T：每次更新前滚动收集多少步（每步同时收集 N 个环境）。
UPDATES       = 2000    # 总更新次数（N×T=80 步/次，共约 32 万步；T 翻倍后总步数保持不变）
GAMMA         = 0.99     # 折扣因子
LAM           = 0.95     # GAE 的 λ（连续调节方差-偏差，λ=0 一步TD，λ=1 MC）
LR            = 1e-4     # 学习率
ENTROPY_COEF  = 0.03     # 熵正则系数 β（笔记 1.6，防策略坍缩）
VALUE_COEF    = 0.5      # critic 损失权重 c1（笔记 2.10）
#SHAPE_COEF    = 1     # 奖励重塑系数（引导奖励叠加进原始奖励的权重，可调）
EVAL_INTERVAL = 100      # 每多少次更新评估一次
EVAL_EPISODES = 10       # 每次评估跑几局
SEED          = 42       # 随机种子
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =================== 共享网络（actor + critic） ===================
class ActorCriticNet(nn.Module):
    """actor 和 critic 共享底层特征提取层（类比 Dueling 的共享底层）
    输入状态 → 共享层(256-256) → 头A(动作概率 logits) + 头V(状态价值)
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(256, action_dim)   # → 各动作 logits
        #看起来很像 Dueling DQN 的 advantage 分支，但它本质上不是 A(s,a)，而是策略参数。
        self.critic_head = nn.Linear(256, 1)            # → V(s)输出形状 [batch, 1]

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor_head(features)              # [batch, action_dim]
        value  = self.critic_head(features).squeeze(-1) # [batch]
        #squeeze(-1)作用是移除形状中长度为 1 的维度。
       #-1 表示从最后一个维度开始检查，如果该维度大小为 1，就把它去掉
       #在compute_gae内作为标量参与计算
       #而在Dueling DQN网络中不需要是因为 PyTorch 广播机制天然支持 [B,1] + [B, action_dim]
       #即需要作为一个矩阵来参与计算
        return logits, value

# =================== 奖励重塑（reward shaping）仅供参考，实际未使用，学习原理 ===================
def shape_reward(states):
    """给"靠近着陆区、机身水平、下降放缓"加中间引导奖励，破解悬浮局部最优
    states: [N, 8]（LunarLander 观测：x, y, vx, vy, angle, angvel, leg1, leg2）
    返回: [N] 每个环境的引导奖励（用负距离 → 越接近目标值奖励越高）
    注意：只在【训练】时叠加；评估/测试用原始奖励，衡量真实表现
    """
    x, y, vx, vy, angle, angvel, leg1, leg2 = states.T   # 每行一个维度
    bonus = np.zeros(len(states))
    bonus -= np.abs(x) * 0.6   # 引导水平对准着陆区（观测 x=0 是着陆区中心，越接近 0 越好）
    bonus -= np.abs(angle)   * 0.3   # 引导机身水平（角度越接近 0 越好）
    bonus -= np.abs(vy)      * 0.5   # 引导垂直速度放缓（利于安全着陆）
    bonus -= np.abs(y)       *0.5
    return bonus    
# =================== potential-based shaping仅供参考，实际未使用，学习原理 ===================
def potential(states):
    """势函数 Φ(s)：越接近目标，势能越高"""
    x, y, vx, vy, angle, angvel, leg1, leg2 = states.T
    # 目标 x=0，角度=0，垂直速度=0
    return -1.0 * np.abs(x) - 1.0 * np.abs(angle) - 0.5 * np.abs(vy)
def potential_based_shaping(states, next_states, gamma=0.99):
    """基于势函数的奖励塑形：F = γΦ(s') - Φ(s)"""
    return gamma * potential(next_states) - potential(states)


# =================== GAE：广义优势估计（笔记 2.6） ===================
def compute_gae(rewards, values, dones, last_values, last_dones, gamma, lam):
    """输入:rewards/values/dones 形状 [T, N]（时间在前），
            last_values [N]:窗口末尾状态的价值（bootstrap），last_dones [N]：末尾是否结束
    输出:advantages [T, N]（GAE），returns [T, N]（= advantages + values，供 critic 训练）
    反向迭代:A_t = δ_t + γλ·(1-done)·A_{t+1},δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    """
    #T时间步数，即轨迹被切成的长度，在实际训练中，一条完整轨迹可能很长（甚至无限）。
    #为了批量训练，我们通常把轨迹切成长度为 T 的片段（窗口）。
    #N：并行环境数量
    #rewards：每个时刻每个环境获得的即时奖励
    #values：Critic 网络对每个时刻每个状态的价值估计
    #dones：每个时刻每个环境是否终止（1 表示该时刻结束后终止）
    #last_values：窗口末尾之后那个状态的价值估计，用于当窗口结束但环境未终止时的 bootstrap
    #last_dones：窗口末尾状态是否为终止状态

    T, N = rewards.shape#把张量rewards的维度分别赋给T,N
    #没有用到N，只是表明了 rewards 的第二个维度是并行环境数
    #为了文档化和可读性，属于无害的冗余写法
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            #T-1之后的下一个状态的价值 V(s_T) 不在窗口内，因为窗口只到 T-1。
            # 窗口末尾：用"窗口末状态"的价值做 bootstrap（若未结束）
            next_value = last_values              # [N]
            next_non_terminal = 1.0 - last_dones  # [N]
        else:#窗口内
            next_value = values[t + 1]            # [N]#下一个状态价值就是当前t+1的value
            next_non_terminal = 1.0 - dones[t + 1]#下一个状态的中止就是当前t+1的dones
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns#返回每个时间步的At(GAE)与回报
#returns 是每个时间步每个状态下，我们希望 critic 输出的价值，是某个状态的价值函数，用于更新critic网络的目标值
#注意在A2C中A=G-V，而在DQN中A=Q-V，是不一样的，可以详见笔记。

# =================== A2C 智能体 ===================
class A2CAgent:
    def __init__(self):
        self.net = ActorCriticNet(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

    def select_actions(self, states):
        """对 N 个状态同时采样动作（批量）
        返回：actions [N]、log_probs [N]、entropies [N]
        注意：这里【不能】加 no_grad！actor 损失直接使用这份 log_probs 反向传播，
        no_grad 会切断梯度路径导致 actor 完全不更新（与 PPO 不同，PPO 更新时会重新前向）
        """
        states_t = torch.FloatTensor(states).to(DEVICE)   # [N, 8]
        logits, _ = self.net(states_t)                    # [N, 4]
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy()
        #不用action.item()，因为此是[N]的动作向量，item()只适用于标量张量

    def update(self, log_probs, entropies, rewards, values, dones, last_states, last_dones):
        """一次更新：用刚收集的 N×T 数据算 GAE 并更新两个头
        loss = -Σ log π(a|s)·A(标准化) + c1·(V-return)² - c2·H
        """
        # ---- 整理形状（时间在前 [T, N]，与收集顺序一致）----
        log_probs = torch.cat(log_probs)                                            # [T*N]
        entropies = torch.cat(entropies)                                            # [T*N]
        rewards   = torch.stack([torch.FloatTensor(r).to(DEVICE) for r in rewards]) # [T, N]
        values    = torch.stack(values)                                             # [T, N]
        dones     = torch.stack([torch.FloatTensor(d).to(DEVICE) for d in dones])   # [T, N]
        with torch.no_grad():
            _, last_values = self.net(torch.FloatTensor(last_states).to(DEVICE))    # [N]
        last_dones = torch.FloatTensor(last_dones).to(DEVICE)                       # [N]

        # ---- GAE（笔记 2.6）+ 优势标准化（笔记 2.9）----
        advantages, returns = compute_gae(rewards, values, dones, last_values,
                                          last_dones, GAMMA, LAM)                   # [T, N]
        # 优势标准化：减均值除标准差，稳定梯度尺度（等价于回报标准化升级版）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 三个损失（笔记 2.10）----
        # ① actor：策略梯度（与 REINFORCE 同源，只是 A 换成 GAE 优势）
        actor_loss = -(log_probs * advantages.reshape(-1)).mean()
        #reshape(-1) 的作用是将张量展平为一维张量，其中 -1 表示“自动计算这个维度的大小”
        # ② critic：TD 误差平方（和 DQN 的 MSELoss(q, target) 同源，学 V 不学 Q）
        critic_loss = ((values.reshape(-1) - returns.reshape(-1)) ** 2).mean()
        # ③ 熵正则（笔记 1.6）：鼓励探索、防策略坍缩

        #reshape(-1) 的核心作用是自动计算并展平（或调整）数组的维度
        #array.reshape(-1)：将任意维度（如 2D、3D）的多维数组转换为一维数组（展平）， 数据按行顺序排列
        # array.reshape(-1, n)：将数组自动调整为 未知行数、固定 n 列 的二维矩阵。
        # array.reshape(n, -1)：将数组自动调整为 固定 n 行、未知列数 的二维矩阵。
        entropy = entropies.mean()

        total_loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()

# =================== 评估函数 ===================
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
agent = A2CAgent()
print(f"===== Rainbow A2C 开始训练LunarLander-v3, N={N_ENVS}, T={ROLLOUT_STEPS}=====")
envs.reset(seed=SEED)

states, _ = envs.reset()   # 初始状态 [N, 8]

#每次更新都是每个环境固定推进 5（ROLLOUT_STEPS） 步，不管这些步跨越了几个回合。

#比如第一次更新：
#环境1：取当前回合的 step 1~5
#环境2：如果它在 step 3 结束，向量环境会自动重置，接着提供新回合的 step 1~2（补齐 5 步）
#环境3：取当前回合的 step 1~5
#第二次更新:
#环境1：取当前回合的 step 6~10
#环境2：取它当前所在位置的接下来 5 步（可能是新回合的 step 3~5 + 再新回合的 step 1~2）
#环境3：取当前回合的 step 6~10
#若某个环节内小于5步就done，会拿下回合的步数补充，直至满足5步，由于有done截止，不会跨回合统计

for update in range(UPDATES):
    # ---------- 1. 滚动收集 N×T 步数据 ----------
    batch_log_probs, batch_entropies = [], []
    batch_rewards, batch_values, batch_dones = [], [], []

    for _ in range(ROLLOUT_STEPS):
        actions, log_probs, entropies = agent.select_actions(states)   # [N]×3
        with torch.no_grad():
            _, values = agent.net(torch.FloatTensor(states).to(DEVICE))  # [N]

        next_states, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
        #actions 是 PyTorch 张量
        # 而 Gym 环境是 CPU 上的 Python 程序，它要求动作输入是 numpy 数组，而不是 PyTorch 张量。
        dones = terminated | truncated

        # ---- 奖励重塑：在原始奖励上叠加引导奖励（基于动作后的新状态）----
        # 让策略有"往着陆区走"的梯度，而不是满足于悬浮保命
        #rewards = rewards + SHAPE_COEF * shape_reward(next_states)   # [N]
        #---- 奖励重塑：potential-based shaping
        #rewards = rewards + SHAPE_COEF * potential_based_shaping(states, next_states, GAMMA)

        batch_log_probs.append(log_probs)#[T*N]
        batch_entropies.append(entropies)#[T*N]
        batch_rewards.append(rewards)#形状[T,N]，每次append N个数据，这N个数据已经自动[数据]
        batch_values.append(values)#[T*N]
        batch_dones.append(dones)#同rewards

        states = next_states   # 结束的环境由 vector env 自动 reset（autoreset）即自动下一回合补足5个

    last_dones = dones   # 窗口末尾每个环境是否结束（GAE bootstrap 用）

    # ---------- 2. 一次更新 ----------
    agent.update(batch_log_probs, batch_entropies,
                 batch_rewards, batch_values, batch_dones,
                 states, last_dones)

    # ---------- 3. 定期评估 ----------
    if (update + 1) % EVAL_INTERVAL == 0:
        avg_reward = evaluate(agent)
        print(f"更新 {update+1:5d} | 评估平均回报: {avg_reward:7.2f} （≥200 可认为学会）")

print("===== 训练完成 =====")

#===================保存模型 ===================
# 保存整个共享网络（含 actor + critic），测试时只取 actor 头
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2c_lunarlander.pth")
torch.save({
    'model_state_dict': agent.net.state_dict(),
    # 注意：state_dim/action_dim 是 numpy 整数（shape[0] 返回 np.int64），
    # 必须用 int() 转成 Python 整数，否则 PyTorch 2.6+ 的 torch.load 默认
    # weights_only=True 会拒绝加载含 numpy 标量的文件（报 UnpicklingError）
    'hyperparams': {'state_dim': int(state_dim), 'action_dim': int(action_dim), 'HIDDEN': 256},
}, SAVE_PATH)
print(f"模型已保存为 {SAVE_PATH}")

envs.close()


# ============================================================================
# 【理解笔记】Rainbow A2C 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体与环境
#   agent = A2CAgent()：一个共享网络（actor 头 + critic 头）+ 一个优化器
#   envs  = SyncVectorEnv(...)：并行 N=8 个 LunarLander，一次操作一批
#
# 第二阶段：更新循环（外层 for update in range(UPDATES)）
#   1. 滚动收集 N×T 步数据（内层 for _ in range(ROLLOUT_STEPS)）：
#      - 对 N 个状态批量采样动作（actor）
#      - 用 critic 批量估计 V(s)（no_grad）
#      - envs.step 批量推进环境，记录 log_prob/entropy/reward/value/done
#      - 结束的环境由 vector env 自动 reset，继续采样
#   2. 一次更新（agent.update）：
#      a. 把数据排成 [T, N]（时间在前）
#      b. 反向迭代算 GAE 优势（compute_gae，窗口末尾用 last_values 做 bootstrap）
#      c. 优势标准化
#      d. 三个损失相加，一次反向传播更新全部参数（共享层+两个头）
#   3. 定期用单环境评估（argmax 贪婪）并打印平均回报
#
# 【与 REINFORCE 的对比】
#   - REINFORCE：单环境，一整个回合结束才更新一次（用 MC 的 G_t）
#   - A2C      ：并行 N 环境，每 T 步就更新一次（用 GAE 的 A_t，方差更小）
#   - 关键新增：critic 网络（学 V(s)，同时当基线）、GAE、并行环境
# ============================================================================
