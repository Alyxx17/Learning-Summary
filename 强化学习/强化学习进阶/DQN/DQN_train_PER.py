# ============================================================
# Prioritized Replay (PER) DQN 训练代码 —— 第五阶段（最终整合）
# 基础栈：Dueling + Double + N-step + Noisy Net（与上一阶段相同）
# 本阶段新增：
#   ① 优先经验回放（PER）：按 |TD误差| 加权抽样，用重要性采样修正偏差
#   ② 存档/续训：每 50 回合自动存档；Ctrl+C 可暂停保存；下次运行自动续训
# 涉及改动：
#   - 新增 SumTree 类（按优先级抽样的数据结构）
#   - ReplayBuffer 改为 PER 版本（保留 N 步 + 优先级 + 重要性采样）
#   - update() 用加权损失，并更新每条经验的优先级
#   - 训练循环加入存档/读档逻辑
# ============================================================
import os
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
env = gym.make("LunarLander-v3", continuous=False)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# =================== 超参数 ===================
EPISODES        = 600
BUFFER_SIZE     = 50000
BATCH_SIZE      = 128
GAMMA           = 0.99
N_STEP          = 3
LR              = 1e-3
TARGET_UPDATE   = 10
SEED            = 42
# --- PER 相关超参数 ---
PER_ALPHA       = 0.6          # 优先级指数：0=均匀抽样，1=完全按优先级
PER_BETA_START  = 0.4          # 重要性采样指数初值（训练中退火到 1.0）
PER_EPS         = 1e-3         # 优先级小常数：避免优先级为 0 的经验永远抽不到
# --- 存档/续训 ---
CHECKPOINT_PATH = "checkpoint_per.pth"   # 存档文件名
SAVE_INTERVAL   = 50           # 每多少回合自动存档一次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

# =================== Noisy 线性层（与上一阶段相同） ===================
class NoisyLinear(nn.Module):
    """带可学习噪声的线性层：y = (μ_W + σ_W⊙ε_W)x + (μ_b + σ_b⊙ε_b)
    训练时采样噪声，测试时只用均值 μ"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu   = nn.Parameter(torch.empty(out_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(0.5 * std)
        self.bias_sigma.data.fill_(0.5 * std)

    def forward(self, x):
        if self.training:
            noise_w = torch.randn_like(self.weight_mu)
            noise_b = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + self.weight_sigma * noise_w
            bias   = self.bias_mu   + self.bias_sigma   * noise_b
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)

# =================== Q 网络定义（Dueling + Noisy，与上一阶段相同） ===================
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

# =================== 求和树（SumTree）：按优先级抽样的数据结构 ===================
class SumTree:#替代了原来的经验池，现在每一条经验都是叶子，组合成二叉树的结构
    #当前BUFFER_SIZE= 50000，下标49999 到 99998（50000个）是经验的下标（叶子）
    #0 到 49998为内部节点（49999个，含根）仅仅是存放下面子节点的分数之和（汇总数据），
    #不是任意一个具体经验(叶子)的优先级
    """用数组实现的二叉树：
       叶子节点存经验与优先级，父节点存两个子节点之和。
       抽样时在 [0, total) 取随机数，从根向下走到叶子 → O(log N) 完成按权重抽样"""
    #属性：树结构
    #方法：加入新经验，更新各节点的数据，获取叶子（经验），获取根的数据
    def __init__(self, capacity):
        self.capacity = capacity#树能容纳的最大经验条数（同时也是叶子节点数）。
        self.data = [None] * capacity            # 叶子：经验数据
        self.tree = [0.0] * (2 * capacity - 1)   # 节点：优先级为浮点数
        #叶子节点的数量是 capacity（用来存经验），
        # 内部节点的数量是 capacity - 1（用来存父子节点的和）。根也是内部节点
        #一个节点左右2个叶子(叶子指的是无法在往下分的，节点可以往下分)
        #总节点数 = 叶子数 + 内部节点数 = capacity + (capacity - 1) = 2 * capacity - 1。
        self.size = 0                            # 记录当前树里实际存了多少条有效的经验数据。
        self.write = 0                           # 循环写入位置，范围是 0 到 capacity-1

    #把新经验丢进池子，并更新树上的分数
    def add(self, priority, data):
        idx = self.write + self.capacity - 1     # 叶子下标 = 数据下标 + capacity - 1
        #capacity - 1是第一个叶子节点的下标，前面的都是节点的下标
        self.data[self.write] = data#self.write是经验列表的下标
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity#+1，当存满时到0，把最旧的数据覆盖循环
        self.size = min(self.size + 1, self.capacity)
        #池子满了后，有效条数不再增加，永远卡在BUFFER_SIZE


    def update(self, idx, priority):
        change = priority - self.tree[idx]       # 优先级变化量，只是为了self.tree[idx] += change，
        #计算上方节点的变化
        #self.tree[idx] 是这片叶子原来的旧分数，
        # priority 是外界传进来的新分数。
        self.tree[idx] = priority#把旧分数覆盖成新分数。
        while idx > 0:                           # 沿路径向上传播到根
            idx = (idx - 1) // 2                #求父节点的运算，根为1时，idx=idx//2,这里为0
            self.tree[idx] += change            #上面若idx=0时，这一步已经把根更新了，再判断条件停止循环

    def get(self, value):#每次只能取到一个叶子
        """value ∈ [0, total)，从根向下走到对应的叶子，返回 (叶子下标, 优先级, 经验)"""
        idx = 0
        while idx < self.capacity - 1:#说明还没走到叶子，capacity - 1是第一个叶子节点的下标
            left = 2 * idx + 1#数组二叉树的固定公式：节点 idx 的左子下标就是 2 * idx + 1。
            #右子下标就是left+1
            #如果value小于等于左子树和，进入左子树
            #否则把value减去左子树和，得到新value，进入右子树
            #value减去左子树是为了减去偏移量，原先的value是针对这一层的，
            #假设左子树是0.3，右边是0.4，在该层视角下，范围是0-0.7，相当于左子树范围0-0.3，右子树0.3-0.7
            #value=0.35时，进入右边，在右子树的视角下，范围是0-0.4，这个右子树下又有2个分支节点，左边假设是0.1，右边就是0.3
            #但是value=0.35是相对于范围为0.7而言的，因此需要减去左边的0.3，才是右子树的视角下的value
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx, self.tree[idx], self.data[idx - (self.capacity - 1)]#叶子的树结构下标，具体优先级分数，具体经验

    @property
    #本质上，它只是一个语法糖（让代码更好看的写法）
    #如果没有，total_value = sumtree.total()   # 注意这里有括号，是个方法调用
    #有了后，total_value = sumtree.total     # 注意这里没有括号，像访问普通变量一样
    #这个值只能读，不能写（不用写()，相当于不能设置其内部的参数等）
    def total(self):
        return self.tree[0]                      # 根节点 = 总优先级之和

# =================== 经验回放池（PER + N 步，本阶段核心改动） ===================
class ReplayBuffer:#回放池是一个树结构，同时也是列表保存了其所有经验(self.data)
    """优先经验回放：按 |TD误差| 优先级抽样，并用重要性采样权重修正偏差；保留 N 步"""
    #属性：树，n步队列
    #方法：经验（n步）入树，清空n步队列，采样，更新优先级，β退火，存档
    def __init__(self, capacity, n_step=N_STEP, gamma=GAMMA, alpha=PER_ALPHA, beta=PER_BETA_START):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.n_step_buffer = deque()
        self.max_priority = 1.0    # 新经验默认用当前最大优先级（保证至少被抽到一次）

    # ---- N 步产出逻辑（与第三阶段相同）----
    def push(self, state, action, reward, next_state, terminated):
        self.n_step_buffer.append((state, action, reward, next_state, terminated))
        if terminated:
            G = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(len(self.n_step_buffer)))
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            self._add(s0, a0, G, next_state, True)
            self.n_step_buffer.clear()
        elif len(self.n_step_buffer) == self.n_step:
            G = sum((self.gamma ** i) * self.n_step_buffer[i][2] for i in range(self.n_step))
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            sN = self.n_step_buffer[-1][3]
            self._add(s0, a0, G, sN, False)
            self.n_step_buffer.popleft()

    def _add(self, s0, a0, G, sN, done):
        # 树里存的是 p^α；新经验先用当前最大优先级入树（保证先被抽到）
        #新进来的先接受TD网络的审判，看看效果，按照TD误差，调整分数
        self.tree.add(self.max_priority ** self.alpha, (s0, a0, G, sN, done))
        #self.max_priority这个最大概率并非固定不变，在update_priorities内变化，是一个单调
        #非递减的值，可见其函数的注释
    def reset_n_step(self):
        self.n_step_buffer.clear()

    def sample(self, batch_size):
        batch, indices, tree_values = [], [], []
        segment = self.tree.total / batch_size#total是池子所有经验的优先级和
        for i in range(batch_size):
            # 分层抽样：把 [0, total) 分成 batch_size 段，每段随机取一点
            # （避免高优先级经验垄断整批样本）
            #如果直接在 [0, total) 里随机取 batch_size 个数，
            # 由于高优先级经验的“地盘”特别大，很有可能随机数全部掉进同一个超级高优先级的区间里，
            # 导致抽出来的 batch 全是同一条经验（严重过拟合）。
            value = random.uniform(segment * i, segment * (i + 1))
            #强制覆盖了从最低分到最高分的整个分数区间。
            value = min(value, self.tree.total * (1 - 1e-8))   # 防止浮点越界
            #因为浮点数运算误差，value 有可能恰好等于 self.tree.total（抽到了尺子的最右端边界）。
            idx, priority, data = self.tree.get(value)
            # 防御1：极少数边界情况下可能命中"从未写入"的叶子（优先级=0、data=None），
            # 此时回退到均匀随机抽一个已写入的有效样本，避免 None 混进 batch 导致崩溃
            if data is None:
                data_idx = random.randrange(self.tree.size)   # 在已写入的 [0, size) 里随机挑一个
                leaf = data_idx + self.capacity - 1           # 换算成叶子下标
                idx, priority, data = leaf, self.tree.tree[leaf], self.tree.data[data_idx]
                #调用tree的公共属性获取其具体优先级分数，具体经验
            indices.append(idx)
            tree_values.append(priority)
            batch.append(data)
        # 抽样概率 P(i) = tree[i] / total（树里存的已是 p^α）
        probs = np.array(tree_values) / self.tree.total
        # 防御2：给概率加一个极小下界，防止出现 0^(-β) = inf 的除以零警告
        probs = np.maximum(probs, 1e-8)#np.maximum是两个数据进行比较，np.max是求自身最大
        # 重要性采样权重：w = (N·P)^(-β)，再归一化（最大权重 = 1，稳定训练）
        weights = (self.tree.size * probs) ** (-self.beta)
        weights = weights / weights.max()
        weights = np.nan_to_num(weights, nan=1.0)   # 防御3：兜底，防止 inf/nan 污染损失
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices, weights

    def update_priorities(self, indices, raw_priorities):
        """用新算出的 |TD误差| 更新对应经验的优先级（树里存 p^α，维护最大原始优先级）"""
        for idx, rp in zip(indices, raw_priorities):
            self.tree.update(idx, rp ** self.alpha)
            self.max_priority = max(self.max_priority, rp)
            #下限是self.max_priority = 1，即使训练的非常好，即rp <self.max_priority
            #也给新来的经验1的max_priority，让它大概率可以被抽到 
            #循环是为了获得这个128个样本内最大的pi，或者都小于1时，选择1
    def anneal_beta(self):
        """β 退火：每回合调用一次，从 0.4 线性升到 1.0"""
        self.beta = min(1.0, self.beta + (1.0 - PER_BETA_START) / EPISODES)

    # ---- 存档 / 读档：保存经验池内部全部状态 ----
    def get_state(self):
        return {'tree_data': self.tree.data, 'tree_tree': self.tree.tree,
                'tree_size': self.tree.size, 'tree_write': self.tree.write,
                'n_step_buffer': list(self.n_step_buffer),
                'max_priority': self.max_priority, 'beta': self.beta}

    def set_state(self, state):
        self.tree.data = state['tree_data']
        self.tree.tree = state['tree_tree']
        self.tree.size = state['tree_size']
        self.tree.write = state['tree_write']
        self.n_step_buffer = deque(state['n_step_buffer'])
        self.max_priority = state['max_priority']
        self.beta = state['beta']

    def __len__(self):
        return self.tree.size

# =================== DQN 智能体 ===================
class DQNAgent:
    def __init__(self):
        self.q_net   = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer  = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state):
        # 探索由 Noisy Net 完成，无 ε
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return self.q_net(state_t).argmax().item()

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        # ---- 采样一个 mini-batch（PER：额外返回 indices 和 IS 权重）----
        (states, actions, rewards, next_states, dones), indices, weights = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        weights     = torch.FloatTensor(weights).unsqueeze(1).to(DEVICE)   # [B,1] IS 权重

        # ---- 当前 Q 值 ----
        q_values = self.q_net(states).gather(1, actions)

        # ---- 目标 Q 值（N 步 + Double）----
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q   = self.target_net(next_states).gather(1, best_actions)
            q_target = rewards + (GAMMA ** N_STEP) * max_next_q * (1 - dones)

        # ---- 加权损失：loss = mean(w_i · (Q - target)²) ----
        td_errors = q_values - q_target                       # [B,1] 每个样本的 TD 误差
        loss = (weights * td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- 用本批 |TD误差| 更新经验优先级（TD误差大 → 优先级高 → 以后更常被抽到）----
        raw_priorities = td_errors.detach().squeeze(1).abs().cpu().numpy() + PER_EPS
        # 防御4：把 NaN/inf 兜底成安全值，防止异常优先级污染整棵求和树
        # （否则 value <= NaN 恒为 False，抽样会一路往右走到未写入的叶子）
        #NaN 就当它几乎不重要（给个极小值），Inf 就当作一个‘中等重要’（给个基础值 1.0）的样本处理
        raw_priorities = np.nan_to_num(raw_priorities, nan=PER_EPS, posinf=1.0, neginf=1.0)
        #.detach()：切断梯度追踪
        # .cpu()因为 update_priorities 里用的是普通 Python 循环和 max 函数，
        # 对 NumPy 数组或列表的兼容性更好，且不需要 GPU 参与这种简单的数据更新操作。
        #PER_EPS加上一个极小的正数（如 0.01），
        # 保证即使是表现完美的经验，也有微小的概率被再次抽到，继续参与训练。
        self.buffer.update_priorities(indices, raw_priorities)

    def sync_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# =================== 存档 / 读档 ===================
#agent只是形参，顶格定义，并非类的方法
def save_checkpoint(agent, ep, episode_rewards):
    """保存完整训练状态：两个网络 + 优化器 + 经验池（含优先级树）+ 进度"""
    checkpoint = {
        'ep': ep,                                  # 已完成的回合数
        'episode_rewards': episode_rewards,        # 历史奖励（续训后接着画图）
        'q_net': agent.q_net.state_dict(),         #Q网络参数
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(), # Adam 动量等（精确续训必需）
        'buffer': agent.buffer.get_state(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)#torch.save 将 Python 字典序列化为 .pth 文件。
    print(f"  [存档] 已保存到 {CHECKPOINT_PATH}（当前进度：第 {ep} 回合）")

def load_checkpoint(agent):
    """检测到存档则恢复全部状态，返回 (已完成的回合数, 历史奖励)"""
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, []
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    agent.q_net.load_state_dict(checkpoint['q_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.buffer.set_state(checkpoint['buffer'])
    return checkpoint['ep'], checkpoint['episode_rewards']

# =================== 训练循环（支持续训 + Ctrl+C 暂停存档） ===================
agent = DQNAgent()

# 启动时尝试读档续训
start_ep, episode_rewards = load_checkpoint(agent)
if start_ep > 0:
    print(f"检测到存档：从第 {start_ep} 回合继续训练（共需 {EPISODES} 回合）")
else:
    episode_rewards = []
    print("无存档，从头开始训练")

print("===== PER DQN 开始训练（Ctrl+C 可暂停并保存存档）=====")
env.reset(seed=SEED)
try:
    for ep in range(start_ep, EPISODES):
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

        agent.buffer.anneal_beta()      # β 退火（每回合 +1 步）
        episode_rewards.append(total_reward)

        # 自动存档（每 SAVE_INTERVAL 回合）
        if (ep + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(agent, ep + 1, episode_rewards)

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"回合 {ep+1:4d} | 平均奖励(近50回合): {avg:.2f} | beta: {agent.buffer.beta:.2f}")
except KeyboardInterrupt:
    print("\n[暂停] 正在保存存档……")
    save_checkpoint(agent, ep, episode_rewards)
    print("已保存。下次运行将从中断处继续。")
    env.close()
    raise SystemExit
print("===== 训练完成 =====")

# =================== 保存最终模型 ===================
torch.save(agent.q_net.state_dict(), "dqn_LunarLander_per.pth")
print("模型已保存为 dqn_LunarLander_per.pth")

# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PER DQN Training Curve")
plt.show()
env.close()
