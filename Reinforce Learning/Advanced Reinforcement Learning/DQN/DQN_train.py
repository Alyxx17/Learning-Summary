#DQN的最基础代码
#结尾有整体框架的运行流程

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

# =================== 超参数 ===================
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
set_seed(SEED)   

# =================== Q 网络定义===================
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

# =================== 经验回放池 ===================
class ReplayBuffer:
    """存储 (s, a, r, s', done) 并支持随机采样"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        #( (s1, s2, ..., s128),     # 所有状态
        #(a1, a2, ..., a128),     # 所有动作
        #(r1, r2, ..., r128),     # 所有奖励
        #(s1', s2', ..., s128'),  # 所有下一状态
        #(d1, d2, ..., d128) )    # 所有结束标志
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
        #(128, 4)，(128,)，(128,)，(128, 4)	，(128,)

    def __len__(self):
        return len(self.buffer)

# =================== DQN 智能体 ===================
class DQNAgent:
    def __init__(self):
        self.q_net   = QNetwork(state_dim, action_dim).to(DEVICE)       # 当前 Q 网络
        #负责决策，也负责被更新
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)    # 目标 Q 网络
        #用于计算目标Q值得maxQ(s',a')，平时不更新
        self.target_net.load_state_dict(self.q_net.state_dict())        # 初始化时同步参数
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer  = ReplayBuffer(BUFFER_SIZE)# 新建一个容量 10000 的经验池
        self.epsilon = EPSILON_START# ε 初始为 1.0，开始全是随机探索

    def select_action(self, state, eval_mode=False):
        """ε-贪婪策略选择动作。eval_mode=True 时关闭探索 (仅测试用)"""
        if eval_mode:
            with torch.no_grad():#关闭自动求导
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                #unsqueeze(0) 就是在行插一个维度：(4,)  ──→  (1, 4)  
                return self.q_net(state_t).argmax().item()
            #等价于 self.q_net.forward(state_t)
            #输入 (1, 4)，经过网络（4→128→128→2），
            # 输出形状 (1, 2)——这一行里就是两个动作的 Q 值估计：
            #Q(s, 左)  Q(s, 右)[  0.32,   0.87 ]     ← 网络输出
            #.argmax()：返回最大值的下标。上面这个例子返回 1（右）。这正好就是"选 Q 值最大的动作"。
            #item()：把只含一个数的张量转成普通 Python 数字。
            # .argmax() 返回的是一个 0 维张量(形状 ())，.item() 把它变成 int 1，这样才能返回给 env.step() 使用。
            #标量（0维）：一个单一的数字，例如 5。
            # 向量（1维）：一排数字，例如 [1, 2, 3]。
            # 矩阵（2维）：一个由行和列组成的表格。
            #argmax不是np.argmax，而是torch.argmax()，张量.max可以省略torch
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
            return#等于return None，立即退出update

        # ---- 采样一个 mini-batch ----
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states      = torch.FloatTensor(states).to(DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)   # [B, 1]
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)  # [B, 1]
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)    # [B, 1]
        # ---- 当前 Q 值 (选用实际执行的动作) ----
        q_values = self.q_net(states).gather(1, actions)                  # [B, 1]
        #        Q(s,左)  Q(s,右)
            #样本0  [ 0.32,   0.87 ]
            #样本1  [ 0.11,  -0.45 ]
            #样本2  [ 0.90,   0.02 ]
            #...        ...
            #样本127 [ ... ]
        #这一行里我们只需要"实际执行的那个动作"对应的 Q 值。比如样本 0 实际执行了动作 1（右），
        # 我们只要 0.87，不要 0.32。    
        #gather(1, actions) 的含义：在**第 1 维（列方向）**上，对每一行，取出 actions 指定的那一列。
        #结果 q_values 形状 (128, 1)：128 个样本各自"实际执行动作"的 Q 值。
        #相当于表格法Q[s][a] 

        # ---- 目标 Q 值 (使用目标网络计算 max_a' Q(s', a')) ----
        with torch.no_grad():#目标网络不需要反向传播更新权重，因此关闭梯度，不记录计算图
            #而上面的q_values需要进行反向传播更新权重，要开梯度，记录计算图
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0] # [B, 1]
            #dim=1：沿着列方向压缩（对每一行取 max）。
            #keepdim=True：保留维度，输出 (128, 1) 而不是 (128,)
            #[0]取值，而非索引
            #这边是取max_a' Q(s', a')，所以不采用gather，而是采用max
            #相当于DQN的np.max(Q[buffer[-1][3][0], buffer[-1][3][1], :])
            #max()是torch.max(),张量.max可以省略torch
            q_target = rewards + GAMMA * max_next_q * (1 - dones)                   # [B, 1]
        #表格中的更新公式为：Q(s,a)←Q(s,a)+a[r+gamma*Q(s',a')-Q(s,a)]
        #本质上就是对平方损失做梯度下降，可见学习笔记
        # ---- 均方误差损失 + 反向传播 ----
        loss = nn.MSELoss()(q_values, q_target)
        self.optimizer.zero_grad()#清空上次的梯度
        loss.backward()#反向传播，自动算出 loss 对每个参数的偏导数
        # 梯度裁剪，提升稳定性 (可选)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()#更新权重，

        # ---- 探索率衰减 ----
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def sync_target_net(self):
        """将当前 Q 网络的权重复制到目标网络 (硬更新)"""
        self.target_net.load_state_dict(self.q_net.state_dict())

# =================== 训练循环 ===================
agent = DQNAgent()## 创建智能体（两个网络、优化器、经验池、ε 全在里面）
episode_rewards = []           # 记录每个回合的总奖励 (用于绘图)

print("===== DQN 开始训练 =====")
env.reset(seed=SEED)        # 给环境的随机数发生器播种（只播种一次）
                            # 之后每次 env.reset() 会按确定序列给出初始状态，保证可复现
for ep in range(EPISODES):
    state, _ = env.reset()#_:info：附加信息字典（这里用不上）。
    #这个reset并非固定，而是在一定范围内随机选取，获得4个随机状态
    total_reward = 0## 本回合累计奖励清零
    done = False

    while not done:#只要回合没结束就继续走
        # 1. 选择动作
        action = agent.select_action(state)

        # 2. 执行动作
        # env.step(action) 在 gymnasium 中返回 5 个值：
        #   next_state : 执行动作后到达的新状态 (4 维数组)
        #   reward     : 这一步得到的奖励 (CartPole 每步存活 +1)
        #   terminated : 是否"任务失败/达成"而结束 (如杆子倒下) → 真回合结束
        #   truncated  : 是否"超时/达步数上限"被强制截断 (如撑过 500 步) → 只是时间到
        #   _ (info)   : 附加信息字典，这里用不上
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated   # 任一为 True 即认为本回合结束
        #terminated:杆子倾角超过阈值，或者小车滑出边界
        #truncated：不满足上述的条件下，称到第500步

        # 3. 存入经验池
        agent.store_transition(state, action, reward, next_state, done)

        # 4. 更新网络 (在线更新，每步都学)
        agent.update()
        #即使前一步已经done了，这一步也要再更新，然后再退出循环
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
torch.save(agent.q_net.state_dict(), "dqn_cartpole.pth")
print("模型已保存为 dqn_cartpole.pth")


# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Curve")
plt.show()
env.close()


# ============================================================================
# 【理解笔记】DQN 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体（对象准备）
#   agent = DQNAgent()，它自带以下固有属性：
#     - q_net      ：当前网络，负责决策、且被更新
#     - target_net ：目标网络，专门算目标值里的 max Q(s',a')，平时冻结，每10回合被复制刷新
#     - optimizer  ：Adam 优化器，用梯度去更新 q_net 的权重
#     - buffer     ：经验回放池（容量10000），用来存经验
#     - epsilon    ：探索率，初始 1.0，逐渐衰减到 0.01
#   提供的方法：选择动作(ε-贪婪)、存放经验、更新当前网络、同步目标网络
#
# 第二阶段：回合循环（外层 for ep in range(600)）
#   1. env.reset() 重置环境 → 得到本回合的【随机】初始状态（每次都不固定！）
#   2. 初始化 total_reward = 0（本回合累计奖励清零）、done = False
#
# 第三阶段：步循环（内层 while not done，每步做四件事）
#   只要 done == False 就持续执行：
#     1. 选动作   ：agent.select_action(state)，按 ε-贪婪选择
#     2. 执行动作 ：把动作送进 env.step(action)（gym 自带），
#                   返回 next_state、reward、terminated、truncated，合并成 done
#     3. 存经验   ：把 (s, a, r, s', done) 存入经验池
#     4. 学一步   ：agent.update()
#                   - 经验池不足 128 条 → return 跳过，这一步不学习
#                   - 够了 → 随机抽 128 条样本，用这批样本更新一次 q_net 权重
#     【关键】第 4 步无论本步 done 与否都会执行（哪怕回合最后一步也会先学一次）；
#             done 只用来决定 while 是否继续。
#     5. 推进状态 ：state = s'，并把本步奖励累加进 total_reward
#
# 第四阶段：回合收尾（步循环结束后）
#   1. 检查是否到了同步目标网络的回合（ep 是 10 的倍数）：
#      到了 → 把 q_net 的权重复制给 target_net；没到 → 不动
#   2. 记录本回合总奖励 total_reward
#
# 然后进入下一个回合，直到 600 个回合全部跑完。
# ============================================================================