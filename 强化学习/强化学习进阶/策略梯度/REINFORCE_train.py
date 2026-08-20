# ============================================================
# REINFORCE（蒙特卡洛策略梯度）训练代码 —— 策略梯度家族最基础的算法
# 与 DQN 的本质区别：
#   1. DQN 学 Q 值 + argmax 决策（确定性）；REINFORCE 直接学策略 π(a|s)，按概率采样（随机策略）
#   2. DQN 是 off-policy（经验回放池复用旧数据）；REINFORCE 是 on-policy（轨迹用完即弃，没有经验池）
#   3. DQN 有目标网络；REINFORCE 没有
# 更新核心：loss = -Σ_t log π(a_t|s_t) · G_t（对目标函数 J(θ) 做梯度上升，负号转成最小化）
#loss并不是传统监督学习里的预测误差，而DQN的 loss 是一个时序差分回归误差(
#传统监督学习中是静态目标，DQN是动态目标，若干回合会复制变化，也就是自举)
#（REINFORCE)它不关心预测某个数值是否准确，而是直接根据实际获得的回报Gt来调整动作的概率：
#如果某个动作带来了高回报Gt>0,就增大它的 log 概率；反之就降低。
#深度学习优化器，比如 Adam、SGD，都是最小化 loss，因此加一个负号
#优化器做的是θ←θ-a▽J(θ)，加个负号就变为θ←θ+a▽J(θ)相当于最大化

# 结尾有整体框架的运行流程
# ============================================================

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from set_seed import set_seed   # 从 set_seed 模块导入 set_seed 函数（直接调用函数，而不是调用模块）

# =================== 环境配置 ===================
env = gym.make("CartPole-v1")         # 经典平衡车环境（与 DQN 同环境，便于对比）
state_dim  = env.observation_space.shape[0]  # 状态维度：4 (位置,速度,角度,角速度)
action_dim = env.action_space.n              # 动作维度：2 (左/右)

# =================== 超参数 ===================
EPISODES = 1000       # 训练回合数
GAMMA    = 0.99       # 折扣因子
LR       = 1e-3       # 学习率
SEED     = 42         # 随机种子（固定训练过程，保证可复现）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =================== 策略网络定义 ===================
class PolicyNet(nn.Module):
    #网络结构与DQN（非noisy DQN）完全一致，
    #网络输出的数字只是一个数学结果。它的含义是由“输入到输出的映射关系”决定的，
    #而这个映射关系完全由训练目标赋予。
    #结构完全相同的神经网络，如果训练目标（损失函数或任务标签）不同，
    #它们学到的内部参数和特征表示就完全不同。
    #即使在某个特定输入下它们的数字输出结果表面上完全一致，
    # 这个输出所代表的实际含义和背后逻辑也是不一样的。
    """输入状态，输出每个动作的概率（离散动作 → softmax)
    注意与 DQN 的 QNetwork 的区别：
    - QNetwork 输出 Q 值 → 决策用 argmax(确定性)
    - PolicyNet 输出 logits → softmax 转概率 → 采样决策（随机策略）
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)   # 输出 logits（未归一化的打分）
            #分数越高，表示网络越倾向于选这个动作，但不是概率，和可以是实数
        )

    def forward(self, x):
        return self.fc(x)                # 输出形状: [batch, action_dim]

# =================== 计算折扣回报 G_t ===================
def compute_returns(rewards, gamma):
    """输入一回合的奖励列表，返回每个时刻的折扣回报 G_t
    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    反向迭代技巧：G_T = r_T；G_t = r_t + γ·G_{t+1}
    （和 MC 控制里反向算 G 完全一样）
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):       # 从回合末尾往开头遍历
        G = r + gamma * G
        returns.insert(0, G)          # 插到最前面，保持时间顺序 [G_0, G_1, ..., G_T]
        #每算一个G就要把这个G插入returns列表的索引0位置，因为G是逆序。
        #第一次算出G=0.1，returns = [0.1]
        #第二次算出G=0.2,returns = [0.2 0.1]0.2被插入了索引0，其他数据向后移动一位，以此类推
    return torch.tensor(returns, dtype=torch.float32).to(DEVICE)#returns转为张量

# =================== REINFORCE 智能体 ===================
class REINFORCEAgent:
    def __init__(self):
        self.policy_net = PolicyNet(state_dim, action_dim).to(DEVICE)  # 策略网络
        #负责决策，也负责被更新（对比 DQN：没有 q_net/target_net 之分）
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
        #对比 DQN：没有经验回放池，因此是on-policy，轨迹用完即弃

    def select_action(self, state):
        """按概率采样动作（不是 argmax!）→ 随机策略的体现
        返回：动作 int、该动作的 log π(a|s)（训练时需要）
        注意：这里【不能】加 no_grad！loss 直接使用这份 log_probs 反向传播，
        no_grad 会切断梯度路径导致整个训练失效（与 PPO 不同）
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # (4,) → [1, 4]
        logits = self.policy_net(state_t)                           # [1, 2] logits
        dist = torch.distributions.Categorical(logits=logits)       # softmax 概率分布
        #softmax是一种算法，把logits转化为概率。
        #具体就是对logits张量内每个数取指数（保证大于0），然后除以每个数取指数的和，得到的就是每个动作的概率
        #指数函数单调递增：所以分数高的动作，概率也高，保持相对顺序,指数函数光滑，适合反向传播。
        #指数会放大分数之间的差距，让高分数动作概率更突出。
        #分母确保总和为 1。
        #torch.distributions.Categorical有2种传入：
        #传入 probs：直接给已经归一化的概率向量，如 [0.75, 0.25]。
        #传入 logits：给未归一化的分数向量，如 [0.8, -0.3]，即此处。
        action = dist.sample()                                      # 按概率采样（随机策略）
        return action.item(), dist.log_prob(action)                 # log π(a|s) 形状 [1]

    def update(self, log_probs, rewards):
        """整条轨迹的一次更新（蒙特卡洛策略梯度）
        损失：loss = -Σ_t log π(a_t|s_t) · G_t
        负号：优化器只做最小化，而我们要最大化 J(θ)，所以最小化 -J
        """
        returns   = compute_returns(rewards, GAMMA)   # [T] 每个时刻的折扣回报
        log_probs = torch.cat(log_probs)              # [T] 拼成一条轨迹的 log 概率，cat为拼接张量函数

        # ---- 蒙特卡洛策略梯度损失 ----
        # 直觉：G_t > 0 → 增大该动作概率；G_t < 0 → 减小该动作概率
        loss = -(log_probs * returns).mean()
        #单条轨迹的 REINFORCE 更新，损失为该轨迹内所有时间步的加权对数概率平均值。
        #严格来说，策略梯度的期望是对轨迹取平均，而不是对时间步取平均。
        #假设有N条轨迹，每条轨迹的 log_probs_i 和 returns_i 分别是长度Ti的张量，那么（伪代码）：
        #total_loss = 0.0
        #for log_probs, returns in zip(list_of_log_probs, list_of_returns):
        #trajectory_loss = -(log_probs * returns).sum() # 每条轨迹内部求和
        #total_loss += trajectory_loss
        #loss = total_loss / N   # 对轨迹数平均
        #但是这里是对单条轨迹的时间步平均，梯度方向不变：mean() 只是把梯度缩小了T(步数)倍
        #有效学习率随轨迹长度变化：相当于你的学习率被动态除以了T，轨迹长时，有效学习率变小；轨迹短时，有效学习率变大。
        #严格来说是有偏的，但它简单、稳定（避免梯度爆炸），而且可以通过调整优化器学习率来弥补尺度问题。
        # ---- 反向传播更新 θ ----
        self.optimizer.zero_grad()#清空上次的梯度
        loss.backward()#反向传播，自动算出 loss 对每个参数的偏导数
        # 梯度裁剪，提升稳定性 (可选)
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()#更新权重

# =================== 训练循环 ===================
agent = REINFORCEAgent()      ## 创建智能体（策略网络、优化器全在里面）
episode_rewards = []          # 记录每个回合的总奖励 (用于绘图)

print("===== REINFORCE 开始训练 =====")
env.reset(seed=SEED)        # 给环境的随机数发生器播种（只播种一次）
                            # 之后每次 env.reset() 会按确定序列给出初始状态，保证可复现
for ep in range(EPISODES):
    state, _ = env.reset()#_:info：附加信息字典（这里用不上）
    total_reward = 0## 本回合累计奖励清零
    done = False

    log_probs = []   # 存每步的 log π(a_t|s_t)，回合结束一起用
    rewards   = []   # 存每步的奖励 r_t

    # ---------- 1. 用当前策略采样一整条轨迹 ----------
    while not done:#只要回合没结束就继续走
        # 1.1 选动作（按概率采样）
        action, log_prob = agent.select_action(state)

        # 1.2 执行动作
        # env.step(action) 在 gymnasium 中返回 5 个值：
        #   next_state : 执行动作后到达的新状态 (4 维数组)
        #   reward     : 这一步得到的奖励 (CartPole 每步存活 +1)
        #   terminated : 是否"任务失败/达成"而结束 (如杆子倒下) → 真回合结束
        #   truncated  : 是否"超时/达步数上限"被强制截断 (如撑过 500 步) → 只是时间到
        #   _ (info)   : 附加信息字典，这里用不上
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated   # 任一为 True 即认为本回合结束

        # 1.3 记录轨迹（注意：这一步【不学习】！与 DQN 的每步 update 不同）
        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state
        total_reward += reward

    # ---------- 2. 回合结束：用整条轨迹更新一次 ----------
    agent.update(log_probs, rewards)
    #对比 DQN：DQN 每步都 update（从经验池抽样）；REINFORCE 每回合只 update 一次（整条轨迹）

    episode_rewards.append(total_reward)

    # 打印训练进度
    if (ep + 1) % 100 == 0:
        avg = np.mean(episode_rewards[-100:])
        print(f"回合 {ep+1:4d} | 平均奖励(近100回合): {avg:.2f}")
print("===== 训练完成 =====")

#===================保存模型 ===================
# 注意：torch.save 用相对路径会保存到"运行命令时所在的目录"（可能不是脚本所在目录）
# 这里改用"脚本所在目录"（policy_gradient/），无论从哪里运行都存对位置
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reinforce_cartpole.pth")
torch.save(agent.policy_net.state_dict(), SAVE_PATH)
print(f"模型已保存为 {SAVE_PATH}")

# =================== 绘制学习曲线 ===================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("REINFORCE Training Curve")
plt.show()
env.close()


# ============================================================================
# 【理解笔记】REINFORCE 训练代码的整体流程
# ============================================================================
# 第一阶段：创建智能体（对象准备）
#   agent = REINFORCEAgent()，它自带以下固有属性：
#     - policy_net ：策略网络，输出动作概率分布，负责决策、且被更新
#     - optimizer  ：Adam 优化器，用梯度去更新 policy_net 的权重
#   （对比 DQN：没有 q_net/target_net 之分，也没有经验回放池——on-policy 用不到）
#   提供的方法：按概率采样动作(随机策略)、整条轨迹更新
#
# 第二阶段：回合循环（外层 for ep in range(EPISODES)）
#   1. env.reset() 重置环境 → 得到本回合初始状态
#   2. 初始化 log_probs=[]、rewards=[]（本回合轨迹收集器）
#
# 第三阶段：步循环（内层 while not done，每步做三件事）
#   1. 选动作   ：agent.select_action(state)，按概率采样（随机策略，不是 argmax）
#   2. 执行动作 ：env.step(action) → next_state、reward、terminated、truncated
#   3. 记轨迹   ：存 log_prob 和 reward（【关键】这一步不学习！）
#
# 第四阶段：回合收尾（步循环结束后，只学一次）
#   1. 计算 G_t ：agent.update 里把整回合 rewards 反向迭代成每个时刻的折扣回报
#   2. 更新网络 ：loss = -Σ_t log π(a_t|s_t)·G_t，反向传播一次
#   3. 记录本回合总奖励 total_reward
#
# 然后进入下一个回合，直到 EPISODES 个回合全部跑完。
#
# 【关键对比 DQN】
#   - 学习时机：DQN 每一步都 update；REINFORCE 每个回合只 update 一次
#   - 学习数据：DQN 从经验池随机抽样（旧数据反复用）；REINFORCE 用刚采的整条轨迹（用完即弃）
#   - 经验回放池 → 在 REINFORCE 里消失了（on-policy 方法里旧策略的数据无效）
# ============================================================================
