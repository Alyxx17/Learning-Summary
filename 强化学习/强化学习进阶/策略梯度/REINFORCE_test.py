# ============================================================
# REINFORCE 测试代码 —— 加载训练好的策略网络并播放动画
# 与 DQN_test.py 的区别：决策从 argmax 改为按概率采样（随机策略）
# ============================================================

import os
import gymnasium as gym
import torch
import torch.nn as nn
import time

# ================= 定义相同的网络结构（与训练时完全一致） =================
class PolicyNet(nn.Module):
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
        return self.fc(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42               # 测试种子：与训练用同一个，保证测试初始状态可复现、可对比

# ================= 加载环境（开启渲染） =================
env = gym.make("CartPole-v1", render_mode="human")#打开可视化窗口
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# ================= 创建网络并加载权重 =================
policy_net = PolicyNet(state_dim, action_dim)
policy_net = policy_net.to(DEVICE)  # 把网络也移到 DEVICE（和输入保持同一设备，否则矩阵乘法报 device mismatch）
# 加载"脚本所在目录"下的模型（与训练保存路径保持一致，不受运行目录影响）
LOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reinforce_cartpole.pth")
policy_net.load_state_dict(torch.load(LOAD_PATH, map_location=DEVICE))
policy_net.eval()  # 切换到评估模式（关闭 Dropout 等，这里没用到但习惯加上）

# ================= 测试循环 =================
print("===== 播放动画 =====")
env.reset(seed=SEED)   # 只播种一次：3 局测试面对确定（且与其它模型相同）的初始状态序列
for i in range(3):
    state, _ = env.reset()
    total = 0
    done = False
    while not done:
        # render_mode='human' 会自动渲染，但加个小延迟可以让动画肉眼可见
        time.sleep(0.02)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits = policy_net(state_t)   # [1, 2] logits
            # 按概率采样动作（随机策略）——与训练时的决策方式保持一致
            action = torch.distributions.Categorical(logits=logits).sample().item()
            # 注意：测试时若想表现更稳定，可改用 argmax 贪婪决策：
            # action = logits.argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    print(f"测试 {i+1}: 总奖励 = {total}")

env.close()
