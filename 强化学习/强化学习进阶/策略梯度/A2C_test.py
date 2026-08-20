# ============================================================
# Rainbow A2C 测试代码 —— 加载训练好的 actor-critic 网络并播放动画
# 只用 actor 头（logits）做决策，测试用 argmax（贪婪，表现更稳定）
# ============================================================

import os
import gymnasium as gym
import torch
import torch.nn as nn
import time

# ================= 定义相同的网络结构（与训练时完全一致：256 共享网络） =================
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(256, action_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor_head(features)
        value  = self.critic_head(features).squeeze(-1)
        return logits, value

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42               # 测试种子：与训练用同一个，保证测试初始状态可复现、可对比

# ================= 加载环境（开启渲染） =================
env = gym.make("LunarLander-v3", render_mode="human")#打开可视化窗口
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# ================= 创建网络并加载权重 =================
net = ActorCriticNet(state_dim, action_dim)
net = net.to(DEVICE)  # 把网络也移到 DEVICE（和输入保持同一设备，否则矩阵乘法报 device mismatch）
# 加载"脚本所在目录"下的模型（与训练保存路径保持一致，不受运行目录影响）
LOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2c_lunarlander.pth")
# weights_only=False：文件是本机训练生成的（可信来源）。
# PyTorch 2.6+ 默认 weights_only=True，会拒绝加载含 numpy 标量的超参数字典；
# 加上此参数可兼容已训练好的旧模型文件（不需要重新训练）
checkpoint = torch.load(LOAD_PATH, map_location=DEVICE, weights_only=False)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()  # 切换到评估模式（关闭 Dropout 等，这里没用到但习惯加上）

# ================= 测试循环 =================
print("===== 播放动画 =====")
env.reset(seed=SEED)   # 只播种一次：测试面对确定（且与其它模型相同）的初始状态序列
for i in range(3):
    state, _ = env.reset()
    total = 0
    done = False
    while not done:
        # render_mode='human' 会自动渲染，但加个小延迟可以让动画肉眼可见
        time.sleep(0.02)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits, _ = net(state_t)           # [1, 4] logits（只用 actor 头）
            action = logits.argmax().item()    # 贪婪决策（测试时更稳定）
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    print(f"测试 {i+1}: 总奖励 = {total:.2f}")

env.close()
