
import gymnasium as gym
import torch
import torch.nn as nn
import time

# ================= 定义相同的网络结构（与训练时完全一致） =================
class QNetwork(nn.Module):
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
q_net = QNetwork(state_dim, action_dim)
q_net = q_net.to(DEVICE)  # 把网络也移到 DEVICE（和输入保持同一设备，否则矩阵乘法报 device mismatch）
q_net.load_state_dict(torch.load("dqn_cartpole.pth"))
q_net.eval()  # 切换到评估模式（关闭 Dropout 等，这里没用到但习惯加上）
#nn.Module自带的函数，用于评估

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
            action = q_net(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    print(f"测试 {i+1}: 总奖励 = {total}")

env.close()