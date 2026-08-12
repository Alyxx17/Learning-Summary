# ============================================================
# N-step DQN 测试代码
# 网络结构与训练一致（Dueling）；N-step 只影响训练阶段，推理逻辑不变
# ============================================================
import gymnasium as gym
import torch
import torch.nn as nn
import time

# ================= 定义相同的网络结构（Dueling，与训练时完全一致） =================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_dim)

    def forward(self, x):
        features = self.feature(x)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

# ================= 加载环境（开启渲染） =================
env = gym.make("CartPole-v1", render_mode="human")
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# ================= 创建网络并加载权重 =================
q_net = QNetwork(state_dim, action_dim)
q_net = q_net.to(DEVICE)
q_net.load_state_dict(torch.load("dqn_cartpole_nstep.pth"))
q_net.eval()

# ================= 测试循环 =================
print("===== 播放动画 =====")
env.reset(seed=SEED)
for i in range(3):
    state, _ = env.reset()
    total = 0
    done = False
    while not done:
        time.sleep(0.02)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action = q_net(state_t).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    print(f"测试 {i+1}: 总奖励 = {total}")

env.close()
