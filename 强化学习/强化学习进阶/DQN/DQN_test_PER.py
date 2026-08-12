# ============================================================
# Prioritized Replay (PER) DQN 测试代码
# 网络结构与训练一致（Dueling + Noisy）；q_net.eval() 后确定性推理
# ============================================================
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ================= 定义相同的网络结构（Dueling + Noisy，与训练时完全一致） =================
class NoisyLinear(nn.Module):
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

# ================= 加载环境（开启渲染） =================
env = gym.make("LunarLander-v3", render_mode="human", continuous=False)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# ================= 创建网络并加载权重 =================
q_net = QNetwork(state_dim, action_dim)
q_net = q_net.to(DEVICE)
q_net.load_state_dict(torch.load("dqn_LunarLander_per.pth"))
q_net.eval()   # 切到评估模式：NoisyLinear 不再采样噪声

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
