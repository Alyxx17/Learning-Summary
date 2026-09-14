# ============================================================
# SAC 测试代码 —— 加载训练好的 actor（策略网络）并播放动画
# 只用 actor：确定性决策 a = tanh(μ(s))（不采样，表现更稳定，同 A2C 测试用 argmax）
# 约定：网络输出的是【归一化动作】∈[-1,1]（tanh 压缩），
#       送进环境前乘 ACTION_SCALE=2 还原成 Pendulum 的真实扭矩 ∈[-2,2]
# ============================================================

import os
import gymnasium as gym
import torch
import torch.nn as nn
import time

# ================= 定义相同的网络结构（与训练时完全一致：256 隐藏层） =================
class ActorNet(nn.Module):
    """注意：虽然测试只用 μ 头，但 log_std 头也要定义——
    load_state_dict 要求参数名/形状与保存时严格一致"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(256, action_dim)   # → μ（决策用）
        self.log_std_head = nn.Linear(256, action_dim)   # → log σ（测试不用，但必须存在）

    def forward(self, x):
        features = self.fc(x)
        mean    = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42               # 测试种子：与训练用同一个，保证测试初始状态可复现、可对比

# ================= 加载环境（开启渲染） =================
env = gym.make("Pendulum-v1", render_mode="human")#打开可视化窗口
state_dim  = env.observation_space.shape[0]   # 3
action_dim = env.action_space.shape[0]        # 1（连续动作用 shape[0]，离散才用 .n）
ACTION_SCALE = float(env.action_space.high[0])  # 动作上限 2.0（与训练一致）

# ================= 创建网络并加载权重 =================
actor = ActorNet(state_dim, action_dim)
actor = actor.to(DEVICE)  # 把网络也移到 DEVICE（和输入保持同一设备，否则矩阵乘法报 device mismatch）
# 加载"脚本所在目录"下的模型（与训练保存路径保持一致，不受运行目录影响）
LOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sac_pendulum.pth")
# weights_only=False：文件是本机训练生成的（可信来源）。
# PyTorch 2.6+ 默认 weights_only=True，会拒绝加载含 numpy 标量的超参数字典；
# 加上此参数可兼容训练保存的文件
checkpoint = torch.load(LOAD_PATH, map_location=DEVICE, weights_only=False)
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()  # 切换到评估模式（关闭 Dropout 等，这里没用到但习惯加上）

# ================= 测试循环 =================
print("===== 播放动画（Pendulum 摆杆稳定在竖直附近） =====")
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
            mean, _ = actor(state_t)                    # [1, 1] 均值（只用 μ 头）
            action = torch.tanh(mean)                   # 确定性动作（归一化 [-1,1]）
            action = action.squeeze(0).cpu().numpy() * ACTION_SCALE   # → 真实扭矩
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += reward
    print(f"测试 {i+1}: 总奖励 = {total:.2f}（Pendulum 回报参考：-1200 乱转 → -150 优秀）")

env.close()
