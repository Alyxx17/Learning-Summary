# -*- coding: utf-8 -*-
"""DDPG 智能体（论文 Section III-D，式 16-29）。

网络结构按论文 Section IV-B：
    Actor  : 2 个隐层 20-20，ReLU，输出经 tanh 缩放到动作区间；
    Critic : 2 个隐层 100-75，ReLU，输入为状态与动作的拼接。
超参数按论文 Table I：缓冲 1e5、批 64、η_Q=1e-3、η_μ=1e-4、τ=1e-3、γ=0.99。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """确定性策略 μ(s|θ^μ)。"""

    def __init__(self, state_dimension, hidden_units=(20, 20), action_scale=1.0, action_bias=0.0):
        super().__init__()
        self.action_scale = float(action_scale)
        self.action_bias = float(action_bias)

        layers = []
        previous = int(state_dimension)
        for units in hidden_units:
            layers.extend([nn.Linear(previous, int(units)), nn.ReLU()])
            previous = int(units)
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        return self.action_scale * torch.tanh(self.layers(state)) + self.action_bias


class CriticNetwork(nn.Module):
    """动作价值 Q(s, a|θ^Q)。"""

    def __init__(self, state_dimension, action_dimension=1, hidden_units=(100, 75)):
        super().__init__()
        layers = []
        previous = int(state_dimension) + int(action_dimension)
        for units in hidden_units:
            layers.extend([nn.Linear(previous, int(units)), nn.ReLU()])
            previous = int(units)
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.layers(torch.cat([state, action], dim=1))


class ReplayBuffer:
    """经验回放缓冲区 D。"""

    def __init__(self, capacity, state_dimension, action_dimension=1):
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, state_dimension), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dimension), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dimension), dtype=np.float32)
        self.terminals = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminal):
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = float(terminal)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, generator=None):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.terminals[indices]),
        )


class OrnsteinUhlenbeckNoise:
    """动作探索噪声（论文式 23 中的 ε）。"""

    def __init__(self, dimension=1, mean=0.0, theta=0.15, sigma=0.2, scale=1.0, seed=None):
        self.mean = mean
        self.theta = theta
        self.sigma = sigma
        self.scale = scale
        self.random = np.random.RandomState(seed)
        self.state = np.zeros(dimension)

    def reset(self):
        self.state = np.zeros_like(self.state)

    def sample(self):
        derivative = self.theta * (self.mean - self.state) + self.sigma * self.random.randn(*self.state.shape)
        self.state = self.state + derivative
        return self.state * self.scale


class DDPGAgent:
    """DDPG 智能体，含目标网络与软更新（式 24-29）。"""

    def __init__(
        self,
        state_dimension,
        action_scale,
        action_bias,
        actor_hidden_units=(20, 20),
        critic_hidden_units=(100, 75),
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        discount=0.99,
        soft_update_rate=1e-3,
        buffer_capacity=100000,
        batch_size=64,
        action_dimension=1,
        device="cpu",
    ):
        self.state_dimension = int(state_dimension)
        self.action_dimension = int(action_dimension)
        self.action_scale = float(action_scale)
        self.action_bias = float(action_bias)
        self.discount = float(discount)
        self.soft_update_rate = float(soft_update_rate)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        self.actor = ActorNetwork(
            self.state_dimension, actor_hidden_units, self.action_scale, self.action_bias
        ).to(self.device)
        self.critic = CriticNetwork(
            self.state_dimension, self.action_dimension, critic_hidden_units
        ).to(self.device)
        self.target_actor = ActorNetwork(
            self.state_dimension, actor_hidden_units, self.action_scale, self.action_bias
        ).to(self.device)
        self.target_critic = CriticNetwork(
            self.state_dimension, self.action_dimension, critic_hidden_units
        ).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.buffer = ReplayBuffer(buffer_capacity, self.state_dimension, self.action_dimension)

    # -- 交互 --------------------------------------------------------------
    def select_action(self, state, noise=None):
        tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(tensor).cpu().numpy().reshape(-1)
        if noise is not None:
            action = action + noise
        return action

    def store_transition(self, state, action, reward, next_state, terminal):
        self.buffer.add(state, action, reward, next_state, terminal)

    # -- 训练 --------------------------------------------------------------
    def update(self):
        """一次梯度更新：先更新 Critic（式 24-25），再更新 Actor（式 26-27）。"""
        if self.buffer.size < self.batch_size:
            return None

        states, actions, rewards, next_states, terminals = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminals = terminals.to(self.device)

        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_values = self.target_critic(next_states, target_actions)
            targets = rewards + self.discount * (1.0 - terminals) * target_values

        values = self.critic(states, actions)
        critic_loss = F.mse_loss(values, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        return float(critic_loss.item()), float(actor_loss.item())

    def _soft_update(self, target, source):
        with torch.no_grad():
            for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                target_parameter.mul_(1.0 - self.soft_update_rate)
                target_parameter.add_(self.soft_update_rate * source_parameter)

    # -- 存取 --------------------------------------------------------------
    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_actor(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint.get("target_actor", checkpoint["actor"]))
        return self
