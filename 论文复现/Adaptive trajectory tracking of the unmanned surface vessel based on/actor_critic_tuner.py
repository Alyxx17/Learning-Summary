"""式(16)-(32) 的 Actor-Critic 参数自适应器（纯 numpy 实现，网络结构按 Table 2/3）。

* 状态  s = [Q_w(1,1), Q_w(2,2), Q_w(3,3)]                —— 式(22)
* 动作  a = [dq1, dq2, dq3]，每个分量取 +0.1 / -0.1 / 0    —— 式(23)
* 奖励  r = -sqrt(xs^2 + ys^2) - |psi_s|                   —— 式(24)
* Critic 用 TD 误差  delta = r + gamma V(s') - V(s) 更新（式(27)(28)），
  Actor 用  delta * grad log pi(a|s) 更新（式(29)）。
* 预测步长 N 由奖励增量 dR = R_t - R_{t-1} 按 Table 5 / 式(31)(32) 选取。
"""
from __future__ import annotations

import numpy as np

HIDDEN_DEFAULT = (264, 128)          # Table 2 / Table 3
N_ACTION_PER_DIM = 3                 # 式(23)：+dq / -dq / 0
ACTION_MULTIPLIERS = (1.0, -1.0, 0.0)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    return z / z.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.moments = None
        self.steps = 0

    def update(self, params, grads):
        if self.moments is None:
            self.moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        self.steps += 1
        bias1 = 1.0 - self.beta1 ** self.steps
        bias2 = 1.0 - self.beta2 ** self.steps
        for param, grad, (mom, var) in zip(params, grads, self.moments):
            mom *= self.beta1
            mom += (1.0 - self.beta1) * grad
            var *= self.beta2
            var += (1.0 - self.beta2) * grad * grad
            param -= self.lr * (mom / bias1) / (np.sqrt(var / bias2) + self.eps)


class DenseNetwork:
    """全连接 ReLU 网络（输出层线性）。"""

    def __init__(self, sizes, rng, out_scale=0.1):
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
        # 输出层接近 0 初始化：策略初始近似均匀，价值初始近似 0
        self.weights[-1] *= out_scale

    def parameters(self):
        return self.weights + self.biases

    def forward(self, x):
        hidden = [x]
        h = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(h @ w + b, 0.0)
            hidden.append(h)
        return h @ self.weights[-1] + self.biases[-1], hidden

    def backward(self, hidden, grad_out):
        n_layer = len(self.weights)
        grad_w = [None] * n_layer
        grad_b = [None] * n_layer
        delta = grad_out
        for i in range(n_layer - 1, -1, -1):
            grad_w[i] = hidden[i].T @ delta
            grad_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (hidden[i] > 0.0)
        return grad_w, grad_b


# 式(32) 的奖励增量分段 + Table 5 的预测步长
_DELTA_EDGES = (50.0, 20.0, 0.0, -20.0)
_TABLE5_HORIZON = (12, 10, 8, 5, 3)


def reward_delta_level(delta_r):
    """式(32)：把奖励增量映射为 x = 1..5（1 对应 dR 最小，5 对应 dR 最大）。"""
    if delta_r >= _DELTA_EDGES[0]:
        return 5
    if delta_r >= _DELTA_EDGES[1]:
        return 4
    if delta_r >= _DELTA_EDGES[2]:
        return 3
    if delta_r >= _DELTA_EDGES[3]:
        return 2
    return 1


def select_horizon(delta_r, mode="table5", amplitude=2):
    """式(30)-(32) 与 Table 5：由奖励增量选择预测步长 N。

    delta_r 必须落在 Table 5 的 20/50 量级上才有意义。论文的奖励量纲与
    "每步奖励差" 不匹配（ΔR 只有 0.1 量级），能同时自洽的解释是
    ΔR 取"相邻训练回合总回报之差"（见 sinusoid_acmpc_experiment 的 horizon_update）。
    """
    level = reward_delta_level(delta_r)
    if mode == "table5":
        return _TABLE5_HORIZON[5 - level]
    if mode == "eq31":
        return int(amplitude * level + 1)
    raise ValueError(f"unknown horizon mode: {mode}")


class ActorCriticTuner:
    """在线调整 MPC 状态权重 Q_w 与预测步长 N。"""

    def __init__(self, q_min=0.1, q_max=12.0, dq=0.1, hidden=HIDDEN_DEFAULT,
                 lr=1e-3, gamma=0.98, seed=0, entropy_weight=0.005,
                 critic_epochs=2, actor_epochs=2, q_scale=5.0):
        self.rng = np.random.default_rng(seed)
        self.q_min, self.q_max, self.dq = float(q_min), float(q_max), float(dq)
        self.gamma = float(gamma)
        self.entropy_weight = float(entropy_weight)
        self.critic_epochs = int(critic_epochs)
        self.actor_epochs = int(actor_epochs)
        self.q_scale = float(q_scale)

        self.action_deltas = np.asarray(ACTION_MULTIPLIERS, dtype=float) * self.dq

        self.actor = DenseNetwork([3, *hidden, 3 * N_ACTION_PER_DIM], self.rng)
        self.critic = DenseNetwork([3, *hidden, 1], self.rng)
        self.adam_actor = Adam(lr)
        self.adam_critic = Adam(lr)

    # ------------------------------------------------------------ 状态编码
    def encode(self, q):
        return (np.asarray(q, dtype=float) - 1.0) / self.q_scale

    # ------------------------------------------------------------ 动作选择
    def act(self, q, greedy=False):
        """式(23)：对 Q_w 的三个分量各自独立地做 +dq / -dq / 0 三选一。"""
        logits, _ = self.actor.forward(self.encode(q)[None, :])
        prob = softmax(logits.reshape(3, N_ACTION_PER_DIM))
        index = np.empty(3, dtype=int)
        for i in range(3):
            p = prob[i] / prob[i].sum()
            index[i] = int(np.argmax(p)) if greedy else int(self.rng.choice(N_ACTION_PER_DIM, p=p))
        delta = np.array([self.action_deltas[index[i]] for i in range(3)])
        return index, delta, np.clip(q + delta, self.q_min, self.q_max)

    # -------------------------------------------------------------- 训练
    @staticmethod
    def _returns_to_go(rewards, gamma):
        """式(25)(27) 的累积回报 R_t = r_{t+1} + γr_{t+2} + ..."""
        out = np.zeros_like(rewards)
        acc = 0.0
        for i in range(rewards.size - 1, -1, -1):
            acc = rewards[i] + gamma * acc
            out[i] = acc
        return out

    def update(self, states, actions, rewards):
        """states 为动作执行前的 Q_w 序列（原始值），rewards 为对应单步奖励。

        优势函数用累积回报减去 Critic 的估计（式(27) 的 V^π），比单步 TD 误差稳定得多：
        单步奖励主要由轨迹相位决定，会把 Q_w 的作用淹没。
        """
        s = self.encode(np.asarray(states, dtype=float))
        a = np.asarray(actions, dtype=int)
        r = np.asarray(rewards, dtype=float)
        count = s.shape[0]
        if count < 2:
            return 0.0

        returns = self._returns_to_go(r, self.gamma)
        # 尺度无关归一化：奖励量纲变化时 Critic 的回归目标仍在 O(1)，便于 Adam 收敛
        returns = returns / max(float(np.mean(np.abs(returns))), 1.0)
        target = returns[:, None]

        for _ in range(self.critic_epochs):
            value, hidden = self.critic.forward(s)
            err = np.clip(target - value, -10.0, 10.0)
            grad_w, grad_b = self.critic.backward(hidden, (-err / count))
            self.adam_critic.update(self.critic.parameters(), grad_w + grad_b)

        value, _ = self.critic.forward(s)
        adv = (target - value)[:, 0]
        adv = adv / max(float(adv.std()), 1e-3)               # 优势归一化，稳定策略梯度

        logits, hidden = self.actor.forward(s)
        prob = softmax(logits.reshape(count, 3, N_ACTION_PER_DIM))
        onehot = np.zeros_like(prob)
        rows = np.arange(count)
        for _ in range(self.actor_epochs):
            for i in range(3):
                onehot[rows, i, a[:, i]] = 1.0
            grad = adv[:, None, None] * (prob - onehot) / count
            if self.entropy_weight > 0.0:
                entropy = -(prob * np.log(prob + 1e-12)).sum(axis=2, keepdims=True)
                grad += self.entropy_weight * prob * (np.log(prob + 1e-12) + entropy)
            grad_w, grad_b = self.actor.backward(
                hidden, grad.reshape(count, 3 * N_ACTION_PER_DIM))
            self.adam_actor.update(self.actor.parameters(), grad_w + grad_b)
            logits, hidden = self.actor.forward(s)
            prob = softmax(logits.reshape(count, 3, N_ACTION_PER_DIM))
        return float(np.mean(adv))

    # ------------------------------------------------------------ 存取
    def state_dict(self):
        data = {}
        for name, net in (("actor", self.actor), ("critic", self.critic)):
            for i, w in enumerate(net.weights):
                data[f"{name}_w{i}"] = w
            for i, b in enumerate(net.biases):
                data[f"{name}_b{i}"] = b
        return data

    def load_state_dict(self, data):
        for name, net in (("actor", self.actor), ("critic", self.critic)):
            for i in range(len(net.weights)):
                net.weights[i] = np.array(data[f"{name}_w{i}"], dtype=float)
            for i in range(len(net.biases)):
                net.biases[i] = np.array(data[f"{name}_b{i}"], dtype=float)
