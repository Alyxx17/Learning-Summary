# RLMPC 核心方法总结

> 论文：Lin, Sun, Xia, Zhang, *"Reinforcement Learning-Based Model Predictive Control for Discrete-Time Systems"*, IEEE TNNLS 35(3), 2024.
> 本文档为个人快速复习用：精炼完整的方法逻辑与数学推导。

---

## 1. 问题设定

离散时间系统，状态/输入约束：

$$
x_{k+1}=f(x_k,u_k),\qquad x_k\in\mathbb{X},\ u_k\in\mathbb{U}
$$

其中 $\mathbb{X},\mathbb{U}$ 为紧集且原点为内点。运行代价 $\ell(x,u)\geq0,\ \ell(0,0)=0$（调节问题通常取 $\ell=x^\top Qx+u^\top Ru$）。

**目标（无限时域最优控制）**：求策略 $\pi:\mathbb{R}^n\to\mathbb{R}^m$ 使

$$
J_\infty(x_k)=\sum_{i=k}^{\infty}\ell(x_i,u_i)\ \text{最小，并镇定原点。}
$$

最优价值函数满足 Bellman 最优性方程：

$$
V^*(x_k)=\min_{u_k}\left\{\ell(x_k,u_k)+V^*(x_{k+1})\right\}
$$

---

## 2. 动机：传统 MPC 的困境

传统 MPC（OP1）：

$$
\min_{u_k}\ \sum_{i=0}^{N-1}\ell(x_{i|k},u_{i|k})+F(x_{N|k})\quad \text{s.t.}\ x_{N|k}\in\mathbb{X}_\Omega
$$

需**离线设计**终端代价 $F$、终端控制器 $\kappa$、终端集 $\mathbb{X}_\Omega$（满足 $F(x)\ge\sum_{i=N}^\infty\ell$ 的 Lyapunov 条件）。非线性系统极难设计，即便设计出来也保守（终端集小、时域要求长、代价高估）。

无终端条件的 MPCWTC（$F=0$、无终端约束）虽易用，但截断预测时域之外的代价被完全忽略，短时域性能差。

**RLMPC 的动机**：用 RL **在线学出**终端代价 $F$，即学价值函数当终端代价。

---

## 3. RLMPC 方法：以策略迭代（PI）桥接 MPC 与 RL

PI 两步映射：

- **策略改进** $\Rightarrow$ MPC 解 OP2（策略隐式定义于优化问题）
- **策略评估** $\Rightarrow$ VFA 回归学习价值函数

### 3.1 策略生成器 MPC（OP2）

$$
\min_{\boldsymbol u_k}\ J(x_k,\boldsymbol u_k)=\sum_{i=0}^{N-1}\ell(x_{i|k},u_{i|k})+V_\pi(x_{N|k})\tag{11a}
$$

$$
\text{s.t.}\ \ x_{0|k}=x_k,\quad x_{i+1|k}=f(x_{i|k},u_{i|k}),\quad u_{i|k}\in\mathbb{U},\quad x_{i|k}\in\mathbb{X}\ (1\le i\le N)
$$

- 仅约束输入/状态，**无终端约束**；
- 策略隐式定义：$\pi(x_k)=u_{0|k}^*$（最优序列首元素，滚动时域执行）；
- $V_\pi(\cdot)\equiv0$ 时即 MPCWTC，作为初始策略 $\pi^0$（Assumption 2,3 + Lemma 1 保证其稳定可行）。

### 3.2 策略评估：VFA + SGD 学价值函数

Weierstrass 逼近定理 $\Rightarrow$ 多项式基可一致逼近连续价值函数：

$$
V_\pi(x_k)=\sum_{i=1}^{p}W_i\Phi_i(x_k)+e(x_k)=W^\top\Phi(x_k)+e(x_k)=\hat V_\pi(x_k,W)+e(x_k)\tag{14}
$$

本质是"单层网络"，唯一参数 $W\in\mathbb{R}^p$。

**训练数据构造**（核心技巧）：在 $\mathbb{X}$ 采样 $q$ 点 $S_k=\{x_{k_1},\dots,x_{k_q}\}$，逐点解 OP2（终端代价固定为当前 $V^t$）得最优代价作为标签：

$$
J^t(x_{k_j},\boldsymbol u_{k_j}^t)=\min_{u}\left\{\sum_{i=0}^{N-1}\ell+V_\pi^t(x_{N|k_j})\right\}=\sum_{i=0}^{N-1}\ell(x_{i|k_j}^t,u_{i|k_j}^t)+V_\pi^t(x_{N|k_j}^t)\tag{17}
$$

即 **N 步 TD 目标**（$N$ 步真实代价 + 旧 $V$ bootstrap 尾巴），且 $N$ 步动作是 MPC 优化出的最优序列。

**SGD 更新推导**（平方误差对 $W$ 求梯度）：

$$
E=\frac12\left[J-W^\top\Phi\right]^2,\qquad
\nabla_W E=-(J-W^\top\Phi)\,\Phi
$$

$$
W\leftarrow W-\frac12\alpha\nabla_W\left[J-W^\top\Phi\right]^2=W+\alpha\big[\underbrace{J(x_{k_j},u^*)-W^\top\Phi(x_{k_j})}_{\text{TD 误差}}\big]\Phi(x_{k_j})\tag{16}
$$

对全部训练数据迭代至 $\big[J-\hat V_\pi\big]\to0$ 即得策略 $\pi^t$ 的评估 $V_\pi^{t+1}$：
$V_\pi^{t+1}(x)\to J^t(x,u^t),\ \forall x\in\mathbb{X}$。（等价批量形式：正规方程 $W=(\Phi^\top\Phi)^{-1}\Phi^\top J$，即 L2 岭回归闭式解。）

### 3.3 PI 循环

1. **初始化**：$V_\pi^0(\cdot)\equiv0$，对应策略 $\pi^0$（MPCWTC）；
2. **评估步**：采样 → 逐点解 OP2 得 $J^t$ → 式(16) 训练 → $V_\pi^{t+1}$；
3. **改进步**：以 $V_\pi^{t+1}$ 为终端代价解 OP2 得 $\pi^{t+1}$：
$$
\boldsymbol u_k^{t+1}=\arg\min_{u}\left\{\sum_{i=0}^{N-1}\ell(x_{i|k},u_{i|k})+V_\pi^{t+1}(x_{N|k})\right\}\tag{19}
$$
4. 重复，直至 $\|W^{t+1}-W^t\|\le\varepsilon$。

> 注：$N=1$ 时式(19) 退化为传统 PI 的改进步骤 $\pi'(s)=\arg\min_a[\ell+V(s')]$。RLMPC 即"一步前瞻"推广为"N 步前瞻"。

---

## 4. 理论性质（收敛/可行/稳定）

### 定理 1（单调不减）：$V_\pi^t\le V_\pi^{t+1},\ \forall t$

证明骨架：定义"任意可行序列累加值" $\Gamma^t(x)=\sum_{i=0}^{N-1}\ell(\tilde x_{i|k},\tilde u_{i|k})+\Gamma^{t-1}(\tilde x_{N|k})$，$\Gamma^0=0$。

- 一方面（式 23）：最优 ≤ 任意可行 $\Rightarrow V^t\le\Gamma^t$；
- 另一方面（式 24-25）：取 $\tilde u=u^t$（上轮最优）并归纳，得 $\Gamma^t\le V^{t+1}$。

合之：$V^t\le\Gamma^t\le V^{t+1}$。代价正定性保证 $V^t$ 正定。

### 定理 2（上界）：$V_\pi^t\le V^*,\ \forall t$

归纳：$V^0=0\le V^*$；若 $V^l\le V^*$，终端项放大使 min 增大：

$$
V^{l+1}=\min_u\left\{\sum\ell+V^l(x_N)\right\}\le\min_u\left\{\sum\ell+V^*(x_N)\right\}=V^*
$$

（末等号为 Bellman 最优性原理的 $N$ 步展开）。

### 定理 3（收敛）：$V_\pi^t\to V^*,\ \pi^t\to\pi^*$

定理 1 + 2 $\Rightarrow$ 单调有界序列收敛（单调收敛定理）$\Rightarrow\exists V^\infty$。对式(22) 取极限，$V^\infty$ 满足 Bellman 最优性方程 $\Rightarrow V^\infty=V^*$，从而 $\pi^t\to\pi^*$。

### 定理 4（等价时域，方法本质）：$\pi^t\equiv$ 时域 $(t+1)N$ 的 MPCWTC

定义 $V_{\tilde N}(x)=\min_u\sum_{i=0}^{\tilde N-1}\ell$（无终端）。由 Bellman 最优性原理拆分：

$$
V_{2N}(x)=\min_u\sum_{i=0}^{2N-1}\ell=\min_u\left\{\sum_{i=0}^{N-1}\ell+\underbrace{\min\sum_{i=N}^{2N-1}\ell}_{V_N(x_N)}\right\}
$$

又 $V_N=V_\pi^1$（$V^0=0$ 时首轮学到者恰为 $N$ 步无终端最优值），故 $V_{2N}=V_\pi^2$。归纳：

$$
V_{(t+1)N}=V_\pi^{t+1}\qquad\forall t\tag{33}
$$

**推论**：
- 每迭代一次 = 预测时域累积 $N$ 步；$t\to\infty$ 时时域 $\to\infty\Rightarrow$ 最优（与定理 3 互证）；
- 时域 $N$ 满足式(13) 则任意 $t$ 的策略均稳定可行 $\Rightarrow$ **任意时刻可安全停机**；
- 实际无需 $t\to\infty$：时域超过 $\bar N$ 后性能不再改善（Remark 7），$\|W\|$ 不动即收敛。

---

## 5. Algorithm 1（在线实现，$r^t\equiv1$）

```
Initialize: N,Q,R,α,ε,p,Φ,W⁰=0,t=0,Flag=0
for k=1,2,3,...:
    测量 x_k
    解 OP2（终端 (W^t)ᵀΦ(x_{N|k})）；施加 u_{0|k} 到真实系统
    if Flag==0:                       # 学习期
        采样 S_k = 100 随机点 + 实际轨迹
        for j=1..q:
            解 OP2 于 x_{k_j}，按式(17) 算 J
            SGD(16) 更新 W 至 ‖ΔW‖≤ε   # 内层循环
        W^{t+1} ← W
    if ‖W^{t+1}−W^t‖≤ε: Flag=1        # 学习完成，冻结 W
    t ← t+1
```

- 采样点上的最优序列**不施加**到真实系统（下标 $k_j$ 区分）；
- 每控制一步做一轮 PI 迭代（在线学习）；
- **Epi.1**（学习段）：$W^0=0$ 起步，边控边学，记录 W 演化与 ACC；
- **Epi.2**（复执段）：用收敛的 W 从同一初态重跑，检验学到的价值函数。
- 施加 u_{0|k} 到真实系统。主要目的是为了控制，也就是边学边控制。此外还作为一个样本进入学习阶段（100+1）个样本。系统被作用一步后，状态转移后的xk+1也仍被作为新的样本
- 学习过程的伪代码有歧义，收敛条件是所有 101 个点的误差同时趋于 0，即一个epoch内，对101个样本，每次执行一次SGD，执行101次后，对比‖W − W_prev‖，执行多次epoch，直至收敛阈值，再和最开始的W对比，每个epoch开始前可以打乱顺序。
- 严格来说SGD每次只能对一个样本进行更新，但不会被最后一个样本覆盖，因为第 1 个样本更新的 W 是第 2 个样本的起点，每个样本都在前人留下的"基础"上微调，最终 W 里含着全部样本的影响。
---

## 6. 仿真基准（复现验收标准）

### 线性（Benchmark，无约束）

$A=\begin{bmatrix}1&0.5\\-0.1&0.9\end{bmatrix},\ B=\begin{bmatrix}1\\0\end{bmatrix},\ Q=I,\ R=0.5,\ N=3,\ x_0=[2.9,2]^\top$

- $\alpha=10^{-6},\ \varepsilon=10^{-8},\ p=9$，$\Phi=[x_1,x_2,x_1^2,x_2^2,x_1x_2,x_1^3,x_1^2x_2,x_2^3,x_1x_2^2]$
- 采样：100 点（$x_1\in[-0.5,3.5],x_2\in[-0.5,2.5]$）+ 实际轨迹
- **11 次迭代收敛；Epi.2 ACC≈LQR 的 28.4626（误差约 $4\times10^{-5}$）**
- 对比：Q-learning 需 $1.3\times10^4$ 数据、PILCO 400、RLMPC 341

### 非线性（核心，unicycle）

$$
x_{k+1}=x_k+g(x_k)u_k,\qquad g(x_k)=\delta\begin{bmatrix}\cos\theta&0\\\sin\theta&0\\0&1\end{bmatrix},\ \delta=0.2
$$

状态 $x=[x,y,\theta]^\top$，输入 $u=[v,\omega]^\top$；约束 $|v|\le1,\ |\omega|\le4,\ 0\le x\le2$

- $x_0=[1.98,5,-\pi/3]^\top$，$Q=\text{diag}(1,2,0.06),\ R=\text{diag}(0.01,0.005)$
- RLMPC：$N=5,\ \alpha=10^{-6},\ \varepsilon=10^{-7},\ p=30$（1~6 阶多项式），每轮采样 100 点（$x\in[0,2],y\in[-1,6],\theta\in[-\pi,\pi]$）+ 实际轨迹
- 传统 MPC（文献[47]）：$N\ge30$，$F=x_N^2+y_N^2+\theta_N^2$，终端集 $F\le\varrho_v=c\varrho$，$\eta=\xi=8$

**Table II（Epi.1）**：

| 方法 | N=5 | N=7 | N=9 | N=30 | 传统MPC N=30 |
|---|---|---|---|---|---|
| ACC | **625.0301** | 624.1901 | 623.7521 | 560.2312 | 736.0854 |
| CR | 25 迭代 | 24 | 23 | 8 | — |
| ACT | 0.0826s | 0.0904s | 0.1143s | 0.3991s | 0.4126s |

- **25 次迭代收敛；Epi.1 ACC=625.03（比传统 MPC 低 15.09%），Epi.2 再降约 16%**
- 学好后 N=1 与 N=5 轨迹几乎重合（ACT 0.0826→0.0222s）
- 求解器：IPOPT 3.11.8

---

## 7. 复现实现要点（MATLAB 失败嫌疑对照）

| 项 | 论文 | 注意 |
|---|---|---|
| 求解器 | IPOPT | casadi 内置，勿用 fmincon |
| 基函数 | 1~6 阶多项式 p=30 | 具体 30 项论文未列，需对照/试验 |
| 传统 MPC | N=30，终端集 $\varrho_v=c\varrho$（≈0.0056） | MATLAB 用了 N=20、半径 0.5 |
| 离散化 | 欧拉 $x+\delta g(x)u$ | MATLAB 用了 RK4 |
| 约束 | 硬约束 | MATLAB 加了松弛 -0.1/-0.2 |
| SGD | 逐点迭代至 $\|\Delta W\|\le\varepsilon$，$\alpha=10^{-6}$ | MATLAB 是 batch+L2 正则 |
| 采样 | 纯均匀随机 + 轨迹 | MATLAB 加了 25% 目标邻域 |
| VI 不可用 | Remark 8：VI 中间价值函数不对应任何策略 | 必须用 PI |
## 8. SGD用岭回归+L2正则替代
假设我们有一组数据样本：

- $\Phi_{all}: n \times p$ 的矩阵，其中 $n$ 是样本数量，$p$ 是基函数维度（如 $x, x^2, \dots$）。
- $J_{all}: n \times 1$ 的向量，是待拟合的目标值（如代价函数）。

这行代码要找到权重 $W$，使以下带正则项的损失函数最小化：

$$
\min_W \left( \| \Phi_{all} W - J_{all} \|^2 + \text{RIDGE} \cdot \| W \|^2 \right)
$$

通过对上述损失函数求导并令导数为零，可以得到**正规方程**（Normal Equation）：

$$
(\Phi_{all}^T \Phi_{all} + \lambda I) W = \Phi_{all}^T J_{all}
$$

其中 $\lambda = \text{RIDGE}$。
