# =================== gym 式环境: 跟踪 MPC + unicycle ===================
#   reset()  -> (obs, info)
#   step(a)  -> (obs, reward, terminated, truncated, info)
#
# 维度约定:
#   action  (5,) 归一化动作 ∈ [-1,1]（SAC 的 tanh 输出）
#             内部 decode_action → 权重向量 w = [q1,q2,q3, r1,r2]（即 Q、R 的对角线本身，
#             作用在归一化误差/控制偏差上；每个分量对数均匀 ∈ [0.005, 20]）
#   obs     (5,) = [e_x, e_y, e_θ, v_ref, ω_ref]
#             e_θ ∈ (-π, π]（wrap 归一化）；位置偏移初值 ≤ 0.4 m；
#             v_ref ∈ [0.3, 0.8]；ω_ref 与曲率有关（|ω_ref| 约 ≤ 2）
#   reward  标量 = -单步评价代价 / C0（量级由 C0 标定）
#             评价口径 = SCORE_W 加权归一化平方和（SCORE_MODE 切换；
#             定义见 unicycle_env.stage_cost；与动作无关——防"改记分板"）
#
# 回合边界: sim_steps 步（truncated）；|e_pos| > diverge_thresh 视为发散（terminated，安全阀）
# info 字段: weights（本步 5 个权重）、solve_ok、solve_time、e_pos

import numpy as np
import unicycle_env as env
from tracking_mpc import TrackingMPC


class MPCEnv:
    """SAC × MPC 接口练习环境（SAC 每步输出 Q、R 的 5 个对角权重）"""

    def __init__(self, N=5, sim_steps=80, c0=0.01, diverge_thresh=1.5, seed=None):
        self.N = N                              # MPC 预测时域
        self.sim_steps = sim_steps              # 每回合步数
        self.c0 = c0                            # 奖励常数归一化（按基线单步代价量级定标）
        self.diverge_thresh = diverge_thresh    # 发散早停阈值 (m)；None = 关闭
        self.mpc = TrackingMPC(N=N)             # MPC 求解器（build 一次，回合间复用）
        self.rng = np.random.default_rng(seed)  # 场景采样随机流（固定 seed 可复现）
        self.obs_dim = 5
        self.action_dim = 5
        # ---- 回合内状态（reset 时初始化）----
        self.state = None                       # 当前真实状态 (3,)
        self.k = None                           # 回合内步数
        self.X_ref = None                       # 参考状态序列 (sim_steps+N, 3)
        self.U_ref = None                       # 参考输入序列 (sim_steps+N, 2)
        self._ref = None                        # ReferenceTrajectory 对象（打印/记录用）
        self.u_guess = None                     # MPC 热启动猜测量

    def reset(self):
        """随机参考轨迹 + 随机初值，开始新回合
        返回: (obs (5,) float32, info dict)"""
        # 参考序列多采 N 步，覆盖最后一步的预测窗口
        self._ref, self.X_ref, self.U_ref = env.sample_reference(self.rng, self.sim_steps + self.N)
        self.state = env.sample_initial_state(self.rng, self._ref)
        self.k = 0
        self.u_guess = np.zeros(2 * self.N)
        return self._obs(), {"ref": repr(self._ref), "x0": self.state.copy()}
        #第一行代码中的env.sample_reference内部已经创建了ref实例，并赋值给self._ref，所以这里的self._ref就是一个ReferenceTrajectory对象。repr(self._ref)会调用ReferenceTrajectory类的__repr__方法，返回一个字符串，描述参考轨迹的类型和参数信息。info字典中包含了参考轨迹的描述和初始状态x0的拷贝，用于记录和打印。

    def _obs(self):
        """obs = [e_x, e_y, e_θ, v_ref, ω_ref]（5,）"""
        e = env.tracking_error(self.state, self.X_ref[self.k])
        return np.concatenate([e, self.U_ref[self.k]]).astype(np.float32)

    def step(self, action):
        """一步环境交互: 动作 → 解一次 MPC → 施加首步控制 → 推进真实系统 → 奖励
        参数: action (5,) 归一化动作 ∈ [-1,1]
        返回: (obs, reward, terminated, truncated, info)
        """
        w = env.decode_action(action)           # (5,) = [q1,q2,q3, r1,r2]
        info = {"weights": w.copy()}            # 本步权重（供记录/分析用）

        # ---- 解一次 MPC（当前状态 + 当前参考窗口）----
        u_opt, _, ok, _, t_solve = self.mpc.solve(
            self.state,
            self.X_ref[self.k: self.k + self.N],
            self.U_ref[self.k: self.k + self.N],
            q_diag=w[:3], r_diag=w[3:], u_guess=self.u_guess)
        info["solve_ok"] = bool(ok)
        info["solve_time"] = float(t_solve)
        if not ok:
            u_opt = np.zeros((self.N, 2))       # 求解失败回退零控制
        u0 = u_opt[0]                           # 滚动时域: 只施加首步控制

        # ---- 奖励: 单步评价代价（评分口径 SCORE_W，见 unicycle_env.stage_cost）/ C0 ----
        cost = env.stage_cost(self.state, u0, self.X_ref[self.k], self.U_ref[self.k])
        reward = -cost / self.c0

        # ---- 推进真实系统 + 热启动左移补零 ----
        self.state = env.dynamics(self.state, u0)
        self.u_guess = np.concatenate([u_opt.reshape(-1)[2:], [0.0, 0.0]])
        self.k += 1

        # ---- 回合边界 ----
        e_pos = float(np.linalg.norm(env.tracking_error(self.state, self.X_ref[self.k])[:2]))
        info["e_pos"] = e_pos
        terminated = bool(self.diverge_thresh is not None and e_pos > self.diverge_thresh)
        truncated = bool(self.k >= self.sim_steps)
        return self._obs(), reward, terminated, truncated, info


