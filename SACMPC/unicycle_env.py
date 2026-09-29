# =================== unicycle 对象 + 跟踪任务组件 ===================
# 与 P02 复现（Paper Reproduction/RLMPC）的区别:
#   ① 去掉状态硬约束 0≤x≤2（跟踪任意参考轨迹时该约束无意义）
#   ② 代价改为跟踪形式: 归一化误差/控制偏差的加权平方和（权重作用在 ê=e/ERR_SCALE、
#      Δû=Δu/CTRL_SCALE 上；5 个对角权重 [q1,q2,q3,r1,r2] 由 SAC 学习——见 decode_action）
#   ③ 评价口径 = SCORE_W 加权归一化平方和（可切换 balanced/error/control）——与动作无关
#      （防"改记分板"）；评分同时是训练奖励口径。默认控制器仍为"单位权重"：
#      balanced 时与评分同源，切到 error/control 即"默认与评分错配"设定
#   ④ 新增: 解析参数化参考轨迹生成器（直线/圆弧/正弦侧偏）+ 随机初始状态采样
#      —— 随机性是"SAC 在线调参"任务的意义来源（每回合轨迹与初值都不同）

import numpy as np

# =================== 系统参数（沿用 P02 论文值） ===================
DT = 0.2                                 # 采样间隔 δ（s）
V_MIN, V_MAX = -1.0, 1.0                 # 线速度约束 |v| ≤ 1 m/s
W_MIN, W_MAX = -4.0, 4.0                 # 角速度约束 |ω| ≤ 4 rad/s

# =================== 动作 → 代价权重 映射范围 ===================
# SAC 学的是 Q、R 对角线上的 5 个权重本身: w = [q1,q2,q3, r1,r2]
# （q1..q3 对应误差 x/y/θ；r1..r2 对应控制偏差 Δv/Δω，作用在归一化信号上）
# 每个分量对数均匀映射到 [WEIGHT_MIN, WEIGHT_MAX]，保正且有界
WEIGHT_MIN, WEIGHT_MAX = 0.005, 20.0

# =================== 信号归一化尺度（使权重数值可比、无量纲） ===================
# 误差尺度 ≈ 轨迹坐标量程（x、y 约 ±10 m；θ ∈ ±π）；控制尺度 = 执行器限幅
# （先按尺度归一化、再谈权重——"默认单位权重"才有清晰含义）
ERR_SCALE  = np.array([10, 10, np.pi])   # 误差量程 [m, m, rad]
CTRL_SCALE = np.array([1.0, 4.0])        # 控制限幅 [m/s, rad/s]

# =================== 评分口径（可切换；同时也是训练奖励的口径） ===================
# 评分 ℓ = Σ SCORE_W[:3]·(e/ERR)² + Σ SCORE_W[3:]·(Δu/CTRL)²（逐段求和 → 对比表"平均代价"）
# 预设（倍数可按需调整；也可在 SCORE_PRESETS 里加自定义条目）:
#   "balanced" 均衡（单位权重，无先验默认）
#   "error"    更看重误差（误差块 ×10）
#   "control"  更看重控制（控制块 ×10）
# 注意: stage_cost 同时定义训练奖励 ⇒ 换口径 = 换训练目标，SAC 必须重新训练；
#       基线控制器仍固定为单位权重（evaluate_compare.BASELINE_W）→ 切到 error/control 即"错配"设定
SCORE_MODE = "error"  # "balanced" / "error" / "control"
SCORE_PRESETS = {
    "balanced": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "error":    np.array([10.0, 10.0, 10.0, 1.0, 1.0]),
    "control":  np.array([1.0, 1.0, 1.0, 10.0, 10.0]),
}
SCORE_W = SCORE_PRESETS[SCORE_MODE]      # 当前生效的评分权重（奖励/对比/日志全部随它变）

# 参考轨迹峰值速度上限（m/s）：执行器线速度上限 v_max=1，留 0.1 机动余量
# （只约束 sine 族振幅——直线/圆弧的参考速度恒等于 v ≤ 0.8）
REF_SPEED_CAP = 0.9

# =================== 模型失配（真机 vs MPC 标称模型；实验用） ===================
# 用途: 给"真实系统" dynamics() 注入与 MPC 内部标称模型不一致的动力学，
#       用于测试"SAC 调权重 vs 固定权重"在【模型失配】下的表现（模型误差补偿实验）。
# 机制（三项可任意组合；MPC 内部模型恒为理想模型——它不知道这些失配）:
#   k_v       —— 线速度执行增益偏差: 真机实际输出 = k_v × 指令（MPC 认为 k_v = 1）
#   k_w       —— 角速度执行增益偏差（同上）
#   slip_beta —— 侧滑角 (rad): 真实速度方向 = 车头朝向 + β（等效横漂/侧滑）
#   wind      —— 恒定外扰 (m/s): 位置积分叠加 [wx, wy]（等效风/坡道漂移）
# 开关: MISMATCH_ENABLE=False 时本块全部无效，dynamics 与原始代码逐位一致（默认关闭）
MISMATCH_ENABLE =  True             # 总开关（跑失配实验时改 True）
MISMATCH_PRESET = "mild"              # 档位: off / mild / strong（可自行加档或改数值）
MISMATCH_PRESETS = {
    #            线速度增益   角速度增益   侧滑角       恒定外扰 (wx, wy)
    "off":    {"k_v": 1.00, "k_w": 1.00, "slip_beta": 0.00, "wind": (0.00, 0.00)},
    "mild":   {"k_v": 0.90, "k_w": 1.10, "slip_beta": 0.03, "wind": (0.00, 0.00)},
    "strong": {"k_v": 0.80, "k_w": 1.20, "slip_beta": 0.06, "wind": (0.05, 0.05)},
}
MISMATCH = MISMATCH_PRESETS[MISMATCH_PRESET]


def wrap_angle(a):
    """角度折叠到 (-π, π]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def dynamics(x, u):
    """一步欧拉动力学（P02 式(36): x_{k+1} = x_k + δ·g(x_k)·u_k）
    x:(3,) 状态 [x,y,θ]，u:(2,) 输入 [v,ω] -> x_next:(3,)

    模型失配（实验开关，默认关闭）: MISMATCH_ENABLE=True 时，"真实系统"与 MPC 标称模型
    （tracking_mpc.py 中的理想 g(x)，增益=1）故意不一致——执行增益 k_v/k_w、侧滑 β、
    恒定外扰 wind；MPC 内部模型不变 ⇒ 形成标准的 plant ≠ model 失配设定"""
    theta = x[2]
    u = np.asarray(u, float)
    if MISMATCH_ENABLE:
        v_act = MISMATCH["k_v"] * u[0]                    # 真实线速度（带执行增益偏差）
        w_act = MISMATCH["k_w"] * u[1]                    # 真实角速度（带执行增益偏差）
        th_v  = theta + MISMATCH["slip_beta"]             # 速度方向 = 车头朝向 + 侧滑角
        x_next = x + DT * np.array([
            v_act * np.cos(th_v) + MISMATCH["wind"][0],   # 位置积分 + 恒定外扰
            v_act * np.sin(th_v) + MISMATCH["wind"][1],
            w_act])
    else:
        g = np.array([[np.cos(theta), 0.0],
                      [np.sin(theta), 0.0],
                      [0.0, 1.0]])          # g(x_k)（3x2）
        x_next = x + DT * g @ u
    x_next[2] = wrap_angle(x_next[2])
    return x_next


def tracking_error(x, x_ref):
    """跟踪误差 e = [x−x_ref, y−y_ref, Δθ]（3,），角度差用 wrap 归一化"""
    e = np.array(x, float) - np.array(x_ref, float)
    e[2] = wrap_angle(e[2])
    return e


def stage_cost(x, u, x_ref, u_ref):
    """单步【评价代价】（评分口径: SCORE_W 加权归一化平方和；与动作无关）
    ℓ = Σ SCORE_W[:3]·(e/ERR_SCALE)² + Σ SCORE_W[3:]·(Δu/CTRL_SCALE)²
    口径由 SCORE_MODE 切换（balanced/error/control，见文件头部）；全流程唯一评分定义
    注意: MPC 内部代价 = "归一化信号 + 5 个可学权重"，是另一回事（评分不随动作变）"""
    e_n  = tracking_error(x, x_ref) / ERR_SCALE
    du_n = (np.asarray(u, float) - np.asarray(u_ref, float)) / CTRL_SCALE
    return float(SCORE_W[:3] @ (e_n ** 2) + SCORE_W[3:] @ (du_n ** 2))
#[:3]代表取前3个元素，即误差部分的权重，后两项是控制部分的权重。
#[3:]代表取从第4个元素开始的所有元素，即控制部分的权重。


def decode_action(a_normalized):
    """归一化动作 a∈[-1,1]^5 → 权重向量 w = [q1,q2,q3, r1,r2]（作用在归一化信号上）
    每个分量对数均匀映射到 [WEIGHT_MIN, WEIGHT_MAX]，保正且有界"""
    a = np.clip(np.asarray(a_normalized, float), -1.0, 1.0)
    return WEIGHT_MIN * (WEIGHT_MAX / WEIGHT_MIN) ** ((a + 1.0) / 2.0)
    # w = w_min(w_max/w_min)^((a+1)/2) 


def norm_action_of_weights(w):
    """decode_action 的逆: 由 5 个权重反解归一化动作（评估固定权重基线用）"""
    w = np.asarray(w, float)
    return 2.0 * np.log(w / WEIGHT_MIN) / np.log(WEIGHT_MAX / WEIGHT_MIN) - 1.0


def physical_weights_of(w_norm):
    """换算: 归一化权重 → 物理单位权重（报告用）"""
    w = np.asarray(w_norm, float)
    return np.concatenate([w[:3] / ERR_SCALE ** 2, w[3:] / CTRL_SCALE ** 2])


class ReferenceTrajectory:
    """解析参数化参考轨迹（标量或数组时间均可调用）

    轨迹族 family:
      'line' —— 匀速直线: 朝向 θ0、速度 v
      'arc'  —— 匀速圆弧: 曲率 κ (rad/m)，转弯率 ω = κ·v（半径 1/|κ|）
      'sine' —— 正弦侧偏: 沿 θ0 基准直线前进 s=v·τ，叠加横向偏移 A·sin(2πτ/T + φ)

    at(τ) 返回单点 (x_ref(3,), u_ref(2,))；
    sequence(t_array) 返回整段 ((T,3), (T,2))；
    u_ref 为精确前馈输入: v=|ṗ(τ)|, ω=dθ/dτ（sine 族速度随侧偏变化，故解析求导给出）
    """

    def __init__(self, family, p0, theta0, v, kappa=0.0, amp=0.0, period=4.0, phase=0.0):
        self.family = family            # 轨迹族
        self.p0 = np.asarray(p0, float) # 起点位置 (2,)
        self.theta0 = float(theta0)     # 基准朝向
        self.v = float(v)               # 基准速度 (m/s)
        self.kappa = float(kappa)       # 曲率（仅 'arc' 用）
        self.amp = float(amp)           # 横向振幅（仅 'sine' 用）
        self.period = float(period)     # 侧偏周期（仅 'sine' 用）
        self.phase = float(phase)       # 侧偏相位（仅 'sine' 用）

    def _eval(self, tau):#eval的完整变量名是 evaluate，_eval是私有方法，表示内部使用的evaluate函数
        """核心计算: tau 为数组 → X_ref (T,3), U_ref (T,2)"""
        tau = np.atleast_1d(np.asarray(tau, float))
        T = tau.size
        X = np.zeros((T, 3))
        U = np.zeros((T, 2))
        th0, v = self.theta0, self.v

        if self.family == "line":
            X[:, 0] = self.p0[0] + v * tau * np.cos(th0)
            X[:, 1] = self.p0[1] + v * tau * np.sin(th0)
            X[:, 2] = th0
            U[:, 0] = v

        elif self.family == "arc":
            k = self.kappa
            w = k * v                              # 转弯率 ω = κ·v
            th = th0 + w * tau
            X[:, 0] = self.p0[0] + (np.sin(th) - np.sin(th0)) / k
            X[:, 1] = self.p0[1] - (np.cos(th) - np.cos(th0)) / k
            X[:, 2] = wrap_angle(th)
            U[:, 0] = v
            U[:, 1] = w

        elif self.family == "sine":
            # 基准直线前进 s = v·τ，横向偏移 d = A·sin(2πτ/T + φ)
            om = 2 * np.pi / self.period
            s = v * tau
            d = self.amp * np.sin(om * tau + self.phase)
            dd = self.amp * om * np.cos(om * tau + self.phase)        # ḋ
            ddd = -self.amp * om**2 * np.sin(om * tau + self.phase)   # d̈
            ct, st = np.cos(th0), np.sin(th0)
            # 位置与速度（固定基准方向 + 横向偏移）
            vx = v * ct - dd * st
            vy = v * st + dd * ct
            ax = -ddd * st              # 加速度
            ay = ddd * ct
            X[:, 0] = self.p0[0] + s * ct - d * st
            X[:, 1] = self.p0[1] + s * st + d * ct
            X[:, 2] = np.arctan2(vy, vx)
            spd2 = vx**2 + vy**2
            U[:, 0] = np.sqrt(spd2)                   # v_ref = |ṗ|
            U[:, 1] = (vx * ay - vy * ax) / spd2      # ω_ref = dθ/dτ

        else:
            raise ValueError(f"未知轨迹族: {self.family}")
        return X, U

    def at(self, tau):
        """单点求值: 返回 (x_ref(3,), u_ref(2,))"""
        X, U = self._eval(np.array([tau], float))
        return X[0], U[0]

    def sequence(self, t_array):
        """整段求值: t_array (T,) → X_ref (T,3), U_ref (T,2)"""
        return self._eval(t_array)

    def __repr__(self):#repr__ 方法用于返回对象的字符串表示，通常用于调试和日志记录。
        # 轨迹族参数化信息（打印/记录用）
        if self.family == "arc":
            extra = f", kappa={self.kappa:+.3f} (R={1/abs(self.kappa):.2f} m)"
        elif self.family == "sine":
            extra = f", A={self.amp:.2f}, T={self.period:.2f}, phi={self.phase:.2f}"
        else:
            extra = ""
        return (f"ReferenceTrajectory({self.family}, "
                f"p0=({self.p0[0]:+.2f},{self.p0[1]:+.2f}), "
                f"theta0={self.theta0:+.2f}, v={self.v:.2f}{extra})")


def sample_reference(rng, n_steps, dt=DT, family=None):
    """随机生成一条参考轨迹（族、参数、起点全随机），并预采样整段序列
    参数:
      rng     —— np.random.Generator（保证可复现）
      n_steps —— 采样点数（环境里取 回合步数 + 预测时域，覆盖 MPC 参考窗口）
      family  —— None 时随机选族；调试时可固定为 'line'/'arc'/'sine'
    说明: sine 族振幅自动受限，保证参考峰值速度 sqrt(v²+(A·ω)²) ≤ REF_SPEED_CAP
          （参考物理可行；line/arc 速度恒为 v ≤ 0.8，天然满足）
    返回: (ref, X_ref (n_steps,3), U_ref (n_steps,2))
    """
    if family is None:
        family = rng.choice(["line", "arc", "sine"])
    v = rng.uniform(0.3, 0.8)                    # 基准速度 (m/s)
    theta0 = rng.uniform(-np.pi, np.pi)          # 基准朝向
    p0 = rng.uniform(-1.5, 1.5, size=2)          # 起点位置

    kwargs = {}
    if family == "arc":
        # 曲率 ±[0.2, 0.8] rad/m → 半径 1.25~5 m；|ω_ref| = |κ|v ≤ 0.64 rad/s
        kwargs["kappa"] = rng.uniform(0.2, 0.8) * rng.choice([-1.0, 1.0])
    elif family == "sine":
        kwargs["period"] = rng.uniform(3.0, 8.0)  # 侧偏周期 (s)
        kwargs["phase"] = rng.uniform(0.0, 2 * np.pi)
        # 振幅受"参考峰值速度"约束: sqrt(v² + (A·ω)²) ≤ REF_SPEED_CAP
        # （不设限会生成参考速度超过执行器上限 v_max=1 的不可行轨迹，
        #   那样的跟踪误差不可消除、对比实验不可解释——诊断发现，2026-09-14）
        om = 2 * np.pi / kwargs["period"]
        amp_cap = np.sqrt(max(REF_SPEED_CAP ** 2 - v ** 2, 0.0)) / om
        kwargs["amp"] = float(min(rng.uniform(0.3, 1.0), amp_cap))

    ref = ReferenceTrajectory(family, p0, theta0, v, **kwargs)
    t_arr = np.arange(n_steps) * dt
    X_ref, U_ref = ref.sequence(t_arr)
    return ref, X_ref, U_ref


def sample_initial_state(rng, ref, pos_max=0.4, theta_max=np.pi / 6):
    #位置偏移最大为 0.4 m，朝向偏移最大为 ±30°（π/6 rad）
    """随机初始状态 = 参考轨迹起点 + 随机偏移
    位置偏移模长 ∈ [0.1, pos_max]，朝向偏移 ∈ ±theta_max（默认 ±30°）"""
    x_ref0, _ = ref.at(0.0)
    ang = rng.uniform(-np.pi, np.pi)             # 偏移方向
    r = rng.uniform(0.1, pos_max)                # 偏移模长
    x0 = x_ref0.copy()
    x0[0] += r * np.cos(ang)
    x0[1] += r * np.sin(ang)
    x0[2] = wrap_angle(x_ref0[2] + rng.uniform(-theta_max, theta_max))
    return x0
