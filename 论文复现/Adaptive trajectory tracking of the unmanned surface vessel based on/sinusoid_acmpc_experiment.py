"""复现论文 Fig.5-Fig.10（场景 1：正弦参考轨迹）的实验与绘图脚本。

阶段（--stage）：
    calibrate  按论文 Fig.6 的 MPC 峰值误差标定环境干扰放大系数
    train      训练 3 个 Actor-Critic 智能体（无干扰/固定步长、有干扰/固定步长、有干扰/变步长）
    simulate   用训练好的智能体与纯 MPC 做 5 组跟踪仿真
    plot       绘制 Fig.5 - Fig.10
    all        依次执行以上全部

示例：
    python sinusoid_acmpc_experiment.py --stage all
    python sinusoid_acmpc_experiment.py --stage train --cases dist_fixed
    python sinusoid_acmpc_experiment.py --stage plot
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cs2_model import (CyberShipII, NoDisturbance, WindCurrentDisturbance,
                       reference_error)
from sinusoid_reference import SinusoidReference
from ltv_mpc_tracker import LTVMPCTracker
from actor_critic_tuner import ActorCriticTuner, select_horizon

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figures"

# 案例名 -> (是否加环境干扰, 预测步长模式)
CASES = {
    "clean_fixed": (False, "fixed"),
    "dist_fixed": (True, "fixed"),
    "dist_variable": (True, "variable"),
}


@dataclass
class ExperimentConfig:
    # 仿真设置：Ts = 1 s，1500 步 = 1500 s，屏幕上 x 轴就是论文 Fig.6/9 的 0-1500 Time/s。
    # 18 m / 1500 s -> 航迹参考纵速 0.012 m/s（论文未给，由 Fig.5 的 x 范围反推）。
    ts: float = 1.0
    steps: int = 1500
    substeps: int = 4
    # 参考轨迹：x_d = vx*t，y_d = A sin(2*pi*x_d/P)
    ref_amplitude: float = 10.0
    ref_period: float = 12.0
    ref_vx: float = 0.012
    # MPC（tau_max 取物理合理的 4 N / 1 N·m：8 N 时起始猛打艏摇会转过头，
    # 而稳态所需推力只有 0.05-3 N，改动限幅不影响稳态精度）
    q_init: tuple = (1.0, 1.0, 1.0)
    r_weight: float = 1e-2
    n_fixed: int = 5
    tau_max: tuple = (4.0, 4.0, 1.0)
    tau_rate: tuple = (3.0, 3.0, 1.0)
    horizon_mode: str = "table5"
    # 变步长的 ΔR 口径：step = 相邻控制步奖励差（式(30) 的字面读法，Fig.11 的 N 在
    # 一条轨迹内连续变化即此意）；episode = 相邻训练回合总回报之差
    horizon_update: str = "step"
    # Table 5 的阈值是 20/50 量级，而式(24) 的单步奖励只有 0.1 量级，
    # 因此 ΔR 需按 "论文奖励量纲 ≈ 100 × 米制奖励" 折算（由 Table 5 反推的假设）。
    # 只作用于 N 的选取，不改变 RL 的奖励与 Fig.10 的纵轴。
    horizon_reward_scale: float = 100.0
    reward_scale: float = 1.0
    # AC
    episodes: int = 300
    gamma: float = 0.98
    learning_rate: float = 1e-3
    hidden: tuple = (264, 128)
    q_min: float = 0.1
    # 上界按论文 Fig.7 反推：有干扰工况 Q(2,2) 最高到约 19
    q_max: float = 20.0
    action_step: float = 0.1
    entropy_weight: float = 0.01
    eval_greedy: bool = False
    eval_seeds: int = 5
    seed: int = 0
    # 环境干扰（Table 4）
    wind_speed: float = 0.3
    wind_dir: float = 45.0
    current_speed: float = 0.03
    current_dir: float = 30.0
    disturbance_scale: float = 100.0
    target_peak_error: float = 0.45


# --------------------------------------------------------------------- 构件
def build_reference(cfg):
    return SinusoidReference(cfg.ref_amplitude, cfg.ref_period, cfg.ref_vx)


def build_plant(cfg, disturbed, scale=None):
    if not disturbed:
        return CyberShipII(cfg.ts, NoDisturbance(), substeps=cfg.substeps)
    dist = WindCurrentDisturbance(cfg.wind_speed, cfg.wind_dir,
                                  cfg.current_speed, cfg.current_dir,
                                  cfg.disturbance_scale if scale is None else scale)
    return CyberShipII(cfg.ts, dist, substeps=cfg.substeps)


def build_tracker(cfg, plant):
    return LTVMPCTracker(plant, ts=cfg.ts, q_diag=cfg.q_init,
                         r_diag=(cfg.r_weight,) * 3, n_horizon=cfg.n_fixed,
                         tau_max=cfg.tau_max, tau_rate=cfg.tau_rate)

# ------------------------------------------------------------------- 仿真
def run_rollout(cfg, ref, plant, tracker, agent=None, n_mode="fixed", n_value=None,
                seed=None, greedy=None):
    """跑一条完整轨迹，返回轨迹/误差/Q 值/奖励等记录。

    n_mode="fixed"：整条轨迹用 n_value（缺省 cfg.n_fixed）作为预测步长。
    n_mode="step" ：按相邻控制步的奖励增量逐步选 N（式(30)(32) 的字面读法，
                    需要 reward_scale 把奖励放大到 Table 5 的量级）。
    """
    if seed is not None and agent is not None:
        agent.rng = np.random.default_rng(seed)
    use_greedy = cfg.eval_greedy if greedy is None else bool(greedy)
    tracker.solver.reset_warm_start()

    steps, ts = cfg.steps, cfg.ts
    state = np.zeros(6)
    q = np.array(cfg.q_init, dtype=float)
    u_prev = np.zeros(3)
    n_const = cfg.n_fixed if n_value is None else int(n_value)

    pos = np.zeros((steps + 1, 2))
    eta_d = np.zeros((steps + 1, 3))
    err = np.zeros(steps + 1)
    psi_err = np.zeros(steps + 1)
    q_hist = np.zeros((steps, 3))
    n_hist = np.zeros(steps, dtype=int)
    rewards = np.zeros(steps)
    states = np.zeros((steps, 3))
    actions = np.zeros((steps, 3), dtype=int)

    pos[0] = state[:2]
    eta_d[0] = ref.state(0.0)[0]
    err[0], psi_err[0] = reference_error(state, eta_d[0])

    for k in range(steps):
        t = k * ts
        if n_mode == "step":
            delta_r = 0.0 if k < 2 else float(
                cfg.horizon_reward_scale * (rewards[k - 1] - rewards[k - 2]))
            n_now = select_horizon(delta_r, mode=cfg.horizon_mode)
        else:
            n_now = n_const

        if agent is not None:
            states[k] = q
            actions[k], _, q = agent.act(q, greedy=use_greedy)

        tau = tracker.solve(state, t, ref, q_diag=q, n_horizon=n_now, u_prev=u_prev)
        u_prev = tau
        state = plant.step(state, tau, t)

        t_next = t + ts
        eta_d[k + 1] = ref.state(t_next)[0]
        pos[k + 1] = state[:2]
        err[k + 1], psi_err[k + 1] = reference_error(state, eta_d[k + 1])
        q_hist[k] = q
        n_hist[k] = n_now
        rewards[k] = cfg.reward_scale * (-err[k + 1] - abs(psi_err[k + 1]))   # 式(24)

    return dict(pos=pos, eta_d=eta_d, err=err, psi_err=psi_err, q_hist=q_hist,
                n_hist=n_hist, rewards=rewards, states=states, actions=actions,
                total_reward=float(np.sum(rewards)), final_state=state)


# ------------------------------------------------------------------ 标定
def steady_peak(err, steps):
    """稳态段（25%-75% 时间窗）的位置误差峰值。

    论文 Fig.6 中起始转向的暂态误差约 0.95 m，并不参与其报告的 0.45 m / 0.25 m
    峰值，因此统计峰值时同样避开起始暂态。
    """
    lo, hi = int(0.25 * steps), int(0.75 * steps)
    return float(np.max(err[lo:hi]))


def calibrate_disturbance(cfg, log=print):
    """二分搜索干扰放大系数，使传统 MPC 的稳态峰值误差接近论文的 0.45 m。"""
    ref = build_reference(cfg)
    target = cfg.target_peak_error
    lo, hi = 0.5, 300.0
    best_diff, best_scale, best_peak = np.inf, cfg.disturbance_scale, 0.0
    log(f"[calibrate] 目标 MPC 稳态峰值误差 = {target:.2f} m")
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        plant = build_plant(cfg, True, scale=mid)
        tracker = build_tracker(cfg, plant)
        out = run_rollout(cfg, ref, plant, tracker)
        peak = steady_peak(out["err"], cfg.steps)
        log(f"[calibrate] scale = {mid:8.2f} -> 稳态峰值 = {peak:.3f} m")
        if abs(peak - target) < best_diff:
            best_diff, best_scale, best_peak = abs(peak - target), mid, peak
        if peak < target:
            lo = mid
        else:
            hi = mid
    log(f"[calibrate] 选定 scale = {best_scale:.2f}（稳态峰值 {best_peak:.3f} m）")
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "disturbance_scale.json").write_text(
        json.dumps({"disturbance_scale": best_scale, "steady_peak": best_peak},
                   indent=2), encoding="utf-8")
    return float(best_scale)


# ------------------------------------------------------------------ 训练
def agent_path(case):
    return RESULT_DIR / f"agent_{case}.npz"


def train_case(cfg, case, scale, log=print):
    disturbed, n_mode = CASES[case]
    ref = build_reference(cfg)
    plant = build_plant(cfg, disturbed, scale=scale)
    tracker = build_tracker(cfg, plant)
    agent = ActorCriticTuner(q_min=cfg.q_min, q_max=cfg.q_max, dq=cfg.action_step,
                             hidden=cfg.hidden, lr=cfg.learning_rate, gamma=cfg.gamma,
                             seed=cfg.seed, entropy_weight=cfg.entropy_weight)

    rollout_mode = "step" if (n_mode == "variable" and cfg.horizon_update == "step") else "fixed"
    rewards = np.zeros(cfg.episodes)
    n_per_episode = np.zeros(cfg.episodes, dtype=int)
    n_now = cfg.n_fixed
    prev_total = None
    delta_r = 0.0
    t_start = time.time()
    log(f"[train:{case}] 干扰={disturbed} 步长模式={n_mode} "
        f"ΔR口径={cfg.horizon_update if n_mode == 'variable' else '-'} 回合数={cfg.episodes}")
    for ep in range(cfg.episodes):
        if n_mode == "variable" and cfg.horizon_update == "episode" and prev_total is not None:
            n_now = select_horizon(delta_r, mode=cfg.horizon_mode)
        out = run_rollout(cfg, ref, plant, tracker, agent=agent,
                          n_mode=rollout_mode, n_value=n_now)
        agent.update(out["states"], out["actions"], out["rewards"])
        rewards[ep] = out["total_reward"]
        n_now = int(round(float(np.mean(out["n_hist"]))))
        n_per_episode[ep] = n_now
        if prev_total is not None:
            delta_r = float(rewards[ep] - prev_total)
        prev_total = rewards[ep]
        last = out

        done = ep + 1
        if done % 10 == 0 or done == cfg.episodes:
            elapsed = time.time() - t_start
            per_ep = elapsed / done
            eta = per_ep * (cfg.episodes - done)
            spread = ""
            if rollout_mode == "step":
                vals = np.unique(out["n_hist"])
                spread = f" 本回合N∈{vals.tolist()}"
            log(f"[train:{case}] 回合 {done:3d}/{cfg.episodes} "
                f"奖励 {rewards[ep]:8.2f} N≈{n_now:2d} ΔR={delta_r:7.1f}{spread} "
                f"已用 {elapsed/60:5.1f} min 预计剩余 {eta/60:5.1f} min")

    cells = np.unique(n_per_episode, return_counts=True)
    log(f"[train:{case}] 回合平均 N 的分布: {dict(zip(cells[0].tolist(), cells[1].tolist()))}")
    if rollout_mode == "step":
        vals, cnt = np.unique(last["n_hist"], return_counts=True)
        log(f"[train:{case}] 末回合逐步 N 的分布: {dict(zip(vals.tolist(), cnt.tolist()))}")
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez(agent_path(case), rewards=rewards, q_hist=last["q_hist"],
             n_hist=last["n_hist"], n_per_episode=n_per_episode,
             final_horizon=int(n_per_episode[-1]), **agent.state_dict())
    return rewards


def load_agent(cfg, case):
    path = agent_path(case)
    if not path.exists():
        raise SystemExit(f"缺少 {path.name}，请先运行：--stage train --cases {case}")
    data = np.load(path)
    agent = ActorCriticTuner(q_min=cfg.q_min, q_max=cfg.q_max, dq=cfg.action_step,
                             hidden=cfg.hidden, lr=cfg.learning_rate, gamma=cfg.gamma,
                             seed=cfg.seed, entropy_weight=cfg.entropy_weight)
    agent.load_state_dict(data)
    return agent, data


# ---------------------------------------------------------------- 仿真阶段
AC_CASES = (("ac_clean", "clean_fixed", False),
            ("ac_dist_fixed", "dist_fixed", True),
            ("ac_dist_variable", "dist_variable", True))


def _run_seeds(cfg, ref, plant, tracker, agent, n_mode, n_value, policy, log, name):
    """同一策略跑 cfg.eval_seeds 个随机种子，返回各条 rollout 与统计量。"""
    outs, peaks, global_peaks = [], [], []
    for i in range(cfg.eval_seeds):
        out = run_rollout(cfg, ref, plant, tracker, agent=agent, n_mode=n_mode,
                          n_value=n_value, seed=1000 + i, greedy=policy == "greedy")
        outs.append(out)
        peaks.append(steady_peak(out["err"], cfg.steps))
        global_peaks.append(float(np.max(out["err"])))
    stat = (float(np.mean(peaks)), float(np.std(peaks)),
            float(np.mean(global_peaks)))
    log(f"[simulate] {name:16s} [{policy:10s}] 稳态峰值 "
        f"{stat[0]:.3f} ± {stat[1]:.3f} m   (起始暂态峰值 {stat[2]:.3f} m)")
    return outs, stat


def simulate(cfg, scale, log=print):
    ref = build_reference(cfg)
    results, stats, dp_stats = {}, {}, {}

    # 纯 MPC 是确定性的，只跑一次
    for name, disturbed in (("mpc_clean", False), ("mpc_dist", True)):
        plant = build_plant(cfg, disturbed, scale=scale)
        tracker = build_tracker(cfg, plant)
        results[name] = run_rollout(cfg, ref, plant, tracker)
        peak = steady_peak(results[name]["err"], cfg.steps)
        dp_stats[name] = (peak, 0.0, float(np.max(results[name]["err"])))
        log(f"[simulate] {name:16s} [deterministic] 稳态峰值 {peak:.3f} m   "
            f"(起始暂态峰值 {dp_stats[name][2]:.3f} m)")

    # AC 案例：多种子 + greedy/随机 两种评估
    seed_outs = {}
    for name, case, disturbed in AC_CASES:
        agent, data = load_agent(cfg, case)
        final_n = int(data["final_horizon"]) if "final_horizon" in data.files else cfg.n_fixed
        n_mode = "step" if (CASES[case][1] == "variable"
                            and cfg.horizon_update == "step") else "fixed"
        plant = build_plant(cfg, disturbed, scale=scale)
        tracker = build_tracker(cfg, plant)
        outs, stat = _run_seeds(cfg, ref, plant, tracker, agent, n_mode, final_n,
                                "stochastic", log, name)
        results[name] = outs[0]
        seed_outs[name] = outs
        stats[name] = stat
        if cfg.eval_seeds > 1:
            _run_seeds(cfg, ref, plant, tracker, agent, n_mode, final_n,
                       "greedy", log, name)

    # 轨迹用种子 0，误差曲线用多种子均值±标准差
    payload = {"names": np.array(list(results.keys())), "steps": cfg.steps}
    for name, out in results.items():
        payload[f"{name}_pos"] = out["pos"].reshape(-1)
        payload[f"{name}_psi"] = out["psi_err"]
        if name in seed_outs:
            errs = np.array([o["err"] for o in seed_outs[name]])
            payload[f"{name}_err"] = errs.mean(axis=0)
            payload[f"{name}_err_std"] = errs.std(axis=0)
        else:
            payload[f"{name}_err"] = out["err"]
            payload[f"{name}_err_std"] = np.zeros_like(out["err"])
    np.savez(RESULT_DIR / "rollouts.npz", **payload,
             q_dist_fixed=results["ac_dist_fixed"]["q_hist"],
             q_clean=results["ac_clean"]["q_hist"],
             n_dist_variable=results["ac_dist_variable"]["n_hist"])
    write_summary(cfg, dp_stats, stats)
    return results


def write_summary(cfg, dp_stats, stats):
    """把均值±标准差写进 results/summary.md，方便直接抄进论文/答复。"""
    lines = ["# 复现结果汇总（稳态段 25%-75% 峰值误差）", "",
             "| 工况 | 峰值误差 (m) | 起始暂态峰值 (m) |", "| --- | --- | --- |"]
    for name, (mean, std, gpeak) in {**dp_stats, **stats}.items():
        spread = f"{mean:.3f} ± {std:.3f}" if std > 0 else f"{mean:.3f}"
        lines.append(f"| {name} | {spread} | {gpeak:.3f} |")
    lines += ["", f"评估种子数 = {cfg.eval_seeds}（AC 案例为随机策略的均值±标准差）",
              f"q_max = {cfg.q_max}, entropy_weight = {cfg.entropy_weight}",
              f"ΔR 口径 = {cfg.horizon_update}, N 固定值 = {cfg.n_fixed}"]
    (RESULT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_rollouts(cfg):
    data = np.load(RESULT_DIR / "rollouts.npz")
    steps = int(data["steps"])
    out = {}
    for name in data["names"]:
        name = str(name)
        out[name] = dict(pos=data[f"{name}_pos"].reshape(-1, 2),
                         err=data[f"{name}_err"],
                         err_std=data[f"{name}_err_std"],
                         psi_err=data[f"{name}_psi"])
    out["q_dist_fixed"] = data["q_dist_fixed"]
    out["q_clean"] = data["q_clean"]
    out["n_dist_variable"] = data["n_dist_variable"]
    return out


# ------------------------------------------------------------------ 绘图
COLORS = {
    "expected": "#c4161c",
    # 两条有干扰曲线都用蓝色（与论文一致），靠线型区分；明度略拉开，
    # 免得局部放大图里在 6pt 字号下看不清
    "mpc_dist": "#4166e8",
    "mpc_clean": "#7a7a7a",
    "ac_dist": "#0d2a9e",
    "ac_clean": "#a9743a",
    "fixed": "#1b3ec8",
    "variable": "#12a523",
}


def add_environment_arrows(ax, cfg):
    """Fig.5/Fig.8 中的流、风矢量场。方向按气象学“来向”取反。"""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xs = np.arange(0.6, xlim[1], 1.9)
    ys = np.arange(ylim[0] + 1.0, ylim[1], 2.6)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ang_c = np.deg2rad(cfg.current_dir) + np.pi
    ang_w = np.deg2rad(cfg.wind_dir) + np.pi
    # 两者方向只差 15°，错开起点并让流箭头略长，避免互相遮盖
    ax.quiver(grid_x, grid_y, np.cos(ang_w), np.sin(ang_w), color="#cfcfcf",
              angles="xy", scale_units="xy", scale=3.0, width=0.0028, headwidth=3.4)
    ax.quiver(grid_x - 0.28, grid_y + 0.30, np.cos(ang_c), np.sin(ang_c),
              color="#7fb2e0", angles="xy", scale_units="xy", scale=3.4,
              width=0.0032, headwidth=3.6, zorder=1.5)


def add_zoom(ax, bounds, rect, series, title=None):
    x0, x1, y0, y1 = bounds
    inset = ax.inset_axes(rect)
    for xv, yv, kw in series:
        inset.plot(xv, yv, **kw)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.tick_params(labelsize=6, length=2)
    if title:
        inset.set_title(title, fontsize=7)
    ax.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.65, lw=0.8)
    return inset


def figure_trajectory(cfg, ref, rollouts, path, with_step_size):
    fig, ax = plt.subplots(figsize=(7.9, 6.6))
    x_drawn = max(float(np.max(r["pos"][:, 0])) for r in rollouts.values()
                  if isinstance(r, dict))
    _, xd, yd = ref.sample(x_drawn / ref.vx, 700)
    ax.plot(xd, yd, color=COLORS["expected"], lw=2.0, label="Expected Trajectory", zorder=3)

    # 主图与局部放大共用同一套 (key, color, linestyle)，否则 inset 里两条蓝线会都变成实线
    if with_step_size:
        curves = (("ac_dist_fixed", COLORS["fixed"], "-", "Fixed Step Size", 1.6, 4),
                  ("ac_dist_variable", COLORS["variable"], "-", "Variable Step Size", 1.6, 5))
    else:
        curves = (("mpc_dist", COLORS["mpc_dist"], "--", "MPC with Disturbance", 1.6, 4),
                  ("mpc_clean", COLORS["mpc_clean"], "-", "MPC without Disturbance", 1.2, 4),
                  ("ac_dist_fixed", COLORS["ac_dist"], "-", "AC-MPC with Disturbance", 1.6, 5),
                  ("ac_clean", COLORS["ac_clean"], "-", "AC-MPC without Disturbance", 1.2, 5))
    for key, color, style, label, width, zorder in curves:
        ax.plot(*rollouts[key]["pos"].T, color=color, lw=width, ls=style,
                label=label, zorder=zorder)

    ax.plot([0.0], [0.0], "o", color="#a11212", ms=5, zorder=6, label="Initial Position")
    ax.set_xlim(-0.6, 18.6)
    ax.set_ylim(-18.5, 21.5)
    add_environment_arrows(ax, cfg)

    # 放大图只画有干扰的两条 + 期望轨迹，线型与主图一致
    zoom_keys = (curves if with_step_size else (curves[0], curves[2]))
    series = [(rollouts[c[0]]["pos"][:, 0], rollouts[c[0]]["pos"][:, 1],
               dict(color=c[1], ls=c[2], lw=1.1)) for c in zoom_keys]
    series.append((xd, yd, dict(color=COLORS["expected"], lw=1.2)))
    add_zoom(ax, (2.55, 3.75, 9.25, 10.55), [0.27, 0.80, 0.21, 0.17], series,
             "Local Trajectory Comparison")
    add_zoom(ax, (8.4, 10.6, -10.6, -8.8), [0.16, 0.18, 0.22, 0.19], series)
    add_zoom(ax, (11.9, 13.1, -2.1, 2.1), [0.80, 0.44, 0.16, 0.16], series)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#9dc3e6", lw=1.4, marker=">", ms=4))
    labels.append("Current")
    handles.append(Line2D([], [], color="#cfcfcf", lw=1.4, marker=">", ms=4))
    labels.append("Wind")
    ax.legend(handles, labels, loc="upper right", fontsize=8.5, framealpha=0.95,
              handlelength=2.6)

    ax.set_xlabel("x Position/m", fontsize=11)
    ax.set_ylabel("y Position/m", fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure_error(cfg, rollouts, path, with_step_size):
    fig, ax = plt.subplots(figsize=(7.9, 6.4))
    axis = np.arange(rollouts[next(iter(rollouts))]["err"].size)
    if with_step_size:
        series = (("ac_dist_fixed", COLORS["fixed"], "--", "Fixed Step Size"),
                  ("ac_dist_variable", COLORS["variable"], "-", "Variable Step Size"))
    else:
        series = (("mpc_dist", COLORS["mpc_dist"], "--", "MPC Error with Disturbance"),
                  ("mpc_clean", COLORS["mpc_clean"], "-", "MPC Error without Disturbance"),
                  ("ac_dist_fixed", COLORS["ac_dist"], ":", "AC-MPC Error with Disturbance"),
                  ("ac_clean", COLORS["ac_clean"], "--", "AC-MPC Error without Disturbance"))
    for key, color, style, label in series:
        mean = rollouts[key]["err"]
        std = rollouts[key].get("err_std")
        ax.plot(axis, mean, color=color, lw=1.5, ls=style, label=label)
        if std is not None and np.max(std) > 1e-4:
            ax.fill_between(axis, np.maximum(mean - std, 0.0), mean + std,
                            color=color, alpha=0.18, lw=0)
    ax.set_xlim(0, cfg.steps)
    peaks = [np.max(r["err"]) for k, r in rollouts.items() if isinstance(r, dict)]
    ax.set_ylim(0, max(2.0, 1.06 * max(peaks)))
    ax.set_xlabel("Time/s", fontsize=11)
    ax.set_ylabel("Position Error/m", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure_q_values(cfg, rollouts, path):
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 5.6), sharex=True)
    colors = ("#1234d0", "#e01b1b", "#12b312")
    for col, (key, title) in enumerate((("q_clean", "Without environmental disturbance"),
                                        ("q_dist_fixed", "With environmental disturbance"))):
        q_hist = rollouts[key]
        axis = np.arange(q_hist.shape[0])
        for row in range(3):
            ax = axes[row, col]
            ax.plot(axis, q_hist[:, row], color=colors[row], lw=1.1)
            ax.set_ylabel("Q Value", fontsize=9)
            ax.set_title(f"Q({row + 1},{row + 1})", fontsize=9)
            if row == 2:
                ax.set_xlabel("Time/s", fontsize=9)
        axes[0, col].annotate(title, xy=(0.5, 1.35), xycoords="axes fraction",
                              ha="center", fontsize=11)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def moving_average(values, window):
    window = max(1, int(window))
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def figure_reward(cfg, rewards_fixed, rewards_variable, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharey=False)
    for ax, rewards, title in ((axes[0], rewards_fixed, "(a) Fixed step size"),
                               (axes[1], rewards_variable, "(b) Variable step size")):
        rounds = np.arange(1, rewards.size + 1)
        ax.plot(rounds, rewards, color="#1234d0", lw=1.0, marker="o", ms=3.2,
                mec="#1234d0", mfc="#1234d0", label="Episode Reward")
        ax.plot(rounds, moving_average(rewards, 15), color="#e01b1b", lw=3.0,
                label="Average Reward")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Number of rounds", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_all(cfg, rollouts, rewards, log=print):
    FIGURE_DIR.mkdir(exist_ok=True)
    ref = build_reference(cfg)

    figure_trajectory(cfg, ref, rollouts, FIGURE_DIR / "fig05_sinusoid_trajectory.png", False)
    figure_error(cfg, rollouts, FIGURE_DIR / "fig06_sinusoid_error.png", False)
    figure_q_values(cfg, rollouts, FIGURE_DIR / "fig07_sinusoid_q_value.png")
    figure_trajectory(cfg, ref, rollouts, FIGURE_DIR / "fig08_sinusoid_trajectory_variable_step.png", True)
    figure_error(cfg, rollouts, FIGURE_DIR / "fig09_sinusoid_error_variable_step.png", True)
    figure_reward(cfg, rewards["dist_fixed"], rewards["dist_variable"],
                  FIGURE_DIR / "fig10_reward_convergence.png")
    log(f"[plot] 已输出 6 张图到 {FIGURE_DIR}")


# --------------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="复现 Fig.5-Fig.10（正弦轨迹）")
    parser.add_argument("--stage", default="all",
                        choices=("all", "calibrate", "train", "simulate", "plot"))
    parser.add_argument("--cases", nargs="*", default=list(CASES.keys()),
                        choices=list(CASES.keys()))
    parser.add_argument("--episodes", type=int, default=None, help="覆盖训练回合数（调试用）")
    parser.add_argument("--steps", type=int, default=None, help="覆盖每回合仿真步数（调试用）")
    parser.add_argument("--scale", type=float, default=None, help="直接指定干扰放大系数")
    parser.add_argument("--horizon-mode", default="table5", choices=("table5", "eq31"),
                        help="变步长映射：Table 5 或式(31) N = A*x + 1")
    parser.add_argument("--horizon-update", default=None, choices=("episode", "step"),
                        help="ΔR 口径：episode=相邻回合总回报之差（默认，与 Table 5 量级自洽）；"
                             "step=相邻控制步奖励差")
    parser.add_argument("--qmax", type=float, default=None, help="Q_w 上限")
    parser.add_argument("--seeds", type=int, default=None, help="AC 评估种子数")
    parser.add_argument("--reward-scale", type=float, default=None,
                        help="奖励放大系数（仅影响 RL 奖励与 Fig.10 纵轴，一般保持 1）")
    parser.add_argument("--horizon-scale", type=float, default=None,
                        help="ΔR 的折算系数（默认 100，即把米制奖励差放大到 Table 5 的 20/50 量级）")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    for name in ("episodes", "steps", "q_max", "eval_seeds", "reward_scale"):
        value = getattr(args, {"q_max": "qmax", "eval_seeds": "seeds"}.get(name, name))
        if value is not None:
            cfg = replace(cfg, **{name: value})
    cfg = replace(cfg, horizon_mode=args.horizon_mode)
    if args.horizon_update is not None:
        cfg = replace(cfg, horizon_update=args.horizon_update)
    if args.horizon_scale is not None:
        cfg = replace(cfg, horizon_reward_scale=args.horizon_scale)

    RESULT_DIR.mkdir(exist_ok=True)

    scale = args.scale
    if scale is None and (RESULT_DIR / "disturbance_scale.json").exists():
        scale = json.loads((RESULT_DIR / "disturbance_scale.json").read_text("utf-8"))["disturbance_scale"]
    if scale is None and args.stage in ("all", "calibrate"):
        scale = calibrate_disturbance(cfg)
    if scale is None:
        scale = cfg.disturbance_scale
    cfg = replace(cfg, disturbance_scale=scale)
    print(f"[config] 干扰放大系数 = {scale:.2f}")

    if args.stage in ("all", "train"):
        for case in args.cases:
            train_case(cfg, case, scale)

    if args.stage in ("all", "simulate"):
        simulate(cfg, scale)

    if args.stage in ("all", "plot"):
        rollouts = load_rollouts(cfg)
        cfg = replace(cfg, steps=int(rollouts["mpc_dist"]["err"].size - 1))
        rewards = {}
        for case in ("dist_fixed", "dist_variable"):
            rewards[case] = np.load(agent_path(case))["rewards"]
        plot_all(cfg, rollouts, rewards)


if __name__ == "__main__":
    main()
