# -*- coding: utf-8 -*-
"""复现论文 Fig. 4：单独 DRL 与 MPC-guided DRL 的训练过程对比。

(a) 学习曲线（episode 回报）
(b) 表面温度
(c) 电压越限量（最大端电压 - V_max）
(d) 平均充电时间

绘图约定（与原文一致的外观）：实线 = 均值，浅色阴影 = 上下界。
原文的均值/上下界是对 9 个随机种子取的（上界为 9 次实验的最大值、下界为最小值）。
若只有 1 个种子，则退化为"滑动窗口内的均值与最小/最大值"，用于呈现同样的
"实线 + 波动带"视觉效果；窗口宽度由 --window 指定（默认 25 个 episode）。
有多个种子时自动改用种子间的最大/最小值，与原文口径一致。

用法：
    D:\\PY\\python.exe plot_training_comparison.py --input results
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from result_paths import ensure_directory

CONTROLLER_STYLE = {
    "drl": {"label": "DRL", "color": "#3F8FC4"},
    "mpc_drl": {"label": "MPC-guided DRL", "color": "#F2A24B"},
}
METRIC_SETTINGS = [
    ("returns", "(a) Learning curve", "Return [-]", 0.0),
    ("maximum_temperature", "(b) Surface temperature", "Surface Temp. [\u00b0C]", 32.0),
    ("voltage_margin", "(c) Voltage violation", "Voltage Violation [V]", 0.0),
    ("charging_time", "(d) Charging time", "Time [min]", 0.0),
]


def load_logs(input_directory, controller):
    paths = sorted(glob.glob(os.path.join(input_directory, "logs", f"{controller}_seed*.npz")))
    return [np.load(path) for path in paths]


def window_statistics(values, window):
    """滑动窗口的均值与上下界；窗口宽度为 1 时即原始序列。"""
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < window:
        return values.copy(), values.copy(), values.copy()
    half = window // 2
    padded = np.pad(values, (half, window - 1 - half), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    with np.errstate(all="ignore"):
        return np.nanmean(view, axis=1), np.nanmin(view, axis=1), np.nanmax(view, axis=1)


def aggregate(logs, key, window):
    """汇总多个种子：均值取各种子均值之平均，上下界取所有种子的极值。"""
    if not logs:
        return None, None, None, None
    means, lows, highs = [], [], []
    episodes = None
    for log in logs:
        series = log[key].astype(float)
        series[~np.isfinite(series)] = np.nan
        mean, low, high = window_statistics(series, window)
        means.append(mean)
        lows.append(low)
        highs.append(high)
        episodes = log["episodes"]
    with np.errstate(all="ignore"):
        mean = np.nanmean(np.vstack(means), axis=0)
        low = np.nanmin(np.vstack(lows), axis=0)
        high = np.nanmax(np.vstack(highs), axis=0)
    return episodes, mean, low, high


def add_zoom_inset(axis, series_by_controller, bounds, position):
    """在坐标轴内加放大图（原文 Fig. 4(a)(c) 的做法），同时画两条曲线。"""
    inset = axis.inset_axes(position)
    for controller, (style, episodes, mean, low, high) in series_by_controller.items():
        inset.fill_between(episodes, low, high, color=style["color"], alpha=0.28, linewidth=0)
        inset.plot(episodes, mean, linewidth=1.2, color=style["color"])
    inset.set_xlim(*bounds)
    lows, highs = [], []
    for _, (_, episodes, _, low, high) in series_by_controller.items():
        window = (episodes >= bounds[0]) & (episodes <= bounds[1])
        if np.any(window):
            lows.append(np.nanmin(low[window]))
            highs.append(np.nanmax(high[window]))
    if lows:
        data_low, data_high = min(lows), max(highs)
        span = max(data_high - data_low, 1e-3)
        inset.set_ylim(data_low - 0.12 * span, data_high + 0.12 * span)
    inset.grid(True, alpha=0.35)
    inset.tick_params(labelsize=8, length=2)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)


def trim_episodes(episodes, values, limit):
    mask = episodes <= limit
    return episodes[mask], values[..., mask]


def main():
    parser = argparse.ArgumentParser(description="复现论文 Fig. 4")
    parser.add_argument("--input", default="results", help="训练结果根目录")
    parser.add_argument("--output", default="results/figures/fig4_training_comparison.png")
    parser.add_argument("--window", type=int, default=25,
                        help="波动带的滑动窗口宽度（单种子时生效，默认 25）")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="只画到指定 episode（默认取训练长度）")
    args = parser.parse_args()

    available = {}
    for controller in CONTROLLER_STYLE:
        logs = load_logs(args.input, controller)
        if logs:
            available[controller] = logs
            print(f"{controller}: 读取 {len(logs)} 个随机种子的训练日志")
        else:
            print(f"{controller}: 未找到训练日志")

    if not available:
        print("没有可用的训练日志，请先运行 train_charging_controllers.py。")
        return

    if max(len(logs) for logs in available.values()) == 1:
        print(f"仅 1 个种子：阴影为滑动窗口（宽度 {args.window}）内的最小/最大值。")

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    axes = axes.ravel()

    for axis, (key, title, ylabel, reference) in zip(axes, METRIC_SETTINGS):
        for controller, logs in available.items():
            style = CONTROLLER_STYLE[controller]
            episodes, mean, low, high = aggregate(logs, key, args.window)
            if episodes is None:
                continue
            limit = args.max_episodes or episodes.max()
            episodes, series = trim_episodes(episodes, np.vstack([mean, low, high]), limit)
            mean, low, high = series
            axis.plot(episodes, mean, color=style["color"], linewidth=1.4, label=style["label"])
            axis.fill_between(episodes, low, high, color=style["color"], alpha=0.28, linewidth=0)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Episode Number")
        axis.set_ylabel(ylabel)
        axis.axhline(reference, color="black", linestyle="-.", linewidth=1.0)
        axis.grid(True, alpha=0.35)
        axis.set_xlim(left=0.0)
    axes[0].legend(loc="lower left", frameon=True, fontsize=9)

    # 原文在 (a) 与 (c) 中给出收敛段的放大图
    for axis, key, position in (
        (axes[0], "returns", [0.42, 0.20, 0.50, 0.34]),
        (axes[2], "voltage_margin", [0.42, 0.26, 0.50, 0.32]),
    ):
        series_by_controller = {}
        for controller, logs in available.items():
            episodes, mean, low, high = aggregate(logs, key, args.window)
            if episodes is None:
                continue
            series_by_controller[controller] = (
                CONTROLLER_STYLE[controller], episodes, mean, low, high
            )
        if not series_by_controller:
            continue
        upper = max(float(item[1].max()) for item in series_by_controller.values())
        if args.max_episodes:
            upper = min(upper, float(args.max_episodes))
        add_zoom_inset(axis, series_by_controller, (upper * 0.33, upper), position)

    figure.tight_layout()
    ensure_directory(args.output)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(os.path.splitext(args.output)[0] + ".pdf", bbox_inches="tight")
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "DejaVu Sans"
    main()
