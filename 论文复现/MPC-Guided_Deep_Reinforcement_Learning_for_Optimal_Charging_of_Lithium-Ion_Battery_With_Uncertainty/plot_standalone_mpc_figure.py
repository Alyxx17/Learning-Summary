# -*- coding: utf-8 -*-
"""复现论文 Fig. 3：单独 MPC 控制器在 50 次未建模误差实现下的仿真结果。

(a) SOC：蓝色为 UKF 估计值，橙色为真实值；
(b) 端电压：虚线为 3.6 V 硬约束，内嵌放大图展示越限细节。

用法：
    D:\\PY\\python.exe plot_standalone_mpc_figure.py --simulations 50
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from battery_parameters import MAXIMUM_VOLTAGE, TARGET_SOC
from mpc_charging_simulation import summarize_standalone_mpc
from result_paths import ensure_directory

ESTIMATED_COLOR = "#4C9BD1"
REAL_COLOR = "#F2A24B"


def plot_results(summary, output_path):
    results = summary["results"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axis_soc, axis_voltage = axes
    for result in results:
        axis_soc.plot(result["time_minutes"], result["soc_estimated"], color=ESTIMATED_COLOR,
                      linewidth=0.9, alpha=0.55)
        axis_soc.plot(result["time_minutes"], result["soc_real"], color=REAL_COLOR,
                      linewidth=0.9, alpha=0.55)
    axis_soc.axhline(TARGET_SOC, color="black", linestyle="--", linewidth=1.2)
    axis_soc.set_xlabel("Time [min]")
    axis_soc.set_ylabel("SOC")
    axis_soc.set_title("(a) State of charge", fontsize=11)
    axis_soc.grid(True, alpha=0.35)
    axis_soc.set_xlim(left=0.0)
    axis_soc.plot([], [], color=ESTIMATED_COLOR, label="Estimated value")
    axis_soc.plot([], [], color=REAL_COLOR, label="Real value")
    axis_soc.legend(loc="lower right", frameon=True)

    for result in results:
        axis_voltage.plot(result["time_minutes"], result["terminal_voltage"],
                          color=REAL_COLOR, linewidth=0.9, alpha=0.55)
    axis_voltage.axhline(MAXIMUM_VOLTAGE, color="black", linestyle="--", linewidth=1.2)
    axis_voltage.set_xlabel("Time [min]")
    axis_voltage.set_ylabel("Voltage [V]")
    axis_voltage.set_title("(b) Terminal voltage", fontsize=11)
    axis_voltage.grid(True, alpha=0.35)
    axis_voltage.set_xlim(left=0.0)

    inset = axis_voltage.inset_axes([0.34, 0.14, 0.55, 0.36])
    for result in results:
        inset.plot(result["time_minutes"], result["terminal_voltage"],
                   color=REAL_COLOR, linewidth=0.8)
    inset.axhline(MAXIMUM_VOLTAGE, color="black", linestyle="--", linewidth=1.0)
    inset.set_ylim(MAXIMUM_VOLTAGE - 0.012, MAXIMUM_VOLTAGE + 0.008)
    inset.set_xlim(2.0, 6.0)
    inset.grid(True, alpha=0.35)
    inset.tick_params(labelsize=7)
    axis_voltage.indicate_inset_zoom(inset, edgecolor="black")

    figure.tight_layout()
    ensure_directory(output_path)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="复现论文 Fig. 3（单独 MPC）")
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/figures/fig3_standalone_mpc.png")
    parser.add_argument("--no-plot", action="store_true", help="只统计不画图")
    args = parser.parse_args()

    summary = summarize_standalone_mpc(simulations=args.simulations, seed=args.seed)
    print(
        f"平均充电时间 {summary['average_charging_time']:.3f} min | "
        f"最大端电压 {summary['maximum_voltage']:.3f} V | "
        f"平均越限率 {summary['average_violation_rate']:.2f} %"
    )
    if not args.no_plot:
        path = plot_results(summary, args.output)
        print(f"已保存: {path}")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "DejaVu Sans"
    main()
