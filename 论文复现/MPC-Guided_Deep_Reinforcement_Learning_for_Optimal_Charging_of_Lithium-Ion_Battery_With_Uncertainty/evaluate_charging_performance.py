# -*- coding: utf-8 -*-
"""生成论文 Table III：不同训练 episode 下两种控制器的充电性能对比。

评估方式与论文一致：训练完成后关闭探索噪声（确定性策略），
用表现最好的智能体在真实系统上做 50 次随机仿真，统计
    最大端电压、平均电压越限率、平均充电时间。

用法：
    D:\\PY\\python.exe evaluate_charging_performance.py --episodes 500 1000 1500 --simulations 50
"""

import argparse
import csv
import os

import numpy as np
import torch

from battery_parameters import MAXIMUM_VOLTAGE
from charging_environments import (
    ACTION_SETTINGS,
    MAXIMUM_STEPS,
    MPCGuidedDRLChargingEnvironment,
    StandaloneDRLChargingEnvironment,
)
from ddpg_agent import DDPGAgent
from result_paths import agent_checkpoint_path, ensure_directory


def build_environment(controller, random_generator):
    if controller == "drl":
        return StandaloneDRLChargingEnvironment(random_generator=random_generator)
    return MPCGuidedDRLChargingEnvironment(random_generator=random_generator)


def build_agent(controller):
    settings = ACTION_SETTINGS[controller]
    return DDPGAgent(
        state_dimension=settings["state_dimension"],
        action_scale=settings["action_scale"],
        action_bias=settings["action_bias"],
    )


def evaluate_agent(controller, agent, simulations=50, seed=12345, maximum_steps=MAXIMUM_STEPS):
    """50 次随机仿真，返回 (最大电压, 平均越限率%, 平均充电时间 min)。"""
    charging_times = []
    violation_rates = []
    maximum_voltages = []

    for index in range(simulations):
        environment = build_environment(controller, np.random.RandomState(seed + index))
        state = environment.reset()
        done = False
        violating_steps = 0
        steps = 0
        while not done:
            action = agent.select_action(state, noise=None)
            state, _, done, info = environment.step(float(action[0]))
            steps += 1
            if info["terminal_voltage"] > MAXIMUM_VOLTAGE:
                violating_steps += 1
            if steps >= maximum_steps:
                break

        charging_time = (
            environment.charging_time
            if environment.charging_time is not None
            else maximum_steps * environment.parameters.sample_time / 60.0
        )
        charging_times.append(charging_time)
        violation_rates.append(100.0 * violating_steps / max(steps, 1))
        maximum_voltages.append(environment.maximum_voltage)

    return (
        float(np.max(maximum_voltages)),
        float(np.mean(violation_rates)),
        float(np.mean(charging_times)),
    )


def main():
    parser = argparse.ArgumentParser(description="生成 Table III 的性能对比")
    parser.add_argument("--controllers", nargs="+", default=["drl", "mpc_drl"],
                        choices=["drl", "mpc_drl"])
    parser.add_argument("--episodes", nargs="+", type=int, default=[500, 1000, 1500],
                        help="要评估的智能体训练 episode（对应检查点）")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="要评估的随机种子（论文取表现最好的智能体，可用 1 个）")
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--input", default="results", help="训练结果根目录")
    parser.add_argument("--output", default="results/table3_charging_performance.csv")
    args = parser.parse_args()

    rows = []
    for controller in args.controllers:
        for episode in args.episodes:
            for seed in args.seeds:
                path = agent_checkpoint_path(args.input, controller, seed, episode)
                if not os.path.exists(path):
                    print(f"[跳过] 未找到检查点 {path}")
                    continue
                agent = build_agent(controller)
                agent.load_actor(path)
                maximum_voltage, violation_rate, charging_time = evaluate_agent(
                    controller, agent, simulations=args.simulations
                )
                rows.append(
                    {
                        "controller": controller,
                        "episodes": episode,
                        "seed": seed,
                        "maximum_voltage_V": maximum_voltage,
                        "violation_rate_percent": violation_rate,
                        "charging_time_min": charging_time,
                    }
                )
                print(
                    f"{controller:8s} | {episode:5d} ep | seed {seed} | "
                    f"U_max {maximum_voltage:.3f} V | 越限率 {violation_rate:6.2f} % | "
                    f"充电时间 {charging_time:6.3f} min",
                    flush=True,
                )

    if not rows:
        print("没有可评估的检查点，请先运行 train_charging_controllers.py。")
        return

    ensure_directory(args.output)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存: {args.output}")

    # 按论文 Table III 的布局打印
    headers = ["Controller"] + [f"{episodes} episodes" for episodes in args.episodes]
    print("\n| 指标 | " + " | ".join(headers[1:]) + " |")
    for controller in args.controllers:
        for metric in ("maximum_voltage_V", "violation_rate_percent", "charging_time_min"):
            values = []
            for episode in args.episodes:
                matches = [
                    row for row in rows
                    if row["controller"] == controller and row["episodes"] == episode
                ]
                values.append(f"{np.mean([m[metric] for m in matches]):.3f}" if matches else "-")
            print(f"| {controller} {metric} | " + " | ".join(values) + " |")


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    main()
