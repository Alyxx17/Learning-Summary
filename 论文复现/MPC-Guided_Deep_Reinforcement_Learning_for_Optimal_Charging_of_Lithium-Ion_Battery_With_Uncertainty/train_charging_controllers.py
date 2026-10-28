# -*- coding: utf-8 -*-
"""训练单独 DRL 与 MPC-guided DRL 充电控制器（论文 Section IV-B、Fig. 4、Table III）。

用法示例
--------
快速验证（1 个种子、300 episodes）：
    D:\\PY\\python.exe train_charging_controllers.py --episodes 300 --seeds 1

论文规模（9 个种子、1500 episodes，并在 500/1000/1500 处保存检查点）：
    D:\\PY\\python.exe train_charging_controllers.py --episodes 1500 --seeds 9 --checkpoints 500 1000 1500

训练日志写入 results/logs/<controller>_seed<k>.npz，智能体检查点写入
results/agents/<controller>/seed<k>/episode_XXXX.pt。
"""

import argparse
import os
import time

import numpy as np
import torch

from battery_parameters import MAXIMUM_VOLTAGE
from charging_environments import (
    ACTION_SETTINGS,
    MAXIMUM_STEPS,
    MPCGuidedDRLChargingEnvironment,
    StandaloneDRLChargingEnvironment,
)
from ddpg_agent import DDPGAgent, OrnsteinUhlenbeckNoise
from result_paths import agent_checkpoint_path, ensure_directory, training_log_path

CONTROLLERS = ("drl", "mpc_drl")

# 论文 Table I 的 DDPG 超参数
DDPG_HYPERPARAMETERS = dict(
    actor_hidden_units=(20, 20),
    critic_hidden_units=(100, 75),
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    discount=0.99,
    soft_update_rate=1e-3,
    buffer_capacity=100000,
    batch_size=64,
)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def exploration_decay(episode_index, episodes, minimum=0.1, decay_fraction=0.7):
    """探索噪声随训练衰减。

    论文未给出 OU 噪声的具体尺度。若噪声保持恒定，由于越限惩罚（λV = -100）
    与步惩罚（λS = -0.1）不对称，"在噪声下最优"的策略会系统性地偏保守
    （补偿电流恒为负，白白浪费电压余量）。衰减到 10% 后策略才能收敛到约束边界。
    """
    decay_span = max(1.0, decay_fraction * episodes)
    factor = 1.0 - (1.0 - minimum) * min(episode_index / decay_span, 1.0)
    return max(minimum, factor)


def build_environment(controller, random_generator):
    if controller == "drl":
        return StandaloneDRLChargingEnvironment(random_generator=random_generator)
    return MPCGuidedDRLChargingEnvironment(random_generator=random_generator)


def train_single_run(
    controller,
    seed,
    episodes,
    checkpoint_episodes,
    output_directory,
    verbose=True,
):
    """训练一个 (controller, seed) 组合并保存日志与检查点。"""
    set_random_seed(seed)
    settings = ACTION_SETTINGS[controller]
    random_generator = np.random.RandomState(seed + 20240)
    environment = build_environment(controller, random_generator)
    agent = DDPGAgent(
        state_dimension=settings["state_dimension"],
        action_scale=settings["action_scale"],
        action_bias=settings["action_bias"],
        **DDPG_HYPERPARAMETERS,
    )
    noise = OrnsteinUhlenbeckNoise(
        dimension=1, sigma=0.2, theta=0.15, scale=settings["noise_scale"], seed=seed + 7
    )

    episodes_log = np.arange(1, episodes + 1)
    returns = np.zeros(episodes)
    episode_steps = np.zeros(episodes, dtype=int)
    maximum_temperatures = np.zeros(episodes)
    voltage_margins = np.zeros(episodes)
    charging_times = np.full(episodes, np.nan)

    start_time = time.time()
    for episode_index in range(episodes):
        state = environment.reset()
        noise.reset()
        noise.scale = settings["noise_scale"] * exploration_decay(episode_index, episodes)
        done = False
        steps = 0
        while not done:
            action = agent.select_action(state, noise.sample())
            next_state, reward, done, info = environment.step(float(action[0]))
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            steps += 1
            if steps >= MAXIMUM_STEPS:
                break

        returns[episode_index] = environment.accumulated_reward
        episode_steps[episode_index] = steps
        maximum_temperatures[episode_index] = environment.maximum_temperature
        voltage_margins[episode_index] = environment.maximum_voltage - MAXIMUM_VOLTAGE
        if environment.charging_time is not None:
            charging_times[episode_index] = environment.charging_time
        else:
            # 未在 600 步内充到目标 SOC：按上限 100 min 记录（与原文 Fig. 4(d) 一致）
            charging_times[episode_index] = MAXIMUM_STEPS * environment.parameters.sample_time / 60.0

        current_episode = episode_index + 1
        if current_episode in checkpoint_episodes:
            path = agent_checkpoint_path(output_directory, controller, seed, current_episode)
            agent.save(ensure_directory(path))

        if verbose and (current_episode % 25 == 0 or current_episode == 1):
            elapsed = (time.time() - start_time) / 60.0
            recent = returns[max(0, episode_index - 24): episode_index + 1]
            print(
                f"[{controller}] seed {seed} | episode {current_episode}/{episodes} | "
                f"return {np.mean(recent):8.3f} | T_s,max {np.max(maximum_temperatures[max(0, episode_index - 24): episode_index + 1]):5.2f} | "
                f"t_charge {np.nanmean(charging_times[max(0, episode_index - 24): episode_index + 1]):6.2f} min | "
                f"{elapsed:.1f} min elapsed",
                flush=True,
            )

    final_path = agent_checkpoint_path(output_directory, controller, seed, episodes)
    agent.save(ensure_directory(final_path))

    log_path = ensure_directory(training_log_path(output_directory, controller, seed))
    np.savez(
        log_path,
        episodes=episodes_log,
        returns=returns,
        steps=episode_steps,
        maximum_temperature=maximum_temperatures,
        voltage_margin=voltage_margins,
        charging_time=charging_times,
    )
    return log_path


def main():
    parser = argparse.ArgumentParser(description="训练 DRL / MPC-guided DRL 充电控制器")
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS), choices=list(CONTROLLERS))
    parser.add_argument("--episodes", type=int, default=300, help="训练 episode 数（论文为 1500）")
    parser.add_argument("--seeds", type=int, default=1, help="随机种子个数（论文为 9）")
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[500, 1000, 1500],
                        help="需要保存智能体检查点的 episode（超过 --episodes 的会被忽略）")
    parser.add_argument("--output", default="results", help="结果根目录")
    args = parser.parse_args()

    checkpoint_episodes = tuple(sorted(e for e in args.checkpoints if 0 < e <= args.episodes))

    for controller in args.controllers:
        for seed in range(1, args.seeds + 1):
            print(f"===== {controller} | seed {seed}/{args.seeds} =====", flush=True)
            path = train_single_run(
                controller,
                seed,
                args.episodes,
                checkpoint_episodes,
                args.output,
            )
            print(f"日志已保存: {path}", flush=True)

    print("训练完成。")


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    main()
