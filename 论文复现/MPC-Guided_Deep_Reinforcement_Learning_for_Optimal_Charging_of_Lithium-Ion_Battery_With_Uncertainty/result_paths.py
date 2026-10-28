# -*- coding: utf-8 -*-
"""结果文件路径约定（训练、评估、绘图共用）。"""

import os


def training_log_path(output_directory, controller, seed):
    return os.path.join(output_directory, "logs", f"{controller}_seed{int(seed)}.npz")


def agent_checkpoint_path(output_directory, controller, seed, episode):
    return os.path.join(
        output_directory, "agents", controller, f"seed{int(seed)}", f"episode_{int(episode):04d}.pt"
    )


def ensure_directory(path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    return path
