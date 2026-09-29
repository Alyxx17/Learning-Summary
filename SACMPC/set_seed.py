# =================== 随机种子工具 ===================
# 与 RL_py/SAC 下用法保持一致：np / random / torch 一次性固定
# （本项目仅在用到随机性（轨迹采样、网络初始化）的脚本中使用）

import random
import numpy as np
import torch

def set_seed(seed):
    """固定所有随机源: numpy / Python random / torch(CPU+GPU)"""
    np.random.seed(seed)     # 影响全局 np.random 调用
    random.seed(seed)        # 影响 Python random
    torch.manual_seed(seed)  # 影响网络权重初始化（CPU+GPU 一次搞定）
