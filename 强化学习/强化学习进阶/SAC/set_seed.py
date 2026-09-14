import random
import numpy as np
import torch
def set_seed(seed):
    """固定所有随机源:numpy / Python random / torch(CPU+GPU)"""
    np.random.seed(seed)     # 影响环境初始状态、经验池采样等随机数
    random.seed(seed)        # 影响其它 Python 随机
    torch.manual_seed(seed)  # 影响网络权重初始化（CPU+GPU 一次搞定）
