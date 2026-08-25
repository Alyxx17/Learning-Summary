import random
import numpy as np
import torch
def set_seed(seed):
    """固定所有随机源:numpy / Python random / torch(CPU+GPU)"""
    np.random.seed(seed)     # 影响 ε-贪婪里的 np.random.rand()
    random.seed(seed)        # 影响经验池的 random.sample()
    torch.manual_seed(seed)  # 影响网络权重初始化（CPU+GPU 一次搞定）
