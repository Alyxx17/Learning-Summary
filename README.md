# 论文复现
- 《*Data Driven System Identification of Quadrotors Subject to Motor Delays*》
[复现代码](https://github.com/Alyxx17/Learning-Summary/tree/main/%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E7%B3%BB%E7%BB%9F%E8%BE%A8%E8%AF%86/1)
> 运行main.py即可  
> 论文原代码仓库：https://github.com/arplaboratory/data-driven-system-identification
-----
- 《*Reinforcement Learning Based Model Predictive Control for Discrete Time Systems*》
[复现代码](https://github.com/Alyxx17/Learning-Summary/tree/main/%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/RLMPC/1)
>  unicycle_env.py：环境动力学  
> poly_basis.py：基函数生成  
> mpc_solver.py:MPC求解器  
> rlmpc_train.py RLMPC训练  
> rlmpc_test.py 对比  
> 使用时先运行rlmpc_train.py再运行 rlmpc_test.py  
> 只复现了非线性，对于线性，按照论文的参数可以轻松复现  
> 代码内注释很详细，在此不赘述  
> 目录内有论文的方法原理，以及终端集，终端约束的基础原理

# 强化学习基础 
[表格型经典强化学习](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E8%A1%A8%E6%A0%BC%E5%9E%8B%E7%BB%8F%E5%85%B8%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)  
- 包含值迭代，策略迭代，蒙特卡洛（柔性策略与探索起点），n步SARSA/Q学习。  
-----
[DQN](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/DQN)
- 学习路径：DQN——Double DQN(DDQN)——Dueling DQN(DuelingDDQN)——nstep DQN(Nstep)——Noisynet DQN——PER DQN。
- 每一种改进都在原有改进上加入，PER DQN为集大成者（5种改进为一体），C51改进暂时没有
-----
[策略梯度](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6)
- 学习路径：REINFORCE——REINFORCE——baseline(基线 + 熵正则 + 回报标准化)——A2C(GAE 优势 + 并行环境 + 共享网络 + 熵正则 + 优势标准化)——PPO(相对A2C新增了clip与重要性minibatch重训练)。
> 对于DQN与策略梯度，每个算法都分为训练和测试，目录内已有对应训练好的模型，若需要重新训练，先运行xxxx_train.py，再运行对应的xxxx_test.py  
> 代码注释很详细，有学习笔记关于各部分的原理与解惑，seed为设定随机种子


# 寻路算法 
BFS，A*，A*(带奖励点和障碍点)  
> 每个代码单独运行即可，主要用于学习原理算法



