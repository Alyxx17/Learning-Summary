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
# MPC相关自学内容  
[SACMPC](https://github.com/Alyxx17/Learning-Summary/tree/main/SACMPC)
├── unicycle_env.py          # 底层物理环境与任务定义
├── tracking_mpc.py          # CasADi 实现的参数化跟踪 MPC 求解器
├── mpc_env.py               # Gym 风格环境封装（MPC 与真实系统的交互接口）
├── sac_agent.py             # SAC 智能体（Actor-Critic 网络、经验池、自动温度）
├── train_sac_qr.py          # SAC 训练脚本（在线学习 Q/R 权重）
├── evaluate_compare.py      # 评估与对比脚本（SAC vs 固定权重基线）
├── set_seed.py              # 随机种子固定工具
├── sac_diag_model_error.pth # [生成] 训练得到的最终模型权重(MISMATCH_PRESET = "mild"下训练出来的模型，下同)
└──  sac_diag_model_best_error.pth # [生成] 训练中评估最优的模型权重  

> 仅作为SAC与MPC结合的学习代码，学习强化学习算法如何与MPC结合。
> 运行训练请执行 train_sac_qr.py；运行评估对比请执行 python evaluate_compare.py。

- unicycle_env.py：定义了独轮车（Unicycle）的运动学模型、参考轨迹生成器（直线/圆弧/正弦）、跟踪误差计算、单步评价代价函数。特别地，这里包含了模型失配（Mismatch）的开关与参数（执行器增益、侧滑角、外扰），以及动作空间到物理权重的映射（decode_action）。  
- tracking_mpc.py：基于 CasADi + IPOPT 的跟踪 MPC 求解器。它接收当前状态、参考窗口以及 SAC 输出的 5 个 Q/R 对角权重，在预测时域内求解最优控制序列，并返回首步控制量。MPC 内部模型使用理想模型，与真实系统形成失配。
- mpc_env.py：Gym 风格的环境封装类 MPCEnv。它负责协调 unicycle_env 和 tracking_mpc，将 SAC 的归一化动作解码为权重，调用 MPC 求解，并将控制量施加到带有失配的真实系统上，最后返回奖励和下一状态。
- sac_agent.py：实现了 Soft Actor-Critic (SAC) 算法。包含双 Q 网络、目标网络软更新、重参数化 tanh 策略以及自动温度调节。它不直接与环境交互，而是提供策略采样和网络更新的接口。
- train_sac_qr.py：训练主脚本。初始化 MPCEnv 和 SACAgent，进行在线交互。SAC 每步输出权重，MPC 求解控制，真实系统推进。脚本会记录训练回报、Q/R 权重的动态变化，并定期在固定评估集上验证，保存最优模型。
- evaluate_compare.py：对比实验脚本。加载训练好的 SAC 模型，在 30 个固定评测场景上运行，并与“固定权重（单位阵）”基线进行对比。输出包括平均等效代价、RMSE、尾段误差、平均权重，并生成对比图表。
- 


# 强化学习基础 
[表格型经典强化学习](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E8%A1%A8%E6%A0%BC%E5%9E%8B%E7%BB%8F%E5%85%B8%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)  
- 包含值迭代，策略迭代，蒙特卡洛（柔性策略与探索起点），n步SARSA/Q学习。  
-----
[DQN](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/DQN)
- 学习路径：DQN——Double DQN(DDQN)——Dueling DQN(DuelingDDQN)——nstep DQN(Nstep)——Noisynet DQN——PER DQN。
- 每一种改进都在原有改进上加入，PER DQN为集大成者（5种改进为一体），C51改进暂时没有
-----
[策略梯度](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6)
- 学习路径：REINFORCE——REINFORCE_baseline(基线 + 熵正则 + 回报标准化)——A2C(GAE 优势 + 并行环境 + 共享网络 + 熵正则 + 优势标准化)——PPO(相对A2C新增了clip与重要性minibatch重训练)。
> 对于DQN与策略梯度，每个算法都分为训练和测试，目录内已有对应训练好的模型，若需要重新训练，先运行xxxx_train.py，再运行对应的xxxx_test.py  
> 代码注释很详细，有学习笔记关于各部分的原理与解惑，seed为设定随机种子
-----
[SAC](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/SAC)
> 自动调节温度系数


# 寻路算法 
BFS，A*，A*(带奖励点和障碍点)  
> 每个代码单独运行即可，主要用于学习原理算法



