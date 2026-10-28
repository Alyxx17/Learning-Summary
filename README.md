# 声明
内容均有AI辅助  
 
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
-----
- 《*Value Approximator-Based Learning Model Predictive Control for Iterative Tasks*》
[复现代码](https://github.com/Alyxx17/Learning-Summary/tree/main/%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/VA-LMPC)

├── A. Numerical Example/                                  # 复现 1：IV-A 数值例子（双积分器）  
├── double_integrator_env.py                           # 底层物理环境与任务定义（动力学 / 代价 Q,R / 状态与输入约束 / 初始可行轨迹 / cost-to-go 标签）  
├── va_basis.py                                        # 价值近似器：六项基函数的数值版与符号版、最小二乘拟合、凸性检查工具  
├── va_lmpc_solver.py                                  #  CasADi+IPOPT 求解器（终端代价 = 学到的价值函数）与闭环滚动  
├── va_lmpc_train.py                                   # Algorithm 1 训练主程序：初始化 W⁰ → 逐轮"求解/执行/更新 W" → 保存训练结果  
├── va_lmpc_test.py                                    # 结果评估与出图：复算闭环、求解 OLPS 基准、输出 3 张图  
├── olps.py                                            # 开环最优参考解（N=100 直接法优化，用于验证最优性）  
├── W_final.npy                                        # 训练产物：最终价值权重 W*  
├── va_data.npz                                        # 训练产物：迭代代价、耗时、W 历史、初始可行轨迹数据  
├── fig_value_approximator.png                         # 输出图：价值近似器曲面  
├── fig_trajectory.png                                 # 输出图：x-y 相平面轨迹对比   
└── fig_iteration_cost.png                             # 输出图：迭代代价收敛曲线  

├── B. Vehicle Dynamics and Fixed Obstacle Avoidance/      # 复现 2：IV-B 车辆避障（固定椭圆障碍）  
├── vehicle_env.py                                     # 底层物理环境与任务定义（车辆模型 / 椭圆障碍 / 代价 / 初始可行轨迹与 OLPS 共用的 OCP）  
├── va_basis.py                                        # 价值近似器（B 版）：六项基函数 + 普通最小二乘 + 带结构约束的最小二乘（W0=0、二次部分半正定、W2≤0）  
├── vehicle_va_mpc.py                                  # 车辆版求解器与闭环滚动（每轮终止判据 J* ≤ 1e-4）  
├── vehicle_train.py                                   # 训练主程序（车辆版）  
├── vehicle_test.py                                    # 结果评估与出图：价值曲面 / 输入对比 / 轨迹  
├── vehicle_W_final.npy                                # 训练产物：最终价值权重 W*  
├── vehicle_data.npz                                   # 训练产物：迭代代价、耗时、W 历史、初始可行轨迹数据  
├── fig_vehicle_value.png                              # 输出图：价值近似器曲面  
├── fig_vehicle_inputs.png                             # 输出图：第 1 轮速度、偏航角输入 vs OLPS  
└── fig_vehicle_traj.png                               # 输出图：轨迹 + 椭圆障碍  
> 两个仿真均先运行train再运行test即可  
> 参考代码：https://github.com/Tongji-BitaAI-Lab/VA_LMPC
-----
- 《*Adaptive trajectory tracking of the unmanned surface vessel based on improved AC-MPC method*》
[复现代码](https://github.com/Alyxx17/Learning-Summary/tree/main/%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/Adaptive%20trajectory%20tracking%20of%20the%20unmanned%20surface%20vessel%20based%20on)

├── Adaptive trajectory tracking of the unmanned surface vessel based on/   # 复现 2：论文场景 1（正弦轨迹）的 AC-MPC 跟踪  
├── cs2_model.py                    # 被控对象：CyberShip II 三自由度模型（式2-6）、风/流干扰（式4-5）、RK4、解析 Jacobian、误差定义  
├── sinusoid_reference.py           # 参考轨迹：x_d=vx·t, y_d=10 sin(2πx_d/12)（式33 的图一致版），位姿/速度采样  
├── ltv_mpc_tracker.py              # 式(7)-(15) 的 LTV-MPC：非线性自由响应 + ZOH 输入响应、Q/R 代价、推力/速率约束、裁剪/可选 QP  
├── dual_qp_solver.py               # 输入/速率约束的对偶 QP（L-BFGS-B + 热启动），use_qp=True 时才启用  
├── actor_critic_tuner.py           # 式(16)-(32) 的 AC：264-128 ReLU、Adam、3 个独立 softmax 头、累积回报更新、Table 5 的 N 选择  
├── sinusoid_acmpc_experiment.py    # 主程序：ExperimentConfig、calibrate/train/simulate/plot 四阶段、5 组仿真、Fig.5-10  
├── 复现笔记_AC-MPC正弦轨迹.md       # 复现记录：参数假设、标定、与论文不一致处、结果表  
├── results/                        # 运行产物（删掉后可重跑）  
│   ├── disturbance_scale.json      # 标定产物：干扰放大系数（≈93.66）与 MPC 稳态峰值（0.450 m）  
│   ├── agent_clean_fixed.npz       # 训练产物：无干扰/固定 N 的 AC 权重 + rewards/q_hist/n_hist/final_horizon  
│   ├── agent_dist_fixed.npz        # 训练产物：有干扰/固定 N  
│   ├── agent_dist_variable.npz     # 训练产物：有干扰/变 N  
│   ├── rollouts.npz                # 仿真产物：5 条曲线的 pos/psi/err/err_std、Q/N 历史（AC 为 5 seeds 均值）  
│   └── summary.md                  # 结果汇总：稳态峰值均值±标准差、评估种子数、q_max、ΔR 口径  
└── figures/                        # 绘图产物（--stage plot 生成）  
    ├── fig05_sinusoid_trajectory.png            # MPC vs AC-MPC 轨迹对比（固定步长）  
    ├── fig06_sinusoid_error.png                 # MPC vs AC-MPC 误差曲线（固定步长）  
    ├── fig07_sinusoid_q_value.png               # 无/有干扰下 Q(1,1)、Q(2,2)、Q(3,3) 变化  
    ├── fig08_sinusoid_trajectory_variable_step.png  # 固定 vs 变步长 AC-MPC 轨迹对比  
    ├── fig09_sinusoid_error_variable_step.png   # 固定 vs 变步长误差曲线  
    └── fig10_reward_convergence.png             # 固定/变步长训练奖励收敛  
> 只复现了scenario 1  
> 论文本身存在一些问题，参数也没给全  
> 没有与原文核对，仅与AI促膝长谈后，AI给出的代码
-----
- 《*MPC-Guided Deep Reinforcement Learning for  Optimal Charging of Lithium-Ion Battery  With Uncertainty*》
[复现代码](https://github.com/Alyxx17/Learning-Summary/tree/main/%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/MPC-Guided_Deep_Reinforcement_Learning_for_Optimal_Charging_of_Lithium-Ion_Battery_With_Uncertainty)

├── 复现说明.md                        # 复现范围、参数来源与标定依据（Lin 2014 + 反标定）、与原文不一致处、实测结果对照、已修复的实现缺陷  
├── battery_parameters.py              # 式(1)–(7) 参数：OCV(SOC) 单调三次插值、R0/R1/R2/C1/C2 的 SOC/温度依赖、约束阈值（Imax=46 A、Vmax=3.6 V、Ts,max=32 °C）、未建模 SOC 扰动幅值 0.001  
├── electro_thermal_model.py           # 式(1)–(9)(13)：五状态 [U1,U2,SOC,Ts,Tc] 电热模型；RC 网络 ZOH 精确离散、热网络矩阵指数离散、解析 Jacobian 与线性化偏移 δ  
├── unscented_kalman_filter.py         # Algorithm 1：UKF 观测器（状态 [U1,U2,SOC]，量测 Ut），P0=diag(1e-7,1e-7,2.45)、Qu、Ru 取原文值，SOC 截断防失锁  
├── linear_mpc.py                      # 式(13)–(15)：逐点线性化 + 预测域内 LTI 的 MPC；Δu 按 Imax 无量纲化；u/Ts/Ut 三类约束；自带原始有效集法 QP（SLSQP 兜底）  
├── ddpg_agent.py                      # 式(16)–(29)：DDPG（Actor 20-20 / Critic 100-75 / ReLU / 软更新 τ=1e-3）、经验回放、OU 探索噪声  
├── charging_environments.py           # 式(21)(22)(30)(31)：单独 DRL（3 维观测，u∈[0,46] A）与 MPC-guided DRL（7 维观测，u^l∈[-6,6] A）两个环境；奖励 λV/λT/λS；状态归一化  
├── mpc_charging_simulation.py         # 单独 MPC 充电仿真（Fig.3 数据源）：UKF 修正 SOC、真实系统叠加 p=0.001·sin(X)  
├── train_charging_controllers.py      # 训练入口（Fig.4 / Table III 数据源）：Table I 超参、探索噪声线性衰减、300/500/1000/1500 检查点  
├── evaluate_charging_performance.py   # Table III 评估：贪心策略 × 50 次随机仿真，统计最大端电压、平均越限率、平均充电时间  
├── plot_standalone_mpc_figure.py      # Fig.3 绘图：50 次实现的 SOC（估计/真实）与端电压，含 3.6 V 越限放大图  
├── plot_training_comparison.py        # Fig.4 绘图：(a) 回报 (b) 表面温度 (c) 电压越限 (d) 充电时间；单种子时用滑动窗口极值替代 9 种子包络  
├── result_paths.py                    # 结果路径约定：logs/<controller>_seed<k>.npz、agents/<controller>/seed<k>/episode_XXXX.pt  
└── results/                           # 1 个种子 × 1500 episodes 的实测输出  
    ├── agents/                        # 智能体检查点（actor/critic + 目标网络 + 优化器状态）  
    │   ├── drl/seed1/episode_{0300,0500,1000,1500}.pt  
    │   └── mpc_drl/seed1/episode_{0300,0500,1000,1500}.pt  
    ├── logs/                          # 逐 episode 的回报、步数、最高表面温度、电压裕度、充电时间  
    │   ├── drl_seed1.npz  
    │   └── mpc_drl_seed1.npz  
    ├── figures/                       # fig3_standalone_mpc.png、fig4_training_comparison.png  
    └── table3_charging_performance.csv # Table III 数值（6 个智能体 × 最大电压 / 越限率 / 充电时间）  
    
> 仅复现fig3,4,TABLEⅢ  
> 论文有未给的参数

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
> 运行训练请执行 train_sac_qr.py；
> 运行评估对比请执行 python evaluate_compare.py。

- unicycle_env.py：定义了独轮车（Unicycle）的运动学模型、参考轨迹生成器（直线/圆弧/正弦）、跟踪误差计算、单步评价代价函数。特别地，这里包含了模型失配（Mismatch）的开关与参数（执行器增益、侧滑角、外扰），以及动作空间到物理权重的映射（decode_action）。  
- tracking_mpc.py：基于 CasADi + IPOPT 的跟踪 MPC 求解器。它接收当前状态、参考窗口以及 SAC 输出的 5 个 Q/R 对角权重，在预测时域内求解最优控制序列，并返回首步控制量。MPC 内部模型使用理想模型，与真实系统形成失配。
- mpc_env.py：Gym 风格的环境封装类 MPCEnv。它负责协调 unicycle_env 和 tracking_mpc，将 SAC 的归一化动作解码为权重，调用 MPC 求解，并将控制量施加到带有失配的真实系统上，最后返回奖励和下一状态。
- sac_agent.py：实现了 Soft Actor-Critic (SAC) 算法。包含双 Q 网络、目标网络软更新、重参数化 tanh 策略以及自动温度调节。它不直接与环境交互，而是提供策略采样和网络更新的接口。
- train_sac_qr.py：训练主脚本。初始化 MPCEnv 和 SACAgent，进行在线交互。SAC 每步输出权重，MPC 求解控制，真实系统推进。脚本会记录训练回报、Q/R 权重的动态变化，并定期在固定评估集上验证，保存最优模型。
- evaluate_compare.py：对比实验脚本。加载训练好的 SAC 模型，在 30 个固定评测场景上运行，并与“固定权重（单位阵）”基线进行对比。输出包括平均等效代价、RMSE、尾段误差、平均权重，并生成对比图表。
  


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
-----
[TD3](https://github.com/Alyxx17/Learning-Summary/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%9B%E9%98%B6/TD3)
> DPG——DDPG——TD3  

# 寻路算法 
BFS，A*，A*(带奖励点和障碍点)  
> 每个代码单独运行即可，主要用于学习原理算法



