# 寻路算法:  
BFS，A*，A*(带奖励点和障碍点)  
### 使用与说明
> 每个代码单独运行即可，主要用于学习原理算法

# 论文复现/系统辨识/1  
来自于论文《Data-Driven_System_Identification_of_Quadrotors_Subject_to_Motor_Delays》。  
### 使用与说明
> 运行main.py即可  
> 论文原代码仓库：https://github.com/arplaboratory/data-driven-system-identification

# 强化学习/表格型经典强化学习:  
包含值迭代，策略迭代，蒙特卡洛（柔性策略与探索起点），n步SARSA/Q学习。
已经加入注释，各代码的相关性很强，部分代码的注释较少，可以参考别的代码，有些代码注释写的比较详细。
### 使用与说明
> 每个代码单独运行即可，主要用于学习原理算法

# 强化学习/强化学习进阶/DQN:  
学习路径：DQN→Double DQN(DDQN)→Dueling DQN(DuelingDDQN)→nstep DQN(Nstep)→Noisynet DQN→PER DQN。每一种改进都在原有改进上加入，PER DQN为集大成者（5种改进为一体），C51改进暂时没有，每一个代码都分为训练和测试。  
代码注释很详细，有学习笔记关于各部分的原理与解惑，seed为设定随机种子。
### 使用与说明
> 每个算法均有对应训练好的模型  
> 若需要重新训练，先运行xxxx_train.py，再运行对应的xxxx_test.py

# 强化学习/强化学习进阶/策略梯度:  
学习路径：REINFORCE→REINFORCE-baseline(基线 + 熵正则 + 回报标准化)→A2C(GAE 优势 + 并行环境 + 共享网络 + 熵正则 + 优势标准化)→PPO(相对A2C新增了clip与重要性minibatch重训练)。每一个代码都分为训练和测试。
代码注释很详细，有学习笔记关于各部分的原理与解惑，seed为设定随机种子。
### 使用与说明
> 每个算法均有对应训练好的模型  
> 若需要重新训练，先运行xxxx_train.py，再运行对应的xxxx_test.py

