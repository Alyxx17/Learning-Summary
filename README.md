### 寻路算法:  
BFS，A*，A*(带奖励点和障碍点)  

### 系统辨识/无人机参数辨识/1:  
来自于论文《Data-Driven_System_Identification_of_Quadrotors_Subject_to_Motor_Delays》。  
main.py为主要程序，其余为读取飞行日志的辅助函数（必要） 

### 强化学习/表格型经典强化学习:  
包含值迭代，策略迭代，蒙特卡洛（柔性策略与探索起点），n步SARSA/Q学习。
已经加入注释，各代码的相关性很强，部分代码的注释较少，可以参考别的代码，有些代码注释写的比较详细。

### 强化学习/强化学习进阶/DQN:  
学习路径：DQN→Double DQN(DDQN)→Dueling DQN(DuelingDDQN)→nstep DQN(Nstep)→Noisynet DQN→PER DQN。每一种改进都在原有改进上加入，PER DQN为集大成者（5种改进为一体），C51改进暂时没有，每一个代码都分为训练和测试。  
代码注释很详细，有学习笔记关于各部分的原理与解惑，seed为设定随机种子。
