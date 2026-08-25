import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =================== Configure ===================
number_of_column = 6
number_of_row = 5

target_state = (3, 2)                     
obstacle_state = [(1, 2), (2, 2), (2, 4), (3, 1), (3, 3)]  

state_space = number_of_column * number_of_row
reward_target = 15
reward_obstacle = -5
reward_bound = -3
reward_step = -0.1 
gamma = 0.9        

actions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
number_of_actions = len(actions)

def next_state_and_reward(state, action_idx):
    a_row, a_column = actions[action_idx]
    new_row = state[0] + a_row
    new_column = state[1] + a_column

    # boundary check
    if not (0 <= new_row < number_of_row and 0 <= new_column < number_of_column):
        return state, reward_bound, False

    new_state = (new_row, new_column)

    # obstacle check
    if new_state in obstacle_state:
        return new_state, reward_obstacle, True

    # target check
    if new_state == target_state:
        return new_state, reward_target, True

    return new_state, reward_step, False


# =================== n-step SARSA/Q-learning===================
type = 'Q'#SARSA or Q
num_episodes = 100000        # total episodes
max_steps_per_episode = 100
epsilon = 0.5              # initial exploration rate
epsilon_end = 0.1           # minimum exploration rate
start_state = (2,3)        # fixed start state
n_step = 2
Q = np.zeros((number_of_row, number_of_column, number_of_actions))
N = np.zeros((number_of_row, number_of_column, number_of_actions))  # visit counter for incremental average

# ε‑greedy action selection
def select_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(number_of_actions)
    else:
        return np.argmax(Q[state[0], state[1], :])

print(f"{type} starts training, ε = {epsilon}, episodes = {num_episodes}")
#MC中需要走到终点或者障碍物才算一回合，方差大，但是无偏。
#那我可不可以没病走几步，把这几步用于计算G呢，这就是SARSA(n_step)，有偏，但方差小，当n无穷时即退化为MC
#本质上是时序差分，即把走到终点/障碍物的真实回报G改为
# 走n步的真实回报+gamma^(n)*最后一个状态以及对应的采取的某个动作的估计Q(s_last，a_last)
#SARSA：当前状态，在当前状态采取的动作，在当前状态采取的动作后得到的立即奖励，在当前状态采取的动作后的状态，这个状态采取的动作
for ep in range(num_episodes):
    state = start_state
    action = select_action(state, Q, epsilon)
    step = 0
    done = False
    buffer = []

    while not done and step < max_steps_per_episode:
        next_s, reward, done = next_state_and_reward(state, action)

        if not done:
            next_action = select_action(next_s, Q, epsilon)
        else:
            next_action = None

        buffer.append((state, action, reward, next_s, next_action, done))

        # ---------- 轨迹遇到终点或障碍物结束 ----------
        #进行MC式接断，即根据MC一样计算每个状态动作对的Q，不需要加上估计Q
        if done:
            
            for idx in range(len(buffer)):
                s, a, _, _, _, _ = buffer[idx]
                G = 0.0
                for j in range(idx, len(buffer)):
                    G += (gamma ** (j - idx)) * buffer[j][2]   
                
                N[s[0], s[1], a] += 1
                Q[s[0], s[1], a] += (G - Q[s[0], s[1], a]) / N[s[0], s[1], a]
            buffer.clear()#清空缓存区
            break 

        # ---------- 轨迹达到n步结束----------
        if len(buffer) >= n_step:
            s,a,_,_,_,_, = buffer[0]#计算第一个状态动作对
            G = 0.0
            for i in range(n_step):
                G += (gamma ** i) * buffer[i][2] #走n步的真实回报
            if type == 'SARSA':
            #加上SARSA中最后SA的Q估计
                G += (gamma ** n_step) * Q[buffer[-1][3][0], buffer[-1][3][1], buffer[-1][4]]
            elif type == 'Q':
                G += (gamma ** n_step) * np.max(Q[buffer[-1][3][0], buffer[-1][3][1], :])
            N[s[0],s[1],a] += 1
            Q[s[0],s[1],a] += (G-Q[s[0],s[1],a])/N[s[0],s[1],a]
            buffer.pop(0)   #滑动移除第一个状态动作对
        state = next_s
        action = next_action
        step += 1

    # ----------达到每个回合的最多步数后Buff内还剩余一些状态动作对 ----------
    while buffer:
        for i in range(len(buffer)):
            s, a, _, _, _, _ = buffer[i]
            G = 0.0
            for j in range(i, len(buffer)):
                G += (gamma ** (j - i)) * buffer[j][2]#走n步的真实回报
            if type == 'SARSA':
            #加上SARSA中最后SA的Q估计
                G += (gamma ** n_step) * Q[buffer[-1][3][0], buffer[-1][3][1], buffer[-1][4]]
            elif type == 'Q':
                G += (gamma ** n_step) * np.max(Q[buffer[-1][3][0], buffer[-1][3][1], :])
            N[s[0], s[1], a] += 1
            Q[s[0], s[1], a] += (G - Q[s[0], s[1], a]) / N[s[0], s[1], a]
        buffer.clear()

    # 探索率指数衰减
    epsilon = max(epsilon_end, epsilon * 0.9999)

    if (ep + 1) % 20000 == 0:
        print(f"{ep + 1} / {num_episodes} episodes are done")

# =================== Compute final policy & value function ===================
final_policy_indices = np.zeros((number_of_row, number_of_column), dtype=int)
V = np.zeros((number_of_row, number_of_column))

for r in range(number_of_row):
    for c in range(number_of_column):
        best_a = np.argmax(Q[r, c, :])
        final_policy_indices[r, c] = best_a
        if (r, c) == target_state:
            V[r, c] = reward_target
        elif (r, c) in obstacle_state:
            V[r, c] = reward_obstacle
        else:
            V[r, c] = np.max(Q[r, c, :])
#stay at target
for s in range(state_space):
    r = s // number_of_column
    c = s % number_of_column
    if (r, c) == target_state:
        final_policy_indices[r,c] = 4
        V[r, c] = 0.0
# =================== Plot ===================
plt.ioff()
fig, (ax_policy, ax_value) = plt.subplots(1, 2, figsize=(12, 5))

for ax in (ax_policy, ax_value):
    ax.set_xlim(0.5, number_of_column + 1.5)
    ax.set_ylim(0.5, number_of_row + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # obstacles
    for (row, col) in obstacle_state:
        ax.add_patch(patches.Rectangle(
            (col + 1, number_of_row - row), 1, 1,
            facecolor="#E60F0FD2", edgecolor='black', linewidth=0.5))
    # target
    ax.add_patch(patches.Rectangle(
        (target_state[1] + 1, number_of_row - target_state[0]), 1, 1,
        facecolor="#26C30ED2", edgecolor='black', linewidth=0.5))
    # grid
    for row in range(number_of_row):
        for col in range(number_of_column):
            ax.add_patch(patches.Rectangle(
                (col + 1, number_of_row - row), 1, 1,
                facecolor='none', edgecolor='black', linewidth=0.5))
    # coordinates
    for col in range(0, number_of_column):
        ax.text(col + 1.5, number_of_row + 1.2, str(col),
                ha='center', va='center', fontsize=10)
    for row in range(0, number_of_row):
        ax.text(0.5, number_of_row - row + 0.5, str(row),
                ha='center', va='center', fontsize=10)

# policy arrows
arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: 'o'}
for r in range(number_of_row):
    for c in range(number_of_column):
        if (r, c) in obstacle_state:
            continue
        ax_policy.text(c + 1.5, number_of_row - r + 0.5,
                       arrow_map[final_policy_indices[r, c]],
                       ha='center', va='center', fontsize=12)

# value numbers
for r in range(number_of_row):
    for c in range(number_of_column):
        ax_value.text(c + 1.5, number_of_row - r + 0.5,
                      f'{V[r, c]:.1f}',
                      ha='center', va='center', fontsize=10)

ax_policy.set_title('Policy (n-step SARSA)')
ax_value.set_title('Value Function')
fig.suptitle(f'{n_step} step {type} |start = {start_state}|episodes = {num_episodes}')
plt.tight_layout()
plt.show()
