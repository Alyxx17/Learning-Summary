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

    
    if not (0 <= new_row < number_of_row and 0 <= new_column < number_of_column):
        return state,reward_bound, False

    new_state = (new_row, new_column)

    
    if new_state in obstacle_state:
        return new_state, reward_obstacle, True

    
    if new_state == target_state:
        return new_state, reward_target, True

    return new_state, reward_step, False

# =================== MC ε‑greedy (On‑Policy) ===================

visit_type = 'first'       
num_episodes = 200000     
max_steps_per_episode = 100
epsilon = 0.5             
epsilon_end = 0.1          

Q = np.zeros((number_of_row, number_of_column, number_of_actions))
N = np.zeros((number_of_row, number_of_column, number_of_actions))

# ε‑greedy 用于遍历所有状态动作对
def select_action(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(number_of_actions)
    else:
        return np.argmax(Q[state[0], state[1], :])

print(f" MC  ε-greedy start training, ε = {epsilon}")
#与MCES不同，因为MCSP只能从固定起点出发，于是需要策略可变，来探索所有状态动作对，然后更新Q表，选取Q最大的作为最终V
#探索率就是除最大Q对应策略之外的动作的概率，前期适当大，用于探索，中后期可以指数衰减，用于稳定
for ep in range(num_episodes):
    state = (2,3) #固定起点
    start =state
    action = select_action(state, Q, epsilon) 

    episode = []   
    step = 0
    done = False

    
    while not done and step < max_steps_per_episode:
        next_s, reward, done = next_state_and_reward(state, action)
        episode.append((state, action, reward))

        if not done:
            state = next_s
            action = select_action(state, Q, epsilon)
        step += 1

    
    G = 0.0
    visited_sa = set()

    for t in reversed(range(len(episode))):
        s, a, r = episode[t]
        G = r + gamma * G

        if visit_type == 'first':
            if (s, a) not in visited_sa:
                N[s[0], s[1], a] += 1
                Q[s[0], s[1], a] += (G - Q[s[0], s[1], a]) / N[s[0], s[1], a]
                visited_sa.add((s, a))
        else:   
            N[s[0], s[1], a] += 1
            Q[s[0], s[1], a] += (G - Q[s[0], s[1], a]) / N[s[0], s[1], a]
        # 指数衰减探索率
    epsilon = max(epsilon_end, epsilon * 0.9999)

    if (ep + 1) % 20000 == 0:
        print(f"{ep + 1} / {num_episodes}episodes are done")

print("Finishing the train and ready to plot......")


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
            V[r, c] = np.max(Q[r,c,:])

# =================== plot ===================
plt.ioff()
fig, (ax_policy, ax_value) = plt.subplots(1, 2, figsize=(12, 5))

for ax in (ax_policy, ax_value):
    ax.set_xlim(0.5, number_of_column + 1.5)
    ax.set_ylim(0.5, number_of_row + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    for (row, col) in obstacle_state:
        ax.add_patch(patches.Rectangle((col + 1, number_of_row - row), 1, 1, facecolor="#E60F0FD2", edgecolor='black', linewidth=0.5))
    ax.add_patch(patches.Rectangle((target_state[1] + 1, number_of_row - target_state[0]), 1, 1, facecolor="#26C30ED2", edgecolor='black', linewidth=0.5))
    
    for row in range(number_of_row):
        for col in range(number_of_column):
            ax.add_patch(patches.Rectangle((col + 1, number_of_row - row), 1, 1, facecolor='none', edgecolor='black', linewidth=0.5))
            
    for col in range(0, number_of_column ):
        ax.text(col + 1.5, number_of_row + 1.2, str(col), ha='center', va='center', fontsize=10)
    for row in range(0, number_of_row ):
        ax.text(0.5, number_of_row - row + 0.5, str(row), ha='center', va='center', fontsize=10)

arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: 'o'}
for r in range(number_of_row):
    for c in range(number_of_column):
        if (r, c) in obstacle_state or (r, c) == target_state: continue
        ax_policy.text(c + 1.5, number_of_row - r + 0.5, arrow_map[final_policy_indices[r, c]], ha='center', va='center', fontsize=12)

for r in range(number_of_row):
    for c in range(number_of_column):
        ax_value.text(c + 1.5, number_of_row - r + 0.5, f'{V[r, c]:.1f}', ha='center', va='center', fontsize=10)

ax_policy.set_title('Target Policy (Max Q)')
ax_value.set_title('Value Function V(s)')
fig.suptitle(f'MCSP| start:{start}')
plt.tight_layout()
plt.show()

