import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

number_of_column = 6
number_of_row = 5

target_state = (3, 2)
obstacle_state = [(1, 2), (2, 2), (2, 4), (3, 1), (3, 3)]

state_space = number_of_column * number_of_row
reward_target = 1
reward_obstacle = -10
reward_step = 0
gamma = 0.9

# up, right, down, left, stay
actions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
number_of_actions = len(actions)

def next_state_and_reward(state, action_idx):
    a_row, a_column = actions[action_idx]
    new_row = state[0] + a_row
    new_column = state[1] + a_column

    # Boundary checking
    if not (0 <= new_row < number_of_row and 0 <= new_column < number_of_column):
        return state, reward_obstacle

    new_state = (new_row, new_column)

    if new_state in obstacle_state:
        return new_state, reward_obstacle

    if new_state == target_state:
        return new_state, reward_target

    return new_state, reward_step

policy = np.zeros((state_space, number_of_actions))
policy[:, 0] = 1
v = np.zeros((number_of_row, number_of_column))
q_table = np.zeros((state_space, number_of_actions))

#plot
fig, (ax_policy, ax_value) = plt.subplots(1, 2, figsize=(12, 5))

for ax in (ax_policy, ax_value):
    ax.set_xlim(0.5, number_of_column + 1.5)
    ax.set_ylim(0.5, number_of_row + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # obstacle
    for (r, c) in obstacle_state:
        ax.add_patch(patches.Rectangle(
            (c + 1, number_of_row - r), 1, 1,
            facecolor="#E60F0FD2", edgecolor='black', linewidth=0.5, zorder=1))

    # target
    ax.add_patch(patches.Rectangle(
        (target_state[1] + 1, number_of_row - target_state[0]), 1, 1,
        facecolor="#26C30ED2", edgecolor='black', linewidth=0.5, zorder=2))

    # white block
    for r in range(number_of_row):
        for c in range(number_of_column):
            ax.add_patch(patches.Rectangle(
                (c + 1, number_of_row - r), 1, 1,
                facecolor='none', edgecolor='black', linewidth=0.5, zorder=2))

    # coordinate
    for col in range(1, number_of_column + 1):
        ax.text(col + 0.5, number_of_row + 1.2, str(col),
                ha='center', va='center', fontsize=10)
    for row in range(1, number_of_row + 1):
        ax.text(0.2, number_of_row - row + 1.5, str(row),
                ha='center', va='center', fontsize=10)


arrow_texts = np.empty((number_of_row, number_of_column), dtype=object)
for r in range(number_of_row):
    for c in range(number_of_column):
        txt = ax_policy.text(c + 1.5, number_of_row - r + 0.5, '↑',
                             ha='center', va='center', fontsize=12, color='black')
        arrow_texts[r, c] = txt


value_texts = np.empty((number_of_row, number_of_column), dtype=object)
for r in range(number_of_row):
    for c in range(number_of_column):
        txt = ax_value.text(c + 1.5, number_of_row - r + 0.5, '0.0',
                            ha='center', va='center', fontsize=12, color='black')
        value_texts[r, c] = txt

ax_policy.set_title('Policy', pad=15)
ax_value.set_title('Value', pad=15)
iter_text = fig.text(0.45, 0.05, 'Iteration: 0', ha='center', va='center', fontsize=12)

# 值迭代
#本质是将策略评估截断为仅进行一次更新，直接将贝尔曼最优算子嵌入价值更新中。
#在每一次对所有状态的扫描中，直接通过 max 操作选择最优动作的价值，同时完成价值评估与隐式策略更新。
Iteration = 0
while True:
    v_pre = v.copy()
    v_new = np.zeros_like(v)
    for s in range(state_space):
        r = s // number_of_column
        c = s % number_of_column
        state = (r, c)

        #对于终点固定策略（停在原地），值函数为0
        if state == target_state:
            policy[s, :] = 0.0
            policy[s, 4] = 1.0   
            v[r, c] = 0.0
            continue
        #对每个状态的每个动作，用上一轮的V更新Q表
        for a in range(number_of_actions):
            new_state, reward = next_state_and_reward(state, a)
            nr, nc = new_state
            q_table[s, a] = reward + gamma * v_pre[nr, nc] #q←r+gamma*V(s')
        #根据Q表更新策略（贪婪）
        best_a = np.argmax(q_table[s, :])
        policy[s, :] = 0.0
        policy[s, best_a] = 1.0

        v_new[r, c] = q_table[s, best_a]#把每个状态对应最大的Q值作为新一轮的V
    v= v_new
    Iteration += 1
    if np.linalg.norm(v-v_pre) < 0.01:
        break

# plot
arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: 'o'}
for r in range(number_of_row):
    for c in range(number_of_column):
        s = r * number_of_column + c
        a = np.argmax(policy[s, :])
        arrow_texts[r, c].set_text(arrow_map[a])
        value_texts[r, c].set_text(f'{v[r, c]:.1f}')

iter_text.set_text(f'Iteration: {Iteration}')
plt.show()
print(f"Value iteration convergence,the number of convergence is {Iteration}")
  

