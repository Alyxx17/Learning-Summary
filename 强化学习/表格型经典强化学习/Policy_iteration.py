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


fig, (ax_policy, ax_value) = plt.subplots(1, 2, figsize=(12, 5))

for ax in (ax_policy, ax_value):
    ax.set_xlim(0.5, number_of_column + 1.5)
    ax.set_ylim(0.5, number_of_row + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    #obstacle
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

# arrow
arrow_texts = np.empty((number_of_row, number_of_column), dtype=object)
for r in range(number_of_row):
    for c in range(number_of_column):
        txt = ax_policy.text(c + 1.5, number_of_row - r + 0.5, '↑',
                             ha='center', va='center', fontsize=12, color='black')
        arrow_texts[r, c] = txt

# text
value_texts = np.empty((number_of_row, number_of_column), dtype=object)
for r in range(number_of_row):
    for c in range(number_of_column):
        txt = ax_value.text(c + 1.5, number_of_row - r + 0.5, '0.0',
                            ha='center', va='center', fontsize=12, color='black')
        value_texts[r, c] = txt

ax_policy.set_title('Policy', pad=15)
ax_value.set_title('Value', pad=15)
iter_text = fig.text(0.45, 0.05, 'Iteration: 0', ha='center', va='center', fontsize=12)

# ==================== PI 核心代码 ====================
#本质是“彻底评估”与“贪婪提升”的交替循环
Iteration = 0
policy_stable = False

#对于终点固定策略（停在原地），值函数为0
for s in range(state_space):
    r = s // number_of_column
    c = s % number_of_column
    if (r, c) == target_state:
        policy[s, :] = 0.0
        policy[s, 4] = 1.0
        v[r, c] = 0.0

while not policy_stable:#策略未收敛就循环
    # ---------- 策略评估 ----------
    while True:
        v_old = v.copy()
        #对每个状态，采取当前策略的动作（贪婪),得到下一个状态，用贝尔曼公式更新每个状态的值函数
        #前后值函数进行对比，直至收敛，本质是拿掉了贝尔曼最优公式的argmax，先进行V的不动迭代V(s')=r+gamma*V(s')
        #实际上np.max(np.abs(v - v_old)) < 0.01只是广义策略迭代，小于一定的阈值就停止，并未收敛到理论最优。
        for s in range(state_space):
            r = s // number_of_column
            c = s % number_of_column
            state = (r, c)
            if state == target_state:
                continue
            a = np.argmax(policy[s, :])
            new_state, reward = next_state_and_reward(state, a)
            nr, nc = new_state
            # update value function
            v[r, c] = reward + gamma * v_old[nr, nc]
        if np.max(np.abs(v - v_old)) < 0.01:
            break

    # ---------- 策略提升 ----------
    policy_stable = True#假设策略已稳定
    #对每个状态
    for s in range(state_space):
        r = s // number_of_column
        c = s % number_of_column
        state = (r, c)
        if state == target_state:
            continue
        #提取每个状态的旧策略
        old_action = np.argmax(policy[s, :])
        for a in range(number_of_actions):
            new_state, reward = next_state_and_reward(state, a)
            nr, nc = new_state
            q_table[s, a] = reward + gamma * v[nr, nc]#用策略评估得到的V更新Q表。
            #Q表(s,a),代表在当前策略下每一个状态的每一个动作后的价值函数
        best_a = np.argmax(q_table[s, :])#根据Q表更新策略（贪婪）
        if old_action != best_a:#某个状态新旧策略不一样，说明未达最优策略
            policy_stable = False#继续循环策略评估
        policy[s, :] = 0.0
        policy[s, best_a] = 1.0#更新下当前的策略，用于下一轮的策略评估

    Iteration += 1

# ==================== plot ====================
arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: 'o'}
for r in range(number_of_row):
    for c in range(number_of_column):
        s = r * number_of_column + c
        a = np.argmax(policy[s, :])
        arrow_texts[r, c].set_text(arrow_map[a])
        value_texts[r, c].set_text(f'{v[r, c]:.1f}')

iter_text.set_text(f'Iteration: {Iteration}')
plt.show()
print(f"Policy iteration convergence, the number of iterations is {Iteration}")