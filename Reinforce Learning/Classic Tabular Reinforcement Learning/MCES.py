import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =================== Configure ===================
number_of_column = 6
number_of_row = 5

target_state = (3, 2)                     
obstacle_state = [(1, 2), (2, 2), (2, 4), (3, 1), (3, 3)]  

state_space = number_of_column * number_of_row
reward_target = 10
reward_obstacle = -5
reward_step = 0
gamma = 0.9

actions = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
number_of_actions = len(actions)

def next_state_and_reward(state, action_idx):
    a_row, a_column = actions[action_idx]
    new_row = state[0] + a_row
    new_column = state[1] + a_column

    # boundary check
    if not (0 <= new_row < number_of_row and 0 <= new_column < number_of_column):
        return state, reward_obstacle, False

    new_state = (new_row, new_column)

    # obstacle check
    if new_state in obstacle_state:
        return new_state, reward_obstacle, True

    # target check
    if new_state == target_state:
        return new_state, reward_target, True

    return new_state, reward_step, False

# =================== MCES ===================

visit_type = 'every'     # first:first-visit; every:every-visit
num_episodes = 100000    # max episodes
max_steps_per_episode = 200  

Q = np.zeros((number_of_row, number_of_column, number_of_actions))
N = np.zeros((number_of_row, number_of_column, number_of_actions))

# 策略初始化
policy = np.zeros((number_of_row, number_of_column, number_of_actions))
for r in range(number_of_row):
    for c in range(number_of_column):
        if (r, c) == target_state:
            policy[r, c, 4] = 1.0
        else:
            a = np.random.randint(number_of_actions)
            policy[r, c, a] = 1.0

non_terminal_states = [(r, c) for r in range(number_of_row) for c in range(number_of_column)
                       if (r, c) != target_state and (r, c) not in obstacle_state]

print(f"MCES ({visit_type}-visit) starts train,all episodes:{num_episodes}")
#当没有模型时，只能与环境大量交互
#从任意状态动作对出发，按照当前策略，直至终点/遇到障碍物，算作一条轨迹。
#然后计算这条轨迹上每一个状态到最后一个状态的平均回报（G），比如一条轨迹从s0出发,经过s1,s2,到s3结束。分别拿到奖励r1,r2,r3
#那么就要计算s0到s3的G，s1到s3的G，s2到s3的G（每个状态都是反向计算，为了方便简洁）
#比如s0到s3应该为r1+gamma*r2+gamma^2*r3,反过来就是第一次：G =r3；第二次，G = r2+gamma*G；第三次:G = r1 +gamma*G
#即使这条轨迹是遇到障碍物结束了，该轨迹上所有状态返回的是负的回报，只要别的大量的轨迹到达终点，也会得到正的回报，最后大数定理发力，得到真值
#只要样本足够多就可以得到每个状态动作对的价值真值
#首次访问就是这条轨迹内的状态动作只计算最后一次出现的，反向计算时再遇到同样的状态动作就跳过。每次访问就是不管，遇到就是计算G后增量更新Q。
#每生成一条轨迹，就计算G，更新Q，策略更新，无收敛条件，生成轨迹越多越准。
for ep in range(num_episodes):
    # ----探索起点，每次的起点选取任意状态动作对 ----
    s0 = non_terminal_states[np.random.randint(len(non_terminal_states))]
    a0 = np.random.randint(number_of_actions)

    episode = []          
    state = s0
    action = a0
    step = 0
    done = False
    #-----创建一条轨迹
    while not done and step < max_steps_per_episode:
        next_s, reward, done = next_state_and_reward(state, action)
        episode.append((state, action, reward))

        if not done:
            state = next_s
            action = np.argmax(policy[state[0], state[1], :])
        step += 1

    
    G = 0.0
    visited_sa = set()  
    #计算这条轨迹内所有状态动作对的Q
    for t in reversed(range(len(episode))):
        s, a, r = episode[t]
        G = r + gamma * G#反向计算G

        if visit_type == 'first':
            if (s, a) not in visited_sa:
                N[s[0], s[1], a] += 1
                Q[s[0], s[1], a] += (G - Q[s[0], s[1], a]) / N[s[0], s[1], a]#增量更新Q
                visited_sa.add((s, a))
        else:  
            N[s[0], s[1], a] += 1
            Q[s[0], s[1], a] += (G - Q[s[0], s[1], a]) / N[s[0], s[1], a]

    # 根据Q，策略提升
    for r in range(number_of_row):
        for c in range(number_of_column):
            if (r, c) == target_state or (r, c) in obstacle_state:
                continue
            max_value = np.max(Q[r,c,:])
            best_actions = np.where(Q[r,c,:]== max_value )[0]
            new_a = np.random.choice(best_actions)
            policy[r,c,:] = 0 
            policy[r,c,new_a] = 1

    if (ep + 1) % 20000 == 0:
        print(f"{ep + 1} / {num_episodes} episodes are done")

print("Finishing the train and ready to plot......")

# =================== 根据最终的Q表得到V ===================
final_policy_indices = np.zeros((number_of_row, number_of_column), dtype=int)
V = np.zeros((number_of_row, number_of_column))

for r in range(number_of_row):
    for c in range(number_of_column):
        a_best = np.argmax(policy[r, c, :])
        final_policy_indices[r, c] = a_best
        if (r, c) == target_state:
            V[r, c] = reward_target
        elif (r, c) in obstacle_state:
            V[r, c] = reward_obstacle
        else:
            V[r, c] = np.max(Q[r, c, :])  

# =================== Plot ===================
plt.ioff()
fig, (ax_policy, ax_value) = plt.subplots(1, 2, figsize=(12, 5))

for ax in (ax_policy, ax_value):
    ax.set_xlim(0.5, number_of_column + 1.5)
    ax.set_ylim(0.5, number_of_row + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    for (r, c) in obstacle_state:
        ax.add_patch(patches.Rectangle(
            (c + 1, number_of_row - r), 1, 1,
            facecolor="#E60F0FD2", edgecolor='black', linewidth=0.5, zorder=1))
    ax.add_patch(patches.Rectangle(
        (target_state[1] + 1, number_of_row - target_state[0]), 1, 1,
        facecolor="#26C30ED2", edgecolor='black', linewidth=0.5, zorder=2))
    for r in range(number_of_row):
        for c in range(number_of_column):
            ax.add_patch(patches.Rectangle(
                (c + 1, number_of_row - r), 1, 1,
                facecolor='none', edgecolor='black', linewidth=0.5, zorder=2))
    for col in range(1, number_of_column + 1):
        ax.text(col + 0.5, number_of_row + 1.2, str(col), ha='center', va='center', fontsize=10)
    for row in range(1, number_of_row + 1):
        ax.text(0.2, number_of_row - row + 1.5, str(row), ha='center', va='center', fontsize=10)

# Arrow
arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: 'o'}
for r in range(number_of_row):
    for c in range(number_of_column):
        if (r, c) in obstacle_state:
            continue
        a = final_policy_indices[r, c]
        ax_policy.text(c + 1.5, number_of_row - r + 0.5, arrow_map[a],
                       ha='center', va='center', fontsize=12, color='black')

# value
for r in range(number_of_row):
    for c in range(number_of_column):
        ax_value.text(c + 1.5, number_of_row - r + 0.5, f'{V[r, c]:.1f}',
                      ha='center', va='center', fontsize=12, color='black')

ax_policy.set_title('Optimal Policy (MC Exploring Starts)')
ax_value.set_title('Value Function')
fig.suptitle(f'MC {visit_type}-visit  |  Episodes: {num_episodes}')
plt.tight_layout()
plt.show()



