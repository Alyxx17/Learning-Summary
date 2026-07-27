import heapq


grid = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]
rows, cols = len(grid), len(grid[0])
start = (0, 0)
end = (1, 4)
actions = [(-1,0), (1,0), (0,-1), (0,1)]
reward_states = [(2,2), (2,4), (4,2), (4,4), (6,2), (6,4), (5,1), (2,6)]
num_rewards = len(reward_states)
reward_index = {pos:i for i ,pos in enumerate(reward_states)}
goal_mask  =(1<<num_rewards)-1#goal_mask 的二进制低 num_rewards 位全部为 1。
#定义启发式函数，从起点到终点的曼哈顿距离
def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar_reward(start,end):
    start_mask = 0
    h_start = heuristic(start,end)
    #堆： (f, g, r, c, mask, path_tuple)
    #f = g + h：总估算代价，堆以此值排序，优先弹出最优节点，初始化为h，因为g=0
    #g：从起点走到当前点的实际步数（代价）
    #h：启发函数值(曼哈顿距离)
    #状态 (r, c, mask)：表示“人在 (r,c)，且已收集到的奖励集合为 mask”
    #path：一个元组，记录从起点到当前点的完整行走路径。
    heap = [(h_start,0,start[0],start[1],start_mask,(start,))]
    closed = set()
    while heap:
        #弹出最优节点，即f最小的元组
        f,g,r,c,mask,path = heapq.heappop(heap)

        state = (r,c,mask)
        #如果该元组内的状态(r,c,mask)已经在闭集内，代表已经出现过，跳过
        if state in closed:
            continue
        #该元组内的状态(r,c,mask)第一次出现，加入闭集
        closed.add(state)
        #到达终点且奖励收集完毕，返回路径
        if(r,c) == end and mask == goal_mask:
            return list(path)
        #从当前状态拓展四周，每一个方向都进行如下检查与收集拼接
        for dr,dc in actions:
            nr,nc  = r+dr,c+dc
            #如果越界/遇到障碍物，跳过
            if not (0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 0):
                continue
            #走回头路，跳过
            if (nr,nc) in path:
                continue
            #继承上一个节点的奖励掩码
            new_mask = mask
            #如果新的格子是奖励点
            if (nr,nc) in reward_index:
                #找到奖励索引
                idx = reward_index[(nr,nc)]
                #与运算，判断如果没收集过，进行或运算将mask该位置1，代表收集
                if not (mask &(1<<idx)):
                    new_mask = mask | (1<<idx)
            #实际步数加1
            ng = g+1
            nh = heuristic((nr,nc),end)# 估算到终点的曼哈顿距离
            nf = ng+nh#新的f
            new_path = path + ((nr, nc),) # 拼接路径
            heapq.heappush(heap, (nf, ng, nr, nc, new_mask, new_path))
            #若四个方向均可行，则会在堆内压入4个元组，回到最开始循环，弹出堆内符合条件的元组，以此循环。
    return None

# 运行
result = astar_reward(start, end)
if result:
    print(f"找到路径，总步数：{len(result)-1}")
    for i, pos in enumerate(result):
        print(f"{i}: {pos}")
    # 验证所有奖励都已收集
    collected = [pos for pos in result if pos in reward_index]
    print(f"收集到的奖励：{collected}")
else:
    print("无解")

