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
goal_mask  =(1<<num_rewards)-1

def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
def astar_reward(start,end):
    start_mask = 0
    h_start = heuristic(start,end)
    # (f, g, r, c, mask, path_tuple)
    heap = [(h_start,0,start[0],start[1],start_mask,(start,))]
    closed = set()
    while heap:
        f,g,r,c,mask,path = heapq.heappop(heap)

        state = (r,c,mask)
        if state in closed:
            continue
        closed.add(state)

        if(r,c) == end and mask == goal_mask:
            return list(path)
        
        for dr,dc in actions:
            nr,nc  = r+dr,c+dc
            if not (0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 0):
                continue
            if (nr,nc) in path:
                continue

            new_mask = mask

            if (nr,nc) in reward_index:
                idx = reward_index[(nr,nc)]
                if not (mask &(1<<idx)):
                    new_mask = mask | (1<<idx)
            
            ng = g+1
            nh = heuristic((nr,nr),end)
            nf = ng+nh
            new_path = path + ((nr, nc),)
            heapq.heappush(heap, (nf, ng, nr, nc, new_mask, new_path))

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

