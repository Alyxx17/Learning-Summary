import  heapq


grid = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]

rows = len(grid)
cols = len(grid[0])

start = (0,0)
end = (1,4)

actions = [(-1,0),(0,1),(1,0),(0,-1)]

#定义启发式函数，从起点到终点的曼哈顿距离
def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar(start,end):
    #计算起点到终点的启发值，作为初始f值（此时g=0）
    h_start = heuristic(start,end)
    #堆： (f, g, (r,c), path)
    #f = g + h：总估算代价，堆以此值排序，优先弹出最优节点，初始化为h，因为g=0
    #g：从起点走到当前点的实际步数（代价）
    #(r,c)：当前所在位置
    #path：列表，记录从起点到当前点的完整行走路径
    heap = [(h_start,0,start,[start])]
    closed = set()  #闭集，记录已经处理过的位置（不包含掩码，因为本算法无奖励收集）

    while heap:
        #弹出最优节点，即f最小的元组
        f,g,(r,c),path = heapq.heappop(heap)

        #如果该位置已经在闭集中，代表已经出现过，跳过（不需要再扩展）
        if (r,c) in closed:
            continue
        #该位置第一次出现，加入闭集
        closed.add((r,c))

        #到达终点，返回路径（因为A*保证首次弹出终点时路径最优）
        if (r,c) == end:
            return path
        
        #从当前状态拓展四周，每一个方向都进行如下检查与收集拼接
        for dr,dc in actions:
            nr,nc = r+dr,c+dc
            #如果越界或遇到障碍物，跳过
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 0:
                #实际步数加1
                ng = g+1
                #计算新位置的启发值（到终点的曼哈顿距离）
                nh = heuristic((nr,nc),end)
                #新的总估算代价
                nf = ng+nh
                #将新节点压入堆中，路径列表通过拼接生成新列表
                heapq.heappush(heap,(nf,ng,(nr,nc),path+[(nr,nc)]))

    #若堆为空仍未找到路径，返回None表示无解
    return None

result = astar(start,end)
if result:
    print(f"找到路径，步数：{len(result)-1}")
    for i ,pos in enumerate(result):
        print(f"第{i}步：{pos}")
else:
    print("无解")

