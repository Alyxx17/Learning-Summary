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

def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])
def astar(start,end):
    h_start = heuristic(start,end)
    #(f, g, (r,c), path)
    heap = [(h_start,0,start,[start])]
    closed =set()

    while heap:
        f,g,(r,c),path = heapq.heappop(heap)

        if (r,c) in closed:
            continue
        closed.add((r,c))

        if (r,c) == end:
            return path
        
        for dr,dc in actions:
            nr,nc = r+dr,c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 0:
                ng = g+1
                nh = heuristic((nr,nc),end)
                nf = ng+nh
                heapq.heappush(heap,(nf,ng,(nr,nc),path+[(nr,nc)]))

    return None

result = astar(start,end)
if result:
    print(f"找到路径，步数：{len(result)-1}")
    for i ,pos in enumerate(result):
        print(f"第{i}步：{pos}")
else:
    print("无解")

