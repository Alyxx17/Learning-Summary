from collections import deque


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

def bfs(start,end):
    queue = deque()
    queue.append([start])
    visited = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == end:
            return path

        r,c  = current
        for dr,dc in  actions:
            nr,nc = r+dr,c+dc

            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] ==0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                new_path = path +[(nr,nc)]
                queue.append(new_path)

    return None


result = bfs(start,end)
if result:
    print(f"找到路径，步数{len(result)-1}")
    for i,pos in enumerate(result):
        print(f"第{i}步:{pos}")
else:
    print("无解")






    