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
    #广度优先搜索（BFS），保证找到的路径是最短路径（无权图）
    queue = deque()  #队列，用于BFS逐层扩展
    queue.append([start])  #初始路径列表，只包含起点
    visited = {start}  #已访问集合，记录所有已经扩展过的位置

    while queue:
        #取出队首路径（最早加入的路径，保证BFS的先进先出）
        #逐层向外扩散的搜索逻辑，确保先被发现的节点先被处理。这种特性保证了在无权图中搜索到的第一条路径就是最短路径
        path = queue.popleft()
        current = path[-1]  #当前路径的最后一个节点，即当前位置

        #到达终点，返回完整路径（BFS首次到达终点时一定是最短路径）
        if current == end:
            return path

        r,c = current  #解包当前位置坐标
        #尝试向四个方向扩展
        for dr,dc in actions:
            nr,nc = r+dr,c+dc

            #检查新位置是否合法：在网格内、不是障碍物、并且未被访问过
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == 0 and (nr,nc) not in visited:
                visited.add((nr,nc))  #标记已访问，避免重复扩展
                new_path = path + [(nr,nc)]  #构造新路径（在原路径后追加新坐标）
                queue.append(new_path)  #将新路径加入队列尾部

    #如果队列为空仍未找到路径，返回None表示无解
    return None


result = bfs(start,end)
if result:
    print(f"找到路径，步数{len(result)-1}")
    for i,pos in enumerate(result):
        print(f"第{i}步:{pos}")
else:
    print("无解")






    
