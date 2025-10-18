import heapq

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(node, grid):
    x, y = node
    rows, cols = len(grid), len(grid[0])
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-neighbors
        nx, ny = x+dx, y+dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            yield (nx, ny)

def reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))

def astar(grid, start, goal):
    if start == goal:
        return [start]
    open_heap = []
    g = {start: 0}
    f = {start: manhattan(start, goal)}
    came_from = {}

    heapq.heappush(open_heap, (f[start], start))

    closed = set()
    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return reconstruct(came_from, current)
        closed.add(current)

        for n in neighbors(current, grid):
            tentative_g = g[current] + 1  # cost of moving to neighbor = 1
            if tentative_g < g.get(n, float('inf')):
                came_from[n] = current
                g[n] = tentative_g
                f[n] = tentative_g + manhattan(n, goal)
                heapq.heappush(open_heap, (f[n], n))
    return None  # no path

# Example usage
if __name__ == "__main__":
    grid = [
        [0,0,0,0],
        [0,1,1,0],
        [0,0,0,0],
        [0,1,0,0],
    ]
    start = (0,0)
    goal = (3,3)
    path = astar(grid, start, goal)
    print("Path:", path)
