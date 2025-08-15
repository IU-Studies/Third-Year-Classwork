graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': ['2', '13', '14'],
    '9': ['10'],
    '13': ['10'],
    '14': [],
    '10': ['11'],
    '11': []
}

def bfs(graph, start):
    visited = set()
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current not in visited:
            print(current, end=" ")
            visited.add(current)
            queue.extend(graph[current])

bfs(graph, '5')
