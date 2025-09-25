graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': ['13', '14'],
    '13': ['10'],
    '14': [],
    '10': ['11'],
    '11': []
}

visited = set()  

def dfs(graph, start):
    if start not in visited:
        print(start, end=" ")
        visited.add(start)

        for i in graph[start]:
            dfs(graph, i)

dfs(graph, '5')
