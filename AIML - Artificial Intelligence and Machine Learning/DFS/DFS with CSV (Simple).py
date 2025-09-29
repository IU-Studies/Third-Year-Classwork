import csv

# Read adjacency matrix from CSV
def load_graph(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        return [list(map(int, row)) for row in reader]

# Simple recursive DFS
def dfs(graph, node, visited):
    visited[node] = True
    print("Visited:", node)

    for neighbor in range(len(graph[node])):
        if graph[node][neighbor] == 1 and not visited[neighbor]:
            dfs(graph, neighbor, visited)

# Main part
graph = load_graph("graph.csv")
visited = [False] * len(graph)
start_node = 0

dfs(graph, start_node, visited)
