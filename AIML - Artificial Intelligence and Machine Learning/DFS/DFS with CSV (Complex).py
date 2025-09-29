import csv

# Read adjacency matrix from CSV with headers
def load_graph(filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

        # First row and column are node labels
        labels = data[0][1:]
        graph = {}

        for i in range(1, len(data)):
            node = data[i][0]
            edges = list(map(int, data[i][1:]))
            graph[node] = {labels[j]: edges[j] for j in range(len(edges))}

        return graph

# DFS using dictionary-based adjacency matrix
def dfs(graph, node, visited):
    if node in visited:
        return
    visited.add(node)
    print("Visited:", node)

    for neighbor in graph[node]:
        if graph[node][neighbor] == 1:
            dfs(graph, neighbor, visited)

# Main part
graph = load_graph("graph.csv")
start_node = '5'  # You can change this to any valid node label
visited = set()

dfs(graph, start_node, visited)
