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








# Function to draw the Tic-Tac-Toe board
def draw_board(cells):
    def row(i):
        print(f"{cells[i]} | {cells[i + 1]} | {cells[i + 2]}")
    sep = "--+---+--"
    row(0)
    print(sep)
    row(3)
    print(sep)
    row(6)

# Initialize the board with cell numbers 0–8
cells = [str(i) for i in range(9)]

# Define all possible winning combinations
win = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
       {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
       {0, 4, 8}, {2, 4, 6}]

# Lists to store moves of Player 1 and Player 2
list1 = []
list2 = []
won = False

# Game loop for maximum 9 turns
for turn in range(9):
    draw_board(cells)

    # Decide current player
    if turn % 2 == 0:
        currp = 1
    else:
        currp = 2

    # Assign move list and symbol for current player
    if currp == 1:
        plist = list1
        symbol = 'X'
    else:
        plist = list2
        symbol = 'O'

    # Take player input for move
    move = int(input(f"Player {currp} turn ({symbol}): "))

    # Validate move
    if move < 0 or move > 8 or cells[move] in ['X', 'O']:
        print("Invalid move, please try again")
        continue

    # Update board and store move
    cells[move] = symbol
    plist.append(move)

    # Check for winning condition
    if len(plist) >= 3:
        for i in win:
            if i.issubset(set(plist)):
                draw_board(cells)
                print(f"Player {currp} won!")
                won = True
                exit()

# If no winner, declare draw
draw_board(cells)
if not won:
    print("It's a draw")






0,1,0,0
1,0,1,1
0,1,0,0
0,1,0,0





,2,3,4,5,7,8,10,11,13,14
2,0,0,0,0,0,0,0,0,0,0
3,1,0,1,0,0,0,0,0,0,0
4,0,0,0,0,0,1,0,0,0,0
5,0,1,0,0,1,0,0,0,0,0
7,0,0,0,0,0,1,0,0,0,0
8,0,0,0,0,0,0,1,0,1,1
10,0,0,0,0,0,0,0,1,0,0
11,0,0,0,0,0,0,0,0,0,0
13,0,0,0,0,0,0,1,0,0,0
14,0,0,0,0,0,0,0,0,0,0














