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
