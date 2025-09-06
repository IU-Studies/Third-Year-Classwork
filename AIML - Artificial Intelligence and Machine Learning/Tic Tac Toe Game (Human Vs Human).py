def draw_board(cells):
    def row(i):
        print(f"{cells[i]} | {cells[i+1]} | {cells[i+2]}")
    sep = "--+---+--"
    row(0)
    print(sep)
    row(3)
    print(sep)
    row(6)

cells = [str(i) for i in range(9)]

win = [{0,1,2}, {3,4,5}, {6,7,8}, {0,3,6}, {1,4,7}, {2,5,8}, {0,4,8}, {2,4,6}]

list1 = []
list2 = []

for turn in range(9):
    draw_board(cells)
    if turn % 2 == 0:
        currp = 1
    else:
        currp = 2
    if currp == 1:
        plist = list1
        symbol = 'X'
    else:
        plist = list2
        symbol = 'O'


    move = int(input(f"Player {currp}'s turn ({symbol}): "))

    if move < 0 or move > 8 or cells[move] in ['X', 'O']:
        print("Invalid move. Try again.")
        continue

    cells[move] = symbol
    plist.append(move)


    if len(plist) >= 3:
        for combo in win:
            if combo.issubset(set(plist)):
                draw_board(cells)
                print(f"Player {currp} ({symbol}) wins!")
                exit()


draw_board(cells)
print("It's a draw!")
