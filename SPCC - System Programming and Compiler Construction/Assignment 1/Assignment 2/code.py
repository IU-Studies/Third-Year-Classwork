# ---------- TABLE STRUCTURES ----------

MOT = [
    {"mnemonic": "STOP", "opcode": 0},
    {"mnemonic": "ADD", "opcode": 1},
    {"mnemonic": "SUB", "opcode": 2},
    {"mnemonic": "MUL", "opcode": 3},
    {"mnemonic": "MOV", "opcode": 4}
]

POT = [
    {"directive": "START", "type": "AD", "code": 1},
    {"directive": "END", "type": "AD", "code": 2},
    {"directive": "LTORG", "type": "AD", "code": 3},
    {"directive": "DS", "type": "DL", "code": 1},
    {"directive": "DC", "type": "DL", "code": 2}
]

SYMTAB = []
LITTAB = []

symCnt = 0
litCnt = 0
LC = 0
poolPtr = 0


# ---------- UTILITY FUNCTIONS ----------

def searchMOT(op):
    for i in range(len(MOT)):
        if MOT[i]["mnemonic"] == op:
            return i
    return -1


def searchPOT(op):
    for i in range(len(POT)):
        if POT[i]["directive"] == op:
            return i
    return -1


def addSymbol(sym, addr):
    global symCnt
    for i in range(len(SYMTAB)):
        if SYMTAB[i]["symbol"] == sym:
            return i

    SYMTAB.append({"symbol": sym, "address": addr})
    symCnt += 1
    return symCnt - 1


def addLiteral(lit):
    global litCnt
    for i in range(len(LITTAB)):
        if LITTAB[i]["literal"] == lit:
            return i

    LITTAB.append({"literal": lit, "address": -1})
    litCnt += 1
    return litCnt - 1


def assignLiterals():
    global LC, poolPtr
    for i in range(poolPtr, litCnt):
        LITTAB[i]["address"] = LC
        LC += 1
    poolPtr = litCnt


# ---------- MAIN PROGRAM ----------

with open("input.asm", "r") as file:

    for line in file:
        parts = line.strip().split()

        if len(parts) == 3:
            label, opcode, operand = parts
        elif len(parts) == 2:
            label = "-"
            opcode, operand = parts
        else:
            continue

        # START
        if opcode == "START":
            LC = int(operand)
            print(f"(AD,01) (C,{LC})")
            continue

        # END
        if opcode == "END":
            assignLiterals()
            print("(AD,02)")
            break

        # LTORG
        if opcode == "LTORG":
            assignLiterals()
            print("(AD,03)")
            continue

        # Add label to symbol table
        if label != "-":
            addSymbol(label, LC)

        # Declarative statements
        if opcode == "DS":
            print(f"(DL,01) (C,{int(operand)})")
            LC += int(operand)
            continue

        if opcode == "DC":
            print(f"(DL,02) (C,{int(operand)})")
            LC += 1
            continue

        # Imperative statements
        m = searchMOT(opcode)
        if m != -1:
            print(f"(IS,{MOT[m]['opcode']:02d}) ", end="")

            if operand.startswith("="):
                l = addLiteral(operand)
                print(f"(L,{l})")
            else:
                s = addSymbol(operand, -1)
                print(f"(S,{s})")

            LC += 1


# ---------- DISPLAY TABLES ----------

print("\nSYMBOL TABLE")
print("Index\tSymbol\tAddress")

for i in range(len(SYMTAB)):
    print(f"{i}\t{SYMTAB[i]['symbol']}\t{SYMTAB[i]['address']}")

print("\nLITERAL TABLE")
print("Index\tLiteral\tAddress")

for i in range(len(LITTAB)):
    print(f"{i}\t{LITTAB[i]['literal']}\t{LITTAB[i]['address']}")
