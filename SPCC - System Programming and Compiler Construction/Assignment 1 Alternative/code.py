file = open("input.asm", "r")

idx = 0
symboltable = {}
literaltable = {}

opcodes = ["add", "sub", "mov", "jmp", "end"]

for line in file:
    words = line.split()

    if len(words) == 0:
        continue

    if words[0].lower() == "start":
        idx = int(words[1])
        continue

    # label detection
    if words[0].lower() not in opcodes:
        symboltable[words[0]] = idx
        operands = words[1:]
    else:
        operands = words[1:]

    # operand detection
    for op in operands:
        if op.isalpha() and op not in symboltable:
            symboltable[op] = "NULL"

        if op.startswith("="):
            literaltable[op] = "NULL"

    idx += 1


print("Symbol Table")
print(symboltable)

print("Literal Table")
print(literaltable)
