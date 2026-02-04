file = open("input.asm", "r")
idx = 0
symboltable = {}
opcodes = ["add", "sub", "mov", "jmp", "end"]

for line in file:
    words = line.split()
    if len(words) == 1:
        continue

    if words[0].lower() == "start":
        idx = int(words[-1]) - 1

    elif len(words) >= 3:
        if words[0].lower() not in opcodes:
            symboltable[words[0]] = idx

        if words[-1].isalpha() and words[-1] not in symboltable:
            symboltable[words[-1]] = 'NULL'

    elif words[-1].isalpha():
        symboltable[words[-1]] = 'NULL'

    elif words[0].isalnum() and not words[0].isalpha():
        if words[0] in symboltable:
            symboltable[words[0]] = idx

    idx += 1

for key, value in symboltable.items():
    print(key, value)
