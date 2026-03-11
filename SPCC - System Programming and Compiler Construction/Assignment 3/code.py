MAX = 20
LENGTH = 100


mnt = []
mdt = []

mnt_count = 0
mdt_count = 0

inside_macro = False

with open("input.asm", "r") as f:
    lines = [line.strip() for line in f]

i = 0
while i < len(lines):

    line = lines[i]

    if line == "END":
        print("END")
        break

    # Start MACRO definition
    if line == "MACRO":
        inside_macro = True

        i += 1
        macro_name = lines[i]

        mnt.append({
            "name": macro_name,
            "mdt_index": len(mdt)
        })

        mnt_count += 1
        i += 1
        continue

    # Store Macro Body
    if inside_macro:
        if line == "MEND":
            inside_macro = False
        else:
            mdt.append(line)
            mdt_count += 1
        i += 1
        continue

    # Check for Macro Call
    found = False

    for entry in mnt:
        if line == entry["name"]:
            print("\nExpanded Code:")

            index = entry["mdt_index"]
            while index < len(mdt):
                print(mdt[index])
                index += 1

            found = True
            break

    # Normal instruction
    if not found:
        print(line)

    i += 1


# Display Tables
print("\nMNT (Macro Name Table)")
print("Index\tName\tMDT Index")
for i, entry in enumerate(mnt):
    print(i, "\t", entry["name"], "\t", entry["mdt_index"])

print("\nMDT (Macro Definition Table)")
for i, line in enumerate(mdt):
    print(i, "\t", line)
