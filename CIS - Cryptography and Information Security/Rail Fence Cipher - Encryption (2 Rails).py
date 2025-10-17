# Rail Fence Cipher - Encryption (2 Rails)
# -------------------------------------------------------
# In Rail Fence Cipher, we write letters in a zig-zag
# pattern across multiple "rails" (rows). For 2 rails:
#
# Example (HELLO):
# H   L   O
#  E L
# Cipher = "HLOEL"
# -------------------------------------------------------

# Input string
a = input("Enter String To Convert Into Coded Text: ")
a = list(a)   # convert string into a list of characters

# Two rails (rows)
b = []   # will store letters at even positions
c = []   # will store letters at odd positions

# Distribute characters alternatively into two rails
for i in range(len(a)):
    if i % 2 == 0:     # even index → goes to rail 1
        b.append(a[i])
    else:              # odd index → goes to rail 2
        c.append(a[i])

# Join the two rails together (rail 1 + rail 2)
x = "".join(b + c)

# Output encrypted text
print("Encrypted Text:", x)
