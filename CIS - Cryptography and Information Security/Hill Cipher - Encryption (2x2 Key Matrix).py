# Hill Cipher - Encryption (2x2 Key Matrix)
# -------------------------------------------------------
# This implementation encrypts messages two letters at a time
# using a 2x2 key matrix. Letters map to numbers: A=0, B=1, ..., Z=25.
# For each pair P = [p1, p2]^T (numbers), ciphertext C = K * P (mod 26).
# -------------------------------------------------------

# Alphabet list for indexing letters
# We use a list so we can convert between letter <-> numeric index easily.
# Index: 0->'A', 1->'B', ..., 25->'Z'
x = ["A","B","C","D","E","F","G","H","I","J","K","L","M",
     "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# 2x2 Key matrix
# This is the matrix K used to transform plaintext vectors.
# K must be invertible modulo 26 for decryption to be possible.
# Here K = [[3,3],[2,5]]. Its determinant = 3*5 - 3*2 = 15 - 6 = 9.
# gcd(det, 26) = gcd(9,26) = 1 → invertible mod 26 (good).
key = [[3, 3],
       [2, 5]]

# Take input from user (only uppercase letters)
# IMPORTANT: This code expects uppercase A-Z only. If you want to accept lower-case,
# you could convert with message = message.upper() (uncomment below).
message = input("Enter message (only uppercase letters): ")
# message = message.upper()  # <-- optional: uncomment to auto-convert lowercase

# If message length is odd, add 'X' as padding
# Hill cipher operates on fixed-size blocks (here size = 2), so the length must be even.
# 'X' is commonly used as padding letter; you can choose another if you prefer.
if len(message) % 2 != 0:
    message += "X"

cipher = ""   # Empty string to store final encrypted message

# Process message in pairs of 2 letters
# Loop steps: 0..len-1 stepping by 2, so i points to first char of each pair.
for i in range(0, len(message), 2):
    # Convert letters to numbers using their index in alphabet list
    # For example, 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
    a = x.index(message[i])     # numeric value of first letter in the pair
    b = x.index(message[i+1])   # numeric value of second letter in the pair

    # Perform matrix multiplication with the key:
    # If K = [[k11, k12],[k21,k22]] and P = [a, b]^T, then
    # C1 = (k11*a + k12*b) mod 26
    # C2 = (k21*a + k22*b) mod 26
    # We take mod 26 to wrap numbers back into the range 0..25 which correspond to A..Z.
    c1 = (key[0][0]*a + key[0][1]*b) % 26
    c2 = (key[1][0]*a + key[1][1]*b) % 26

    # Convert numbers back to letters and add to cipher
    # x[c1] gives the letter corresponding to number c1 (0->'A', ..., 25->'Z')
    cipher += x[c1] + x[c2]

# Print results
# Show the (possibly padded) message and the resulting encrypted text.
print("Message   :", message)
print("Encrypted :", cipher)

# ----------------------- Extra notes (not executed) -----------------------
# 1) Why modulo 26?  Because there are 26 letters; after multiplying we reduce to this range.
# 2) Example (manual small calculation):
#    Suppose the first pair is "HI":
#      H -> 7, I -> 8  (because A=0)
#      c1 = (3*7 + 3*8) % 26 = (21 + 24) % 26 = 45 % 26 = 19  -> 'T'
#      c2 = (2*7 + 5*8) % 26 = (14 + 40) % 26 = 54 % 26 = 2   -> 'C'
#    So "HI" encrypts to "TC".
# 3) Decryption: to decrypt you need the inverse of the key matrix modulo 26.
#    That requires det(K) to be coprime with 26 (i.e., gcd(det,26)=1). Our key's det=9,
#    gcd(9,26)=1, so an inverse exists and decryption is possible.
# 4) Security note: Hill cipher is a classical cipher and is not secure for modern use,
#    but it is a great learning tool for linear algebra + modular arithmetic.
# ------------------------------------------------------------------------
