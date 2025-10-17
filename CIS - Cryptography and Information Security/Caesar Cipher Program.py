# Caesar Cipher Program
# --------------------------------------------------
# This is a simple substitution cipher where each
# letter is shifted by 3 positions in the alphabet.
# Example: a -> d, b -> e, ..., x -> a, y -> b, z -> c
# --------------------------------------------------

# ----------------- Encryption -----------------
a = input("Enter String To Convert Into Caesar Cipher (use lowercase): ")

# List of alphabets a-z
b = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
     'n','o','p','q','r','s','t','u','v','w','x','y','z']

c = ""  # empty string to build encrypted text

for i in a:  
    if i in b:  
        # find position of the letter in alphabet
        # then shift it 3 steps forward using (index + 3)
        # % 26 makes sure it "wraps around" after 'z'
        d = (b.index(i) + 3) % 26
        c += b[d]   # add shifted letter
    else:
        # if it's not a letter (space, number, symbol)
        # keep it exactly the same
        c += i

print("Encrypted Text:", c)

# ----------------- Decryption -----------------
a = input("\nEnter String To Decrypt Caesar Cipher (use lowercase): ")

# same alphabet list
b = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
     'n','o','p','q','r','s','t','u','v','w','x','y','z']

c = ""  # empty string to build decrypted text

for i in a:
    if i in b:
        # find position of the letter
        # then shift it 3 steps backward (index - 3)
        # again use % 26 to wrap around (so 'a' goes back to 'x')
        d = (b.index(i) - 3) % 26
        c += b[d]
    else:
        # keep spaces/symbols unchanged
        c += i

print("Decrypted Text:", c)
