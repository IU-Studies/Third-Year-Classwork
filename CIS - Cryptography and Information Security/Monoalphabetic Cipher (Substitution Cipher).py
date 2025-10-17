# Monoalphabetic Cipher (Substitution Cipher)
# ------------------------------------------------
# In this cipher, each letter of the alphabet is replaced
# by another fixed letter from a "cipher alphabet".
# Example: A -> O, B -> W, C -> E, etc.
# ------------------------------------------------

# Plain and cipher alphabets
plain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"           # Normal alphabet (A to Z)
cipher = "OWERTYUIOPASDFGHJKLZXCVBNM"          # Substitution alphabet (shuffled)

# -------- Encryption --------
text = input("Enter text to encrypt (CAPITAL letters only): ")
encrypted = ""  # Empty string to store encrypted result

for ch in text:  
    if ch in plain:
        # Find position of character in plain alphabet
        index = plain.index(ch)   
        # Replace with corresponding letter from cipher alphabet
        encrypted += cipher[index]
    else:
        # Keep spaces, numbers, or symbols unchanged
        encrypted += ch

print("Encrypted Text:", encrypted)

# -------- Decryption --------
text = input("\nEnter text to decrypt (CAPITAL letters only): ")
decrypted = ""  # Empty string to store decrypted result

for ch in text:  
    if ch in cipher:
        # Find position of character in cipher alphabet
        index = cipher.index(ch)
        # Replace with corresponding letter from plain alphabet
        decrypted += plain[index]
    else:
        # Keep spaces, numbers, or symbols unchanged
        decrypted += ch

print("Decrypted Text:", decrypted)
