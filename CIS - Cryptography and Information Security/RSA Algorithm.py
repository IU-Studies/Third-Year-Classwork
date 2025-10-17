def is_prime(n):
    if n <= 1: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def modinv(a, m):
   
    m0, x0, x1 = m, 0, 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        a, m = m, a % m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

p = int(input("Enter prime p: "))
q = int(input("Enter prime q: "))

if p == q or not (is_prime(p) and is_prime(q)):
    print("Error: p and q must be different primes")
    exit()

n = p * q
phi = (p - 1) * (q - 1)

coprimes = [e for e in range(2, phi) if gcd(e, phi) == 1]

print(f"\nn = {n}, phi = {phi}")
print(f"Possible values for e (public exponent):")
print(coprimes)

e = int(input("\nSelect an e from the list above: "))
if e not in coprimes:
    print("Error: Selected e is not valid.")
    exit()

d = modinv(e, phi)
print(f"Private key exponent d calculated as: {d}")

m = int(input("\nEnter message m (integer) such that m < n: "))
if m >= n:
    print("Error: Message m must be less than n.")
    exit()

c = pow(m, e, n)
print(f"Encrypted message: {c}")

m_decrypted = pow(c, d, n)
print(f"Decrypted message: {m_decrypted}")
