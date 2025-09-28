\
from math import gcd
from collections import Counter

def factorize(N: int):
    n = N
    fac = Counter()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fac[d] += 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        fac[n] += 1
    return fac

def choose_plan(factors: Counter):
    N = 1
    for p, a in factors.items():
        N *= p ** a
    # For correctness, route all sizes to cooley-tukey (NumPy backend)
    return {'kind': 'cooley-tukey', 'N': N}
