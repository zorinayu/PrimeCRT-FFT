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
    # prime
    if len(factors) == 1 and list(factors.values())[0] == 1:
        return {'kind': 'bluestein', 'N': N}
    # try coprime split
    primes = []
    for p, a in factors.items():
        primes += [p] * a
    a = 1
    for p in primes[::2]:
        a *= p
    b = N // a
    from math import gcd as _g
    if a > 1 and b > 1 and _g(a, b) == 1:
        return {'kind': 'good-thomas', 'N': N, 'a': a, 'b': b}
    return {'kind': 'cooley-tukey', 'N': N}
