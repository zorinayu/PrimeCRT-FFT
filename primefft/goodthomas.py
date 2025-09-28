\
import numpy as np
from math import gcd

def modinv(a, m):
    # modular inverse via extended Euclid
    a %= m
    if gcd(a, m) != 1:
        raise ValueError("No inverse")
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0:
        q = r // newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    if t < 0:
        t += m
    return t

def _crt_permutation_indices(a, b):
    """Return arrays map_n[(n1,n2)] -> n and map_k[(k1,k2)] -> k using CRT.
    Uses Garner form with m1 = b * (b^{-1} mod a), m2 = a * (a^{-1} mod b).
    """
    N = a * b
    ainv_mod_b = modinv(a, b)
    binv_mod_a = modinv(b, a)
    m1 = (b * binv_mod_a) % N  # congruent to 1 mod a, 0 mod b
    m2 = (a * ainv_mod_b) % N  # congruent to 0 mod a, 1 mod b
    map_n = np.empty((a, b), dtype=int)
    for n1 in range(a):
        for n2 in range(b):
            n = (n1 * m1 + n2 * m2) % N
            map_n[n1, n2] = n
    map_k = np.empty((a, b), dtype=int)
    for k1 in range(a):
        for k2 in range(b):
            k = (k1 * m1 + k2 * m2) % N
            map_k[k1, k2] = k
    return map_n, map_k

def fft_good_thomas_possible(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """Good-Thomas FFT when gcd(a,b)=1 and len(x)=a*b.
    Performs two 1D DFTs (size a and size b) without twiddles, plus permutations.
    """
    N = x.shape[0]
    assert N == a * b, "Length mismatch"
    if gcd(a, b) != 1:
        raise ValueError("a and b must be coprime")
    x = np.asarray(x, dtype=np.complex128)
    map_n, map_k = _crt_permutation_indices(a, b)
    # permute input into (a,b) grid using map_n
    Xgrid = np.empty((a, b), dtype=np.complex128)
    for n1 in range(a):
        for n2 in range(b):
            n = map_n[n1, n2]
            Xgrid[n1, n2] = x[n]
    # DFT along axis 0 (size a)
    n1 = np.arange(a)
    k1 = n1[:, None]
    Wa = np.exp(-2j * np.pi * k1 * n1 / a)  # F_a
    Y = Wa @ Xgrid
    # DFT along axis 1 (size b)
    n2 = np.arange(b)
    k2 = n2[:, None]
    Wb = np.exp(-2j * np.pi * k2 * n2 / b)  # F_b
    Z = Y @ Wb.T
    # unpermute outputs using map_k
    out = np.empty(N, dtype=np.complex128)
    for kk1 in range(a):
        for kk2 in range(b):
            k = map_k[kk1, kk2]
            out[k] = Z[kk1, kk2]
    return out
