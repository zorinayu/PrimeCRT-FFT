\
import numpy as np

def fft_cooleytukey(x: np.ndarray) -> np.ndarray:
    \"\"\"Recursive Cooley-Tukey FFT that handles any composite N.\"\"\"
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N == 1:
        return x.astype(np.complex128)
    # find a non-trivial factor
    a = None
    for f in range(2, int(np.sqrt(N)) + 1):
        if N % f == 0:
            a = f
            break
    if a is None:
        # N is prime; use direct DFT (small) to avoid circular dependency
        n = np.arange(N)
        k = n[:, None]
        W = np.exp(-2j * np.pi * k * n / N)
        return W @ x
    b = N // a
    # split
    x_mat = x.reshape(b, a).T.copy()  # shape (a, b)
    # FFT along rows (a) recursively
    X_rows = np.vstack([fft_cooleytukey(x_mat[i, :]) for i in range(a)])
    # twiddle factors
    n1 = np.arange(a)[:, None]
    k2 = np.arange(b)[None, :]
    twiddle = np.exp(-2j * np.pi * (n1 * k2) / (a * b))
    X_rows *= twiddle
    # FFT along columns (b) recursively
    X = np.hstack([fft_cooleytukey(X_rows[:, j])[:, None] for j in range(b)])
    # reorder to 1D: k = k1 + b*k2
    out = np.empty(N, dtype=np.complex128)
    idx = 0
    for kk2 in range(b):
        for kk1 in range(a):
            out[idx] = X[kk1, kk2]
            idx += 1
    return out
