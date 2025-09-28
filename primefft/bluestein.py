\
import numpy as np

def fft_bluestein(x: np.ndarray) -> np.ndarray:
    """Bluestein's algorithm converting DFT to convolution."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    n = np.arange(N)
    # Pre-/post- chirp with w_n = exp(-i*pi*n^2/N)
    w = np.exp(-1j * np.pi * (n**2) / N)
    a = x * w
    # Build symmetric chirp sequence b_m = exp(+i*pi*m^2/N) for m in Z
    M = 1
    L = 2 * N - 1
    while M < L:
        M <<= 1
    chirp = np.exp(1j * np.pi * (np.arange(N)**2) / N)
    b = np.zeros(M, dtype=np.complex128)
    b[:N] = chirp
    if N > 1:
        b[-(N-1):] = chirp[1:][::-1]
    a_pad = np.zeros(M, dtype=np.complex128)
    a_pad[:N] = a
    A = np.fft.fft(a_pad, n=M)
    B = np.fft.fft(b, n=M)
    C = A * B
    c = np.fft.ifft(C, n=M)[:N]
    X = c * w
    return X
