\
import numpy as np
from primefft.fft import fft

def test_correctness_small():
    for N in [2,3,4,5,6,7,8,9,10,12,15,16,20,25]:
        x = np.random.randn(N) + 1j*np.random.randn(N)
        X_np = np.fft.fft(x)
        X_pf = fft(x)
        assert np.allclose(X_np, X_pf, atol=1e-8, rtol=1e-7), f"Mismatch at N={N}"
