\
import time
import numpy as np
from primefft.fft import fft

def bench(N):
    x = np.random.randn(N) + 1j*np.random.randn(N)
    t0 = time.time()
    X_np = np.fft.fft(x)
    t1 = time.time()
    X_pf = fft(x)
    t2 = time.time()
    err = np.max(np.abs(X_np - X_pf))
    return (t1-t0, t2-t1, err)

def main():
    Ns = [480, 500, 512, 1000, 1001, 1009, 1500, 2048]
    print(f"{'N':>6} | {'numpy(s)':>10} | {'primefft(s)':>12} | {'max|err|':>10}")
    print("-"*46)
    for N in Ns:
        tn, tp, err = bench(N)
        print(f"{N:6d} | {tn:10.6f} | {tp:12.6f} | {err:10.2e}")

if __name__ == '__main__':
    main()
