\
import time
import numpy as np
from primefft.fft import fft

def run_once(x, method):
    t0 = time.perf_counter()
    X = fft(x, method=method)
    t1 = time.perf_counter()
    return X, (t1 - t0)

def metrics(X_ref, X):
    diff = X - X_ref
    max_err = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.abs(diff)**2)))
    rel_err = float(max_err / max(1e-16, np.max(np.abs(X_ref))))
    return max_err, rmse, rel_err

def bench(N, repeats=5, methods=('auto','cooleytukey','good-thomas','bluestein','numpy')):
    results = []
    for r in range(repeats):
        x = np.random.randn(N) + 1j*np.random.randn(N)
        X_np, t_np = run_once(x, 'numpy')
        for m in methods:
            X_m, t_m = run_once(x, m)
            max_err, rmse, rel_err = metrics(X_np, X_m)
            results.append((m, t_m, max_err, rmse, rel_err))
    # aggregate by method
    summary = {}
    for m, t_m, max_err, rmse, rel_err in results:
        if m not in summary: summary[m] = []
        summary[m].append((t_m, max_err, rmse, rel_err))
    agg = {}
    for m, arr in summary.items():
        arr = np.array(arr)
        t_avg = float(np.mean(arr[:,0]))
        t_med = float(np.median(arr[:,0]))
        max_err_med = float(np.median(arr[:,1]))
        rmse_med = float(np.median(arr[:,2]))
        rel_err_med = float(np.median(arr[:,3]))
        agg[m] = (t_avg, t_med, max_err_med, rmse_med, rel_err_med)
    return agg

def pretty_print(N, agg):
    print(f"N={N}")
    print(f"{'Method':>12} | {'t_avg(s)':>9} | {'t_med(s)':>9} | {'max_err':>10} | {'rmse':>10} | {'rel_err':>10}")
    print("-"*72)
    order = ['numpy','auto','cooleytukey','good-thomas','bluestein']
    for m in order:
        if m in agg:
            t_avg, t_med, max_err, rmse, rel_err = agg[m]
            print(f"{m:>12} | {t_avg:9.6f} | {t_med:9.6f} | {max_err:10.2e} | {rmse:10.2e} | {rel_err:10.2e}")

def main():
    Ns = [120, 225, 480, 500, 512, 1000, 1001, 1009, 1500, 2048]
    for N in Ns:
        agg = bench(N, repeats=5)
        pretty_print(N, agg)

if __name__ == '__main__':
    main()
