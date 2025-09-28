\
import numpy as np
from .cooleytukey import fft_cooleytukey
from .goodthomas import fft_good_thomas_possible
from .bluestein import fft_bluestein
from .planner import factorize, choose_plan

def fft(x: np.ndarray) -> np.ndarray:
    """Compute the length-N DFT of x using a factorization-aware planner."""
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    factors = factorize(N)
    plan = choose_plan(factors)
    kind = plan['kind']
    if kind == 'good-thomas':
        a, b = plan['a'], plan['b']
        return fft_good_thomas_possible(x, a, b)
    elif kind == 'cooley-tukey':
        return fft_cooleytukey(x)
    else:
        return fft_bluestein(x)
