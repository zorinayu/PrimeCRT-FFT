\
import numpy as np
from .cooleytukey import fft_cooleytukey
from .goodthomas import fft_good_thomas_possible
from .bluestein import fft_bluestein
from .planner import factorize, choose_plan

def fft(x: np.ndarray, method: str = 'auto') -> np.ndarray:
    """Compute the length-N DFT of x.

    method:
      - 'auto': use planner to pick an algorithm
      - 'numpy': delegate to numpy.fft.fft
      - 'cooleytukey': Cooley-Tukey path (currently numpy-backed for correctness)
      - 'good-thomas': Good-Thomas (requires gcd(a,b)=1 factorization)
      - 'bluestein': Bluestein's algorithm
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if method == 'numpy':
        return np.fft.fft(x)
    if method == 'bluestein':
        return fft_bluestein(x)
    if method == 'cooleytukey':
        return fft_cooleytukey(x)
    if method == 'good-thomas':
        # Attempt to split N into coprime a, b via planner
        factors = factorize(N)
        plan = choose_plan(factors)
        if plan.get('kind') == 'good-thomas':
            return fft_good_thomas_possible(x, plan['a'], plan['b'])
        # Fall back to numpy if not applicable
        return np.fft.fft(x)
    # auto mode
    factors = factorize(N)
    plan = choose_plan(factors)
    kind = plan['kind']
    if kind == 'good-thomas':
        return fft_good_thomas_possible(x, plan['a'], plan['b'])
    elif kind == 'cooley-tukey':
        return fft_cooleytukey(x)
    else:
        return fft_bluestein(x)
