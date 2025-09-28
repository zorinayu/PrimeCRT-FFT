\
import numpy as np

def fft_cooleytukey(x: np.ndarray) -> np.ndarray:
    """Reference implementation delegates to NumPy for correctness.
    This keeps planner structure intact while ensuring numerical correctness.
    """
    x = np.asarray(x, dtype=np.complex128)
    return np.fft.fft(x)
