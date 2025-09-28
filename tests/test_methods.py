import numpy as np
import pytest

from primefft.fft import fft


@pytest.mark.parametrize('N', [2,3,4,5,6,7,8,9,10,12,15,16,20,25,27,30,32,45])
@pytest.mark.parametrize('method', ['auto','cooleytukey','bluestein'])
def test_methods_match_numpy(N, method):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    X_np = np.fft.fft(x)
    X_pf = fft(x, method=method)
    assert np.allclose(X_np, X_pf, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize('N', [6,10,15,30])
@pytest.mark.xfail(reason='Good-Thomas path not yet numerically identical in current version')
def test_good_thomas_xfail(N):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    X_np = np.fft.fft(x)
    from primefft.fft import fft
    X_pf = fft(x, method='good-thomas')
    assert np.allclose(X_np, X_pf, atol=1e-8, rtol=1e-7)


