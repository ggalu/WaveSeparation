"""
FROZEN REFERENCE -- do not use for real work, and do not refactor.

A deliberately literal transcription of `wave_separation3` from
Prog_Treat/a05.m: three gauges only, MATLAB's argument order, no input
validation, n_point and f_max passed in by hand. The one intentional
departure from the original is the frequency axis, which MATLAB computes as
f_max/(n_point-1) where the true FFT bin spacing is f_max/n_point.

Use wave_separation.py instead. That module generalises to any number of
gauges, validates its inputs, and carries the result through to specimen
stress/strain.

This file is kept so that sep_test.py's check on wave_separation.py stays
INDEPENDENT. Merging the two would make the check circular -- it would show
only that the code agrees with itself. Because this version was written
straight from the MATLAB and never refactored alongside the library, the
agreement between them is evidence rather than tautology.

Measured agreement: 7.1e-07 relative. The residual is a single FFT bin --
the code below reproduces MATLAB's ifft(..., 'symmetric') on a zero-padded
half spectrum, which forces the Nyquist bin to zero, while
wave_separation.py uses rfft/irfft and computes it.

Known limitations, present on purpose because MATLAB has them too:
  - eta = 0 divides by a zero determinant and returns all-NaN behind a
    RuntimeWarning, rather than raising;
  - the exponential window uses raw t rather than t - t[0], so a record whose
    time base starts far from zero underflows (NaN at t + 2000 ms, eta = 1).
"""

import numpy as np

def wave_separation(t, ea, eb, ec, a, b, c, C0, eta, n_point, f_max, cp_curve=None):
    """Transcription of wave_separation3 (freq-axis bug fixed)."""
    N = int(n_point); half = N//2
    win = np.exp(-eta*np.asarray(t))
    Ea, Eb, Ec = (np.fft.fft(e*win, N)[:half] for e in (ea, eb, ec))
    f = np.arange(half)*(f_max/N)
    w = 2*np.pi*f
    cp = np.full(half, C0) if cp_curve is None else np.interp(f, *cp_curve)*C0
    xi = (w - 1j*eta)/cp; xb = np.conj(xi)
    x = (a, b, c); Es = (Ea, Eb, Ec)
    h1 = sum(np.exp(-1j*(xi-xb)*d) for d in x)
    h2 = sum(np.exp(+1j*(xi-xb)*d) for d in x)
    g  = sum(np.exp(+1j*(xi+xb)*d) for d in x)
    E1 = sum(E*np.exp(+1j*xb*d) for E, d in zip(Es, x))
    E2 = sum(E*np.exp(-1j*xb*d) for E, d in zip(Es, x))
    det = h1*h2 - g*np.conj(g)
    A = (h2*E1 - g*E2)/det
    B = (-np.conj(g)*E1 + h1*E2)/det
    return [np.fft.irfft(np.concatenate([X, [0.0]]), n=N)[:len(t)]*np.exp(eta*np.asarray(t))
            for X in (A, B)]
