"""
Identify a bar's ATTENUATION alpha(f) from two gauges on it, using no boundary
condition and no ground truth.

    from identify_attenuation import fit_attenuation
    att = fit_attenuation(t, signals, positions, f1, f2)
    P, M = separate(t, signals, positions, c0=c, eta=eta,
                    attenuation=att['table'])

A module, not a script: identify_bar_compression.py and
reconstruct_interface.py both want this and neither should own it. It needs the
edge times the identification already measured, so it runs after that and
nothing here re-derives them.

--------------------------------------------------------------------------
Why a metal rig never needed this, and a polymer one does
--------------------------------------------------------------------------
`separate` fits ONE pair of waves to ALL the gauges at once. That works while
every gauge sees the same wave shape, which in aluminium it does. In
polycarbonate it does not: the leading edge of the PC calibration shot broadens
from 20 us to 34 us over the 371 mm between its two gauges, and the plateau
loses 3.5 %. Handed those two records and told the bar is lossless, `separate`
cannot satisfy both, and the residual comes out as a free-surface null stuck at
9e-2 and an interface force that goes 12 % of peak TENSILE -- which a contact
that can only push cannot do.

--------------------------------------------------------------------------
How it is measured: a transfer function between two gauges
--------------------------------------------------------------------------
Over a window in which only ONE wave is present, gauge j and gauge k differ by
a known propagation distance dx and nothing else, so

    E_k(w) / E_j(w) = exp(-i w dx / c_p(f)) exp(-alpha(f) dx)

Magnitude gives alpha, phase gives c_p. Three details make this work on a real
record rather than in principle:

  * DIFFERENTIATE FIRST. The record is a long step, so its own spectrum is
    almost entirely DC and the ratio is noise everywhere else. Its derivative is
    the EDGE -- broadband, and starting and ending near zero, so the window
    truncates almost nothing. d/dt is a factor i*w on both sides and cancels
    out of the ratio exactly. This is the same reason the identification times
    edges rather than pulses.
  * DE-LAG TO THE SUB-SAMPLE. Each window is cut at its own arrival, which is
    an integer number of samples; the remainder is applied as a phase, so no
    rounding enters. Skipping this leaves a fraction of a sample of delay that
    the fit absorbs into alpha.
  * BAND THE ANSWER. Bin-by-bin the ratio is far too noisy to use directly --
    neighbouring bins land above and below unity. Averaging |ln H| over bands a
    few kHz wide, weighted by where the near gauge actually has energy, is what
    makes it a curve.

The result is returned as an (freq, alpha) TABLE ending at f_hi, which is what
band-limits it: np.interp holds that last value above f_hi, so the
de-attenuation exp(+alpha x) stops growing. That is not a nicety. Left to grow,
it amplifies high-frequency noise without limit -- taken to Nyquist it
overflows, and just short of that it produces a free-end null residual ~15x
better than the truth, built entirely on amplified noise.

--------------------------------------------------------------------------
What this does NOT do
--------------------------------------------------------------------------
It does not pick alpha by making the boundary conditions look good. That was
tried and it does not work: on the PC record the free-end null, the tensile
violation and the post-separation residual ALL improve monotonically as alpha is
raised, with no minimum, because more damping quietly suppresses everything.
They establish that alpha > 0 is needed -- 0.118 tensile violation against
0.029 -- and they cannot pin its value. So alpha is measured HERE, from
magnitudes alone, and the boundary conditions are left to validate it. Fitting
on one and checking on the other is the only way that check means anything.
"""

import numpy as np

__all__ = ['fit_attenuation']


def fit_attenuation(t, signals, positions, f1, f2, f_lo=2.0, f_hi=50.0,
                    band=4.0, snr=0.005, margin=0.03, monotone=True):
    """
    Attenuation alpha(f) and phase velocity c_p from two or more gauges.

    Parameters
    ----------
    t : (N,) array
        Uniform time base, as for `separate`.
    signals : sequence of (N,) arrays
        The gauge records. Strain, force or volts -- the ratio is scale-free in
        each channel separately, so a per-channel calibration constant cancels
        only if it is the same at both. It usually is; if it is not, that shows
        up as a constant offset in alpha at low frequency.
    positions : (n_gauge,) array
        Identified distances from the interface, same order as `signals`.
    f1, f2 : (n_gauge,) arrays
        Arrival and free-end-echo times at each gauge [same units as t], from
        the identification. `f2 - f1` is what bounds the single-wave window.
    f_lo, f_hi : float
        Band over which alpha is reported [1/time unit of t, i.e. kHz for ms].
        `f_hi` is also the BAND LIMIT of the returned table -- see the module
        docstring.
    band : float
        Width of the averaging bands. A few kHz: narrow enough to follow a real
        curve, wide enough that each band holds enough bins to average.
    snr : float
        Keep only bins where the near gauge carries at least this fraction of
        its own spectral peak.
    margin : float
        Clearance kept between the end of the window and the echo, in the units
        of t. The window must contain ONE wave; ending it exactly on the echo
        lets the first sample of the echo in.
    monotone : bool
        Constrain alpha(f) to be non-decreasing. A viscoelastic solid's is, over
        this range, and the constraint removes the odd band that lands low on
        noise without changing the fit measurably (0.1597 -> 0.1592 residual on
        the PC record).

    Returns
    -------
    dict
        table   : (freq, alpha) pair, ready for separate(attenuation=...)
        k       : the linear-law fit alpha ~ k*f, as a single-number summary
                  for comparison with literature. NOT what `table` contains.
        c_p     : phase velocity from the transfer-function phase [length/time].
                  An INDEPENDENT estimate of c0 -- it uses the gauge-to-gauge
                  phase, where the round trip 2L/c uses reflections.
        misfit  : relative L2 of the far gauge predicted from the near one,
                  with the table, over the band. Compare `misfit_lossless`.
        misfit_lossless : the same with alpha = 0. The gap is the evidence.
        n_window, pairs : how much record and how many gauge pairs were used.
    """
    t = np.asarray(t, float)
    sig = [np.asarray(s, float) for s in signals]
    x = np.asarray(positions, float)
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    if len(sig) < 2:
        raise ValueError('need at least two gauges to form a transfer function')
    if not (len(sig) == len(x) == len(f1) == len(f2)):
        raise ValueError('signals, positions, f1 and f2 must be the same length')

    dt = float(np.mean(np.diff(t)))
    span = float(np.nanmin(f2 - f1)) - margin
    if span <= 0:
        raise ValueError(
            f'no single-wave window: the shortest gauge-to-echo gap is '
            f'{float(np.nanmin(f2 - f1)):.4g} against a margin of {margin:.4g}. '
            'A gauge very close to the free end has no clean window at all.')
    n = int(span / dt)
    n_fft = 1 << int(np.ceil(np.log2(4 * n)))
    f = np.fft.rfftfreq(n_fft, dt)

    E = [_edge_spectrum(s, f1[k], dt, n, n_fft, f) for k, s in enumerate(sig)]

    # Every ordered pair with a positive separation. Two gauges give one pair;
    # more give more, and each is an independent look at the same alpha.
    pairs = [(j, k) for j in range(len(sig)) for k in range(len(sig))
             if x[k] > x[j]]
    if not pairs:
        raise ValueError('gauge positions are not distinct')

    keep = (f >= f_lo) & (f <= f_hi)
    edges = np.arange(0.0, f_hi + band, band)
    fb, ab, wb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        num = den = 0.0
        for j, k in pairs:
            w = np.abs(E[j])
            m = (f >= lo) & (f < hi) & keep & (w > snr * w.max())
            if m.sum() < 3:
                continue
            # -ln|H| / dx, averaged over the band and weighted by where the
            # near gauge has energy. Bins with no signal carry no information
            # and must not be allowed to vote.
            ln = np.log(np.abs(E[k][m] / E[j][m]))
            num += float(np.sum(w[m] * -ln)) / (x[k] - x[j])
            den += float(np.sum(w[m]))
        if den > 0:
            fb.append(0.5 * (lo + hi))
            ab.append(max(num / den, 0.0))     # alpha < 0 would amplify: reject
            wb.append(den)
    if len(fb) < 2:
        raise ValueError('not enough usable bands; loosen snr or widen the band')

    fb, ab, wb = np.array(fb), np.array(ab), np.array(wb)
    if monotone:
        ab = np.maximum.accumulate(ab)
    # Anchor at DC: a bar does not attenuate a static load.
    if fb[0] > 0:
        fb, ab = np.concatenate(([0.0], fb)), np.concatenate(([0.0], ab))
    table = (fb, ab)

    k_lin = float(np.sum(wb * fb[-len(wb):] * ab[-len(wb):])
                  / np.sum(wb * fb[-len(wb):] ** 2))

    alpha_f = np.interp(f, fb, ab)
    j, k = pairs[0]
    misfit, tau = _misfit(E[j], E[k], f, x[k] - x[j], alpha_f, f_hi)
    lossless, _ = _misfit(E[j], E[k], f, x[k] - x[j], np.zeros_like(f), f_hi)
    lag = f1[k] - f1[j]
    c_p = (x[k] - x[j]) / (lag + tau) if (lag + tau) != 0 else np.nan

    return dict(table=table, k=k_lin, c_p=float(c_p), misfit=misfit,
                misfit_lossless=lossless, n_window=n, pairs=len(pairs),
                span=span, f_lo=f_lo, f_hi=f_hi, band=band, tau=float(tau))


def _edge_spectrum(s, t_arrive, dt, n, n_fft, f):
    """
    Spectrum of the differentiated single-wave window, de-lagged to sub-sample.

    The window is cut at the integer sample nearest the arrival and the
    remaining fraction of a sample is removed as a phase, so every gauge's
    spectrum is referred to its own arrival exactly. The linear de-trend
    removes the residual step between the two ends of the derivative window,
    which would otherwise leak across the whole band.
    """
    i0 = int(round(t_arrive / dt))
    if i0 < 0 or i0 + n > len(s):
        raise ValueError(f'single-wave window [{i0}, {i0+n}) runs off a record '
                         f'of {len(s)} samples')
    seg = np.gradient(s[i0:i0 + n], dt)
    ends = max(4, n // 40)
    seg = seg - np.linspace(seg[:ends].mean(), seg[-ends:].mean(), n)
    frac = t_arrive - i0 * dt
    return np.fft.rfft(seg, n_fft) * np.exp(2j * np.pi * f * frac)


def _misfit(E_near, E_far, f, dx, alpha_f, f_hi):
    """
    Predict the far gauge from the near one and report the relative L2.

    The residual delay tau is fitted along with it: both spectra are already
    referred to their own arrivals, so what is left is only what DISPERSION
    does, and it is small. Returning it gives c_p for free.
    """
    m = f <= f_hi
    decay = np.exp(-alpha_f[m] * dx)
    ref = float(np.sum(np.abs(E_far[m]) ** 2))
    best = (np.inf, 0.0)
    for tau in np.arange(-0.03, 0.03, 2e-4):
        pred = E_near[m] * np.exp(-2j * np.pi * f[m] * tau) * decay
        v = float(np.sum(np.abs(pred - E_far[m]) ** 2)) / ref
        if v < best[0]:
            best = (v, tau)
    return best
