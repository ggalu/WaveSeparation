"""
Multi-point wave separation for Hopkinson-bar signals, and the reduction from
separated waves to specimen stress / strain.

The separation follows the Laplace-domain least-squares method implemented in
Prog_Treat/a05.m (`wave_separation3`), generalised to an arbitrary number of
gauges and with the frequency-axis off-by-one of the MATLAB original fixed.

--------------------------------------------------------------------------
Coordinates and sign conventions
--------------------------------------------------------------------------
Each bar is treated in its own LOCAL coordinate x, measured from the
bar/specimen interface, positive going INTO the bar (away from the specimen).
The interface is x = 0 and is the plane everything is reconstructed at.

The strain at gauge k, a distance x_k from the interface, is modelled as the
superposition of two travelling waves

    E_k(w) = P(w) exp(-i xi x_k) + M(w) exp(+i xi x_k)

so that

    P  ("plus")   travels toward +x, i.e. AWAY from the specimen
    M  ("minus")  travels toward -x, i.e. TOWARD the specimen

In a classical SHPB input bar, M is the incident wave and P the reflected one.
In a direct-impact configuration the loading wave is generated AT the interface
and travels away from it, so on the input bar the loading wave is P, not M --
the roles are swapped. `bar_interface` takes the bar's orientation explicitly
rather than assuming either case.

Strain follows the usual solid-mechanics sign: COMPRESSION IS NEGATIVE. Force
is therefore also negative in compression. `specimen_response` flips this at the
very end, because stress/strain curves are conventionally plotted with
compression positive.

--------------------------------------------------------------------------
Regularisation
--------------------------------------------------------------------------
The separation is ill-posed at zero frequency and at every frequency where the
gauge spacings put all the phase factors back in step. Both are cured by the
exponential window exp(-eta t), which shifts the transform off the real
frequency axis. To leading order the system determinant behaves as

    det ~ (w^2 + eta^2) (2/c0)^2 sum_{j<k} (x_j - x_k)^2

so eta acts as a LOW-FREQUENCY FLOOR: content below roughly f = eta / 2pi is not
separated, only damped. eta = 0 divides by zero at DC.

eta is squeezed from both sides, and the two limits have very different costs:

  too small -- the determinant is tiny near DC, low-frequency noise is amplified
               enormously, and because strain is obtained by INTEGRATING the
               velocities that error accumulates without bound. On the
               direct-impact test case, dropping eta from 1.0 to 0.05 /ms left
               the reconstructed stress unchanged but blew the strain error up
               from 1.6e-3 to 3.9 -- a useless result.
  too large  -- real signal below eta / 2pi is damped away with the noise.

Force/stress is remarkably insensitive to the choice (it needs no integration);
strain is not. Pick eta from the strain, not the stress: a decade below the
event's fundamental frequency is a good start. For the direct-impact case here
(event ~0.8 ms, fundamental ~1.25 kHz) the optimum is eta ~ 1 /ms, i.e. a floor
of about 160 Hz.

eta carries units of 1/time and must match the units of `t`. With t in seconds
(as in a05.m) sensible values are a few hundred; with t in milliseconds (as in
simulate_compression.py) the same physical damping is a few tenths to a few units.

exp(+eta*t) is applied on the way out, so eta * t_max much above ~30 overflows.
"""

import numpy as np

__all__ = ['separate', 'separate_field', 'separate_time_domain',
           'separate_time_domain_field',
           'backpropagate', 'bar_interface', 'specimen_response',
           'conditioning', 'single_wave_window', 'wavefront_time']


def _curve(spec, f, scale=1.0):
    """
    Evaluate a None | callable | (x, y) table specification on the axis `f`.

    np.interp HOLDS THE ENDPOINT VALUES outside the table, and that is relied
    on: an attenuation identified over 2-50 kHz is then automatically flat above
    50 kHz rather than extrapolating, which is what band-limits the de-
    attenuation. See `attenuation` in `separate`.
    """
    if spec is None:
        return None
    if callable(spec):
        return np.asarray(spec(f), float) * scale
    xp, yp = np.asarray(spec[0], float), np.asarray(spec[1], float)
    return np.interp(f, xp, yp) * scale


def _wavenumber(f, c0, eta, dispersion, attenuation=None):
    """
    Complex wavenumber on a one-sided axis:

        xi = (w - i eta) / c_p(f)  -  i alpha(f)

    The first term is the elastic propagator plus the Laplace window; the second
    is material ATTENUATION, in 1/length. The plus wave carries

        exp(-i xi x) = exp(-i w x / c_p) exp(-eta x / c_p) exp(-alpha x)

    so it decays going away from the interface, and the minus wave carries
    exp(+i xi x), i.e. exp(+alpha x) -- correct, because a wave heading TOWARD
    the interface was larger further out. That growing branch is the ill-posed
    one; see the band-limit note under `attenuation` in `separate`.
    """
    w = 2.0 * np.pi * f
    cp = _curve(dispersion, f, c0)
    if cp is None:
        cp = np.full(f.shape, float(c0))
    xi = (w - 1j * eta) / cp
    a = _curve(attenuation, f)
    if a is not None:
        if np.any(a < 0):
            raise ValueError('attenuation must be >= 0; a negative alpha '
                             'amplifies the plus wave as it propagates')
        xi = xi - 1j * a
    return xi


def _pm_spectra(t, signals, positions, c0, eta, n_fft, dispersion,
                attenuation=None):
    """
    Validate the inputs and solve the normal equations, in the frequency domain.

    Returns the two wave SPECTRA at x = 0 -- still windowed, still complex --
    together with the axes needed to invert them. `separate` and
    `separate_field` are both thin wrappers around this, so the method exists
    once and only once.

    Returns
    -------
    P, M : (n_fft//2 + 1,) complex
        The plus and minus spectra at x = 0, in the exp(-eta t) domain.
    xi : (n_fft//2 + 1,) complex
        The wavenumber axis. Propagating to a station x is multiplication by
        exp(-i xi x) for P and exp(+i xi x) for M.
    n_fft : int
        The resolved transform length.
    tau : (n,) float
        t - t[0], so that exp(+eta tau) can undo the window.
    """
    t = np.asarray(t, float)
    sig = [np.asarray(s, float) for s in signals]
    x = np.asarray(positions, float)

    if eta <= 0:
        raise ValueError('eta must be > 0: the system is singular at DC for eta = 0')
    if len(sig) != len(x):
        raise ValueError(f'{len(sig)} signals but {len(x)} positions')
    if len(sig) < 2:
        raise ValueError('need at least two gauges to separate two waves')
    if np.any(x <= 0):
        raise ValueError('gauge positions must be > 0 (distance from the interface)')
    if len(set(np.round(x, 12))) != len(x):
        raise ValueError('gauge positions must be distinct')
    n = len(t)
    if any(s.shape != (n,) for s in sig):
        raise ValueError('all signals must have the same length as t')

    dt = float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6):
        raise ValueError('t must be uniformly sampled')
    if eta * (t[-1] - t[0]) > 700:
        raise ValueError(f'eta * record length = {eta*(t[-1]-t[0]):.1f}; '
                         'exp(+eta t) will overflow. Reduce eta.')

    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(4 * n)))
    n_fft = int(n_fft)
    if n_fft < n:
        raise ValueError('n_fft must be at least len(t)')

    # forward: exponential window, then one-sided transform
    tau = t - t[0]
    win = np.exp(-eta * tau)
    E = [np.fft.rfft(s * win, n_fft) for s in sig]
    f = np.fft.rfftfreq(n_fft, dt)
    xi = _wavenumber(f, c0, eta, dispersion, attenuation)
    xc = np.conj(xi)

    # exp(+alpha x) on the minus branch is the de-attenuation, and it is the
    # ill-posed half of the model: it grows without bound with frequency. The
    # same ceiling that bounds eta bounds it. A table-form `attenuation` is its
    # own band limit (np.interp holds the endpoints); a callable is not, and
    # this is what catches one that was never rolled off.
    # -xi.imag IS the exponent's coefficient: eta/c_p plus alpha, together.
    a_max = float(np.max(np.maximum(-xi.imag, 0.0))) * float(np.max(x))
    if a_max > 700:
        raise ValueError(
            f'max(alpha) * max(position) = {a_max:.1f}; '
            'exp(+alpha x) will overflow. Band-limit the attenuation -- an '
            '(freq, alpha) table holds its endpoint value beyond the table and '
            'is the easy way to do that.')

    # least-squares normal equations, summed over gauges
    #   [h1  g ] [P]   [E1]
    #   [g* h2 ] [M] = [E2]
    h1 = sum(np.exp(-1j * (xi - xc) * d) for d in x)     # real, > 0
    h2 = sum(np.exp(+1j * (xi - xc) * d) for d in x)     # real, > 0
    g = sum(np.exp(+1j * (xi + xc) * d) for d in x)
    E1 = sum(Ek * np.exp(+1j * xc * d) for Ek, d in zip(E, x))
    E2 = sum(Ek * np.exp(-1j * xc * d) for Ek, d in zip(E, x))

    det = (h1 * h2 - g * np.conj(g)).real                # real and >= 0
    if np.any(det <= 0):
        raise FloatingPointError('non-positive determinant; eta is too small '
                                 'or the gauge positions are degenerate')

    P = (h2 * E1 - g * E2) / det
    M = (-np.conj(g) * E1 + h1 * E2) / det
    return P, M, xi, n_fft, tau


def separate(t, signals, positions, c0, eta, n_fft=None, dispersion=None,
             attenuation=None):
    """
    Separate measured strain histories into the two travelling waves at x = 0.

    Parameters
    ----------
    t : (N,) array
        Uniformly sampled time. Must start at the beginning of the record and
        the signals must be quiescent at t[0].
    signals : sequence of (N,) arrays
        Strain (or any quantity proportional to it -- force, volts) at each
        gauge. Two or more; three is the usual choice.
    positions : sequence of float
        Distance of each gauge from the interface, same order as `signals`,
        in the same length unit as `c0 * t`. All must be > 0.
    c0 : float
        Elastic bar wave speed.
    eta : float
        Exponential-window / Laplace damping, units 1/time. Must be > 0.
    n_fft : int, optional
        Transform length. Defaults to the next power of two above 4*N, which
        zero-pads enough to keep the wrap-around out of the record.
    dispersion : None | callable | (freq, cp_over_c0)
        Phase-velocity dispersion. None means non-dispersive (c_p = c0), which
        is the right choice for 1D simulated data. A callable or a lookup table
        returns c_p / c0 as a function of frequency -- e.g. the Pochhammer-Chree
        curve in Results_Raw/pochhammer.mat.
    attenuation : None | callable | (freq, alpha)
        Material attenuation alpha(f), in 1/length -- the same units as
        1/positions. None means a lossless bar, which is right for metal and for
        simulated data. A real POLYMER bar is not lossless: polycarbonate
        measures alpha ~ 5.7e-5 * f [1/mm, f in kHz], a linear law, i.e. a
        constant loss angle. Without it the two gauges cannot be fitted at once
        and the residual shows up as a free-surface null that will not go below
        ~9e-2 and an interface force that goes tensile where a contact cannot
        pull.

        PREFER THE TABLE FORM. `np.interp` holds the endpoint values outside the
        table, so a table identified over 2-50 kHz is flat above 50 kHz instead
        of extrapolating -- and that flat top is the BAND LIMIT, which is not
        optional. The minus branch carries exp(+alpha x), so an unbounded
        alpha(f) amplifies high-frequency noise without limit; taken to Nyquist
        it overflows outright, and just short of that it produces a null
        residual that looks 15x better than the truth. `_pm_spectra` raises
        rather than overflow, but the quiet case is the dangerous one.

    Returns
    -------
    eps_plus, eps_minus : (N,) arrays
        Strain histories at x = 0 of the wave travelling toward +x and toward
        -x respectively. Their sum is the total strain at the interface.

    Notes
    -----
    The ordering of the gauges is irrelevant: every term in the normal
    equations is a symmetric sum over gauges. Permuting `signals` and
    `positions` together cannot change the result.
    """
    P, M, xi, n_fft, tau = _pm_spectra(t, signals, positions, c0, eta,
                                       n_fft, dispersion, attenuation)
    # inverse: back to time, then undo the window
    n = len(tau)
    amp = np.exp(+eta * tau)
    return (np.fft.irfft(P, n=n_fft)[:n] * amp,
            np.fft.irfft(M, n=n_fft)[:n] * amp)


def separate_field(t, signals, positions, c0, eta, x, n_fft=None,
                   dispersion=None, decimate=1, chunk=64, attenuation=None):
    """
    The two separated waves as a FIELD: reconstructed at many x, not just x = 0.

    The same solve as `separate` -- literally the same private core -- evaluated
    at every station in `x`. This is what a Lagrange (x-t) diagram consumes.

    The propagation is applied in the FREQUENCY domain, to P(w) and M(w), and
    that is the only correct way to do it. Taking `separate`'s TIME-domain
    output and re-transforming it is LOSSY: `separate` ends with
    `irfft(X, n_fft)[:n]`, discarding n_fft - n samples whose content is not
    zero, so the re-transform is a different signal. Measured on the shipped
    tension dump, reproducing the recorded gauge strains:

        this route, spectra kept          9.3e-15
        re-FFT of separate()'s output     1.4e-01     <-- silently wrong

    `backpropagate` is no help either: it re-transforms on every call, and it
    rejects `position <= 0`, which excludes the interface plane itself.

    Parameters
    ----------
    t, signals, positions, c0, eta, n_fft, dispersion, attenuation
        Exactly as for `separate`. With `attenuation` the field is no longer a
        pure shear: |eps_plus| decays as exp(-alpha x) along the bar instead of
        being constant, which is the physically right picture for a polymer bar
        and worth knowing before reading a Lagrange diagram of one.
    x : (n_x,) array
        Where to reconstruct, in this bar's LOCAL coordinate: 0 is the
        interface, positive goes INTO the bar, away from the specimen. Unlike
        `positions` these are unrestricted -- 0 and negative values are fine,
        because propagating is multiplication by a phase, not division by one.
        Reconstructing outside the uniform bar is EXTRAPOLATION; this function
        cannot know where the bar ends, so masking that is the caller's job.
    decimate : int
        Block-mean this many consecutive samples into one output row. The
        record is heavily oversampled -- the wavefront here is ~380 samples
        wide -- so 16 costs 0.5 % of the peak and saves 16x the memory.
    chunk : int
        Stations synthesised per FFT batch. The intermediate is
        (chunk, n_fft//2+1) complex plus (chunk, n_fft) real, so this bounds
        peak memory: measured 208 MB at 64 against 913 MB for 400 in one go.

    Returns
    -------
    eps_plus, eps_minus : (n_x, n_out) arrays
    t_out : (n_out,) array
        Block-centre times; identical to `t` when decimate == 1.

    Notes
    -----
    With `dispersion=None` this is an EXACT time shift and nothing more:
    xi = (w - i eta)/c0, so exp(-i xi x) = exp(-i w x/c0) exp(-eta x/c0) and
    the eta factor cancels against the exp(+eta t) that undoes the window.
    Hence eps_plus(x, t) = eps_plus(0, t - x/c0) to ~1e-15, and |eps_plus| is
    CONSTANT along x. The separated field is therefore a shear of two 1-D
    signals: a good picture and a sharp test, but not by itself a validation.
    Their SUM is not trivial, and it is what carries the physics -- at a free
    surface the two must cancel, and they do.
    """
    P, M, xi, n_fft, tau = _pm_spectra(t, signals, positions, c0, eta,
                                       n_fft, dispersion, attenuation)
    x = np.atleast_1d(np.asarray(x, float))
    n = len(tau)
    c0 = float(c0)

    # The minus branch grows as exp(+eta x / c0) on the way out; combined with
    # the record's own exp(+eta T) this is what can overflow.
    span = (tau[-1] - tau[0]) + np.abs(x).max() / c0
    if eta * span > 700:
        raise ValueError(f'eta * (record + max transit) = {eta*span:.1f}; '
                         'exp(+eta t) will overflow. Reduce eta or the x range.')

    q = max(1, int(decimate))
    n_out = n // q
    if n_out < 1:
        raise ValueError(f'decimate={q} leaves no samples in a record of {n}')
    keep = n_out * q                       # drop the ragged tail, if any
    amp = np.exp(+eta * tau)[:keep]
    t_out = np.asarray(t, float)[:keep].reshape(n_out, q).mean(axis=1)

    out_p = np.empty((len(x), n_out))
    out_m = np.empty((len(x), n_out))
    for i in range(0, len(x), int(chunk)):
        xb = x[i:i + int(chunk)][:, None]
        for X, sign, dst in ((P, -1j, out_p), (M, +1j, out_m)):
            y = np.fft.irfft(X[None, :] * np.exp(sign * xi[None, :] * xb),
                             n=n_fft, axis=1)[:, :keep] * amp
            dst[i:i + int(chunk)] = y.reshape(len(xb), n_out, q).mean(axis=2)
            del y
    return out_p, out_m, t_out


def separate_time_domain(t, signals, positions, c0, arrival_frac=0.02):
    """
    Two-gauge wave separation, done as a shift in TIME rather than a phase in
    frequency -- exact for a lossless, non-dispersive bar (c_p = c0,
    alpha = 0), and nothing else: unlike `separate`, there is no `eta`, no
    `dispersion`, no `attenuation` parameter here, because the whole point of
    this function is the case where none of those apply. Use `separate` (with
    `dispersion=None, attenuation=None` for the same lossless assumption) for
    anything more than two gauges, or where eta-regularisation is wanted.

    --------------------------------------------------------------------------
    The method
    --------------------------------------------------------------------------
    Let gauge 1 (nearer the interface) sit at x1 and gauge 2 at x2 > x1, and
    write the two waves AS SEEN AT GAUGE 1:

        P1(t) = p(t - x1/c0)      M1(t) = m(t + x1/c0)

    so that e1(t) = P1(t) + M1(t) directly. Gauge 2, further along, sees the
    same two waves shifted by the transit time tau = (x2 - x1)/c0:

        e2(t) = P1(t - tau) + M1(t + tau)

    Eliminating M1 between the two gives an exact CAUSAL recursion for P1:

        P1(t) = P1(t - 2*tau) + e1(t) - e2(t - tau)

    which is marched forward from P1 = 0 before the wave arrives -- the record
    must be quiescent for at least 2*tau before the first arrival, or this
    seed is wrong and everything built on it is too. M1 = e1 - P1 then falls
    out algebraically, and both are pure time shifts away from the interface
    (x = 0), exact because nothing here disperses or attenuates:

        p(t) = P1(t + x1/c0)      m(t) = M1(t - x1/c0)

    Verified against `separate(..., dispersion=None, attenuation=None)` on a
    measured shot (120/1200 mm gauge pair): relative RMS difference in P, M
    and F = P+M all ~2e-3, max ~3e-2 at the sharp wavefront edges -- two
    different numerical routes to the same lossless answer.

    Parameters
    ----------
    t : (N,) array
        Uniformly sampled time. Must start at the beginning of the record,
        quiescent, for at least 2*tau before the first arrival on either
        gauge -- see "What this costs you" below.
    signals : sequence of exactly two (N,) arrays
        Strain (or force, or anything linear in it) at the two gauges, in the
        same order as `positions`. Order does not matter -- as in `separate`,
        permuting the two together cannot change the result.
    positions : sequence of exactly two floats
        Distance of each gauge from the interface, matching `signals`. Both
        must be > 0 and distinct.
    c0 : float
        Elastic bar wave speed.
    arrival_frac : float
        Fraction of a gauge's own peak that counts as its first arrival, for
        the quiescent lead-in check. See `_time_domain_pm`; the default suits
        a clean calibration shot, a noisier real one may need it raised to
        `[<case>.trim].threshold`.

    Returns
    -------
    eps_plus, eps_minus : (N,) arrays
        Strain histories at x = 0, same convention as `separate`'s return.

    What this costs you
    --------------------------------------------------------------------------
    Two things `separate` does not have to worry about:

      * QUIESCENT LEAD-IN: seeding P1 = 0 needs at least 2*tau of clean record
        before the first arrival. Too little and the seed is simply wrong --
        this is checked below and raises rather than returning a silently
        biased result. Widening [<case>.trim].lead (or .baseline_before) in
        config.toml is the fix on a measured shot.
      * EDGE LOSS: propagating to x = 0 needs data up to x1/c0 PAST the time
        of interest for eps_plus (so the last x1/c0 of the record is
        unreliable/extrapolated) and x1/c0 BEFORE it for eps_minus (so the
        first x1/c0 is). This mirrors `backpropagate`'s own window discussion
        and is not a bug -- x = 0 is simply further from gauge 1's own
        vantage point in time than gauge 1's record alone can cover at either
        end.

    Fractional-sample shifts (tau and x1/c0 are rarely whole multiples of dt)
    are done by linear interpolation throughout, which is where the ~3e-2 max
    error above comes from -- concentrated at the steep wavefront edge.
    """
    P1, M1, x1, t = _time_domain_pm(t, signals, positions, c0, arrival_frac)
    c0 = float(c0)
    eps_plus = np.interp(t + x1 / c0, t, P1, left=0.0, right=P1[-1])
    eps_minus = np.interp(t - x1 / c0, t, M1, left=0.0, right=M1[-1])
    return eps_plus, eps_minus


def _time_domain_pm(t, signals, positions, c0, arrival_frac=0.02):
    """
    Shared core of `separate_time_domain` and `separate_time_domain_field`:
    the causal recursion, stopping one step short of shifting to a station.

    Returns P1, M1 -- the two waves AS SEEN AT GAUGE 1 (nearer the interface),
    in that function's own notation -- plus x1, gauge 1's distance from the
    interface, and t itself (as an array, for callers that only had a list).
    Propagating to any station x (0 for the interface, or any other local
    coordinate) is then just `interp(t +/- (x1 - x) / c0, t, P1 or M1, ...)`,
    which is all `separate_time_domain` does and all
    `separate_time_domain_field` does at more than one x.

    `arrival_frac` is the fraction of a gauge's own peak that counts as its
    first arrival, for the quiescent lead-in check below. The default, 0.02,
    is fine on a clean calibration shot; on a noisier real record a single
    early sample can sit just above 2 % of peak from pickup rather than the
    wavefront (measured on data/SHTB_PC_2026-09-03.txt: gauge 0 crosses 2 % at
    54 us on nothing but noise, and does not cross 5 % until 1169 us, ~1.1 ms
    later) and this needs raising to match. `[<case>.trim].threshold` in
    config.toml is the same "fraction of own peak" quantity, already tuned per
    case to find the real arrival for the trim -- callers reconstructing a
    real shot should pass that, not leave the default.
    """
    t = np.asarray(t, float)
    sig = [np.asarray(s, float) for s in signals]
    x = np.asarray(positions, float)

    if len(sig) != 2 or len(x) != 2:
        raise ValueError('separate_time_domain needs exactly two gauges; got '
                         f'{len(sig)} signals and {len(x)} positions')
    if np.any(x <= 0):
        raise ValueError('gauge positions must be > 0 (distance from the interface)')
    if x[0] == x[1]:
        raise ValueError('gauge positions must be distinct')
    n = len(t)
    if any(s.shape != (n,) for s in sig):
        raise ValueError('all signals must have the same length as t')

    dt = float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6):
        raise ValueError('t must be uniformly sampled')

    # sort so gauge 1 is nearer the interface, whatever order the caller used
    order = np.argsort(x)
    x1, x2 = x[order]
    e1, e2 = sig[order[0]], sig[order[1]]
    c0 = float(c0)
    tau = (x2 - x1) / c0

    # Quiescent lead-in check: first sample, on EITHER gauge, past a small
    # fraction of that gauge's own peak. Same idea as identify_bar_tension.py's
    # _rise_index, kept deliberately simple -- this only has to catch "not
    # enough lead-in", not time an edge precisely.
    thresh = float(arrival_frac)
    hit1 = np.abs(e1) > thresh * np.abs(e1).max()
    hit2 = np.abs(e2) > thresh * np.abs(e2).max()
    i_arrival = int(min(np.argmax(hit1) if hit1.any() else n,
                        np.argmax(hit2) if hit2.any() else n))
    lead = i_arrival * dt
    if lead < 2.0 * tau:
        raise ValueError(
            f'need >= 2*tau = {2*tau*1e3:.1f} us of quiescent lead-in before '
            f'the first arrival to seed the time-domain recursion; got only '
            f'{lead*1e3:.1f} us. Widen [<case>.trim].lead (or .baseline_before) '
            'in config.toml, or use separate() instead.')

    # e2 shifted by tau is needed at every sample, and e2 is fully known
    # up front, so this one is a single vectorised interpolation.
    e2_tau = np.interp(t - tau, t, e2, left=0.0, right=e2[-1])
    drive = e1 - e2_tau                       # e1(t) - e2(t - tau)

    # The recursion P1(t) = P1(t - 2*tau) + drive(t) is causal: by the time
    # sample i is being computed, every P1 value it needs (at t[i] - 2*tau,
    # strictly earlier) already exists. A plain loop, interpolating linearly
    # into the already-computed prefix, is simplest and fast enough --
    # O(N) at a few tens of thousands of samples costs milliseconds.
    P1 = np.zeros(n)
    lag = 2.0 * tau
    t0 = t[0]
    for i in range(n):
        t_lag = t[i] - lag
        if t_lag <= t0:
            p_lag = 0.0                        # before the record -> quiescent
        else:
            idx = (t_lag - t0) / dt
            k = int(idx)
            frac = idx - k
            k = min(k, i - 1)                  # guard the last half-open step
            p_lag = (1.0 - frac) * P1[k] + frac * P1[min(k + 1, i - 1)]
        P1[i] = p_lag + drive[i]
    M1 = e1 - P1
    return P1, M1, x1, t


def separate_time_domain_field(t, signals, positions, c0, x, arrival_frac=0.02):
    """
    `separate_time_domain`, evaluated at many stations instead of just x = 0 --
    the time-domain analogue of `separate_field`, for the same reason: a
    slider that moves the reconstruction plane off the interface needs the
    field, not just its value there.

    Propagating a pure travelling wave is a time shift and nothing else, so
    unlike `separate_field` this needs no frequency-domain round trip: with
    P1, M1 the two waves as seen at gauge 1 (distance x1 from the interface,
    see `_time_domain_pm`), the wave at any local station `x` (0 = interface,
    positive INTO the bar, same convention as `separate_field`) is just

        eps_plus(x, t)  = P1(t + (x1 - x) / c0)
        eps_minus(x, t) = M1(t - (x1 - x) / c0)

    which reduces to `separate_time_domain`'s own x1/c0 shifts at x = 0.
    Exact for a lossless, non-dispersive bar, and nothing else -- see
    `separate_time_domain` for what that costs (quiescent lead-in, edge loss).

    Parameters
    ----------
    t, signals, positions, c0
        Exactly as for `separate_time_domain` -- two gauges only.
    x : float or (n_x,) array
        Where to reconstruct, in this bar's local coordinate. Unrestricted,
        exactly as in `separate_field`; reconstructing outside the uniform bar
        (or past the other gauge) is extrapolation.
    arrival_frac : float
        As for `separate_time_domain`.

    Returns
    -------
    eps_plus, eps_minus : (n_x, N) arrays
        One row per station in `x`.
    """
    P1, M1, x1, t = _time_domain_pm(t, signals, positions, c0, arrival_frac)
    c0 = float(c0)
    xs = np.atleast_1d(np.asarray(x, float))
    out_p = np.empty((len(xs), len(t)))
    out_m = np.empty((len(xs), len(t)))
    for i, xv in enumerate(xs):
        shift = (x1 - xv) / c0
        out_p[i] = np.interp(t + shift, t, P1, left=0.0, right=P1[-1])
        out_m[i] = np.interp(t - shift, t, M1, left=0.0, right=M1[-1])
    return out_p, out_m


def backpropagate(t, signal, position, c0, eta=0.0, n_fft=None, dispersion=None,
                  direction='plus', attenuation=None):
    """
    Single-gauge reconstruction at x = 0, ASSUMING ONLY ONE WAVE IS PRESENT.

    This is not wave separation. One gauge gives one equation per frequency and
    there are two unknowns, so the second wave cannot be recovered -- it must be
    known to be absent. Where that assumption holds this is exact; where it
    fails the result is wrong by the whole of the neglected wave, silently.

    The assumption is legitimate in two situations:

      * a classical SHPB gauge placed far enough from the specimen that the
        incident and reflected pulses arrive at separate times (the textbook
        arrangement -- but it breaks down for long pulses / large strains,
        which is the very case multi-gauge separation exists to handle);
      * a direct-impact bar, before the reflection off its far free end gets
        back. The loading wave is generated AT the interface and travels away
        from it, and in a uniform bar nothing travels back until the free end
        returns it.

        Where that window ENDS needs care, because there are two answers and
        only one of them is the one you want. For a bar of length L and a gauge
        at distance d, measuring from the moment the wave left x = 0:

            in the GAUGE RECORD          the echo reaches the gauge at
                                         (2L - d) / c0
            in THIS FUNCTION'S OUTPUT    2 (L - d) / c0

        They differ by exactly d / c0, because the output IS the gauge record
        advanced by that much -- reconstructing at x = 0 means undoing the
        travel time, so everything in it happens d / c0 earlier than the gauge
        saw it. The second number is the one to check a reconstruction against,
        and it is the SHORTER of the two. Using the first overstates the valid
        window by 354 us on a gauge 498 mm out in polycarbonate.

        Note also which gauge expires first: the window is 2 (L - d) / c0, so a
        gauge FURTHER from the interface has a SHORTER window, not a longer one.
        Its echo has less bar to cross. Two gauges D apart expire 2D / c0 apart.

    Check the window before trusting the result. `direction` says which wave is
    the surviving one, in the local convention of this module: 'plus' travels
    away from the specimen, 'minus' toward it.

    Parameters
    ----------
    t, signal, position, c0, n_fft, dispersion
        As for `separate`, but with a single gauge.
    eta : float
        Exponential window. Unlike `separate` this may be 0 -- there is no
        determinant to regularise, and with eta = 0 and no dispersion the
        operation reduces to an exact time shift. Non-zero eta amplifies by
        exp(eta * position / c0), which is harmless for the usual values.
    direction : 'plus' | 'minus'
        Which of the two waves is assumed to be the only one present.

    Returns
    -------
    eps_plus, eps_minus : arrays
        The reconstructed wave and an array of zeros, in that order, so the
        result can be passed straight to `bar_interface` like `separate`'s.
    """
    t = np.asarray(t, float)
    s = np.asarray(signal, float)
    d = float(position)
    if direction not in ('plus', 'minus'):
        raise ValueError("direction must be 'plus' or 'minus'")
    if eta < 0:
        raise ValueError('eta must be >= 0')
    if d <= 0:
        raise ValueError('position must be > 0 (distance from the interface)')
    n = len(t)
    if s.shape != (n,):
        raise ValueError('signal must have the same length as t')

    dt = float(np.mean(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6):
        raise ValueError('t must be uniformly sampled')
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(4 * n)))
    n_fft = int(n_fft)

    tau = t - t[0]
    E = np.fft.rfft(s * np.exp(-eta * tau), n_fft)
    f = np.fft.rfftfreq(n_fft, dt)
    xi = _wavenumber(f, c0, eta, dispersion, attenuation)

    # E = W exp(-/+ i xi d)  ->  W = E exp(+/- i xi d)
    W = E * np.exp((1j if direction == 'plus' else -1j) * xi * d)
    w = np.fft.irfft(W, n=n_fft)[:n] * np.exp(+eta * tau)
    zero = np.zeros_like(w)
    return (w, zero) if direction == 'plus' else (zero, w)


def wavefront_time(t, w, frac=0.1):
    """
    When the loading wavefront left x = 0, robustly against a slow precursor.

    Every window that has to be placed relative to "the wave left the interface"
    -- the free-end echo at 2L/c, the single-wave window (2L-d)/c0 -- needs this
    instant, and the obvious rule, the first crossing of a few per cent of peak,
    is not safe on a real record. It latches onto whatever happens FIRST, and on
    a record with a low-level precursor that is the precursor. Measured on
    data/2026-08-20_PC_AFC.txt, whose trigger fires during a slow rise 1.6 ms
    ahead of the shot, a 2 % rule put the departure 1638 us early and moved
    every window with it -- silently, because every printed number stayed
    plausible.

    So anchor on the STEEPEST part of the rise, which a slow precursor cannot
    win, and walk back to where the slope falls below `frac` of its peak. That
    is the foot of the wavefront. On a record with no precursor it agrees with
    the naive rule to a few samples.

    The search is confined to BEFORE the peak of |w|, which matters as much as
    the rest: the steepest slope in a whole record is often the final unloading,
    and on the record above that put the "departure" 1.6 ms LATE instead of
    1.6 ms early. Loading precedes the peak by definition; unloading follows it.

    Parameters
    ----------
    t : (N,) array
        Uniform time base.
    w : (N,) array
        The wave reconstructed AT x = 0 -- `separate`'s P for a bar loaded at
        its interface. Its own onset IS the departure, which is why this takes
        the reconstruction and not a gauge record.
    frac : float
        Fraction of peak slope that counts as the foot of the rise.

    Returns
    -------
    float
        The departure time, in the units of `t`.
    """
    t = np.asarray(t, float)
    w = np.asarray(w, float)
    g = np.abs(np.gradient(w, float(np.mean(np.diff(t)))))
    i_pk = max(int(np.argmax(np.abs(w))), 1)
    i = int(np.argmax(g[:i_pk]))
    thr = frac * g[i]
    while i > 0 and g[i] > thr:
        i -= 1
    return float(t[i])


def single_wave_window(length, position, c0):
    """
    Time until the far-end reflection reaches a gauge, i.e. how long
    `backpropagate` remains valid on a direct-impact bar.

    `length` is the distance from the interface to the bar's free end and
    `position` the gauge's distance from the interface.
    """
    return (2.0 * float(length) - float(position)) / float(c0)


def bar_interface(eps_plus, eps_minus, E, A, c0, outward=+1, v0=0.0):
    """
    Force and particle velocity at the reconstruction plane of one bar.

    Parameters
    ----------
    eps_plus, eps_minus : arrays
        Output of `separate` for this bar.
    E, A : float
        Young's modulus and cross-sectional area of the BAR.
    c0 : float
        Bar wave speed.
    outward : +1 or -1
        Direction, in the GLOBAL frame, of the bar's local +x axis (which
        points from the interface into the bar, away from the specimen).
        For a specimen sandwiched between an input bar on the left and an
        output bar on the right, with global x increasing to the right:
            input bar   -> outward = -1   (its interior lies to the left)
            output bar  -> outward = +1
        The returned velocity is in the global frame; the force does not
        depend on this choice.
    v0 : float
        Rigid-body velocity of the bar before any wave arrives, in the GLOBAL
        frame. THIS IS NOT OPTIONAL FOR DIRECT IMPACT. Wave separation recovers
        only the wave content: a bar translating uniformly carries no strain,
        so no gauge can see its rigid-body motion, and the reconstructed
        velocity is the CHANGE from the initial state. In a classical SHPB all
        bars start at rest and v0 = 0 throughout. In a direct-impact test the
        flyer bar arrives at its impact velocity, and omitting it here makes
        the closing velocity wrong by exactly that amount -- which integrates
        into a strain error growing linearly in time.

    Returns
    -------
    force, velocity : arrays
        Force is negative in compression. Velocity is positive along global +x.
    """
    force = E * A * (eps_plus + eps_minus)
    # In the LOCAL frame a +x-travelling wave of strain e carries particle
    # velocity -c0*e, and a -x-travelling wave carries +c0*e:
    v_local = c0 * (eps_minus - eps_plus)
    return force, outward * v_local + v0


def specimen_response(t, force_in, vel_in, force_out, vel_out,
                      length, area, true_measures=False, contact_threshold=0.02,
                      loading='compression'):
    """
    Reduce the two interface states to the specimen's stress/strain response.

    Parameters
    ----------
    t : (N,) array
        Time, uniformly sampled.
    force_in, vel_in : arrays
        Force and global-frame velocity at the input-bar face (from
        `bar_interface`).
    force_out, vel_out : arrays
        Same at the output-bar face.
    length, area : float
        Original specimen length and cross-sectional area.
    true_measures : bool
        False (default) returns ENGINEERING stress and strain, referred to the
        original length and area. True returns logarithmic strain and true
        stress assuming constant volume. Use False to compare against a
        simulation whose specimen has a fixed cross-section.
    contact_threshold : float
        Fraction of peak force below which the bars are taken to have separated
        from the specimen. Outside contact the closing velocity of the two bar
        faces no longer describes specimen deformation -- the faces keep moving
        but the specimen does not follow -- so the strain integral is frozen
        rather than allowed to run away. Set to 0 to integrate unconditionally.
        In tension the specimen is threaded into both bars and cannot separate,
        so this only gates the quiescent parts of the record (and anything after
        the specimen fails).
    loading : 'compression' | 'tension'
        Which sense counts as positive in the returned stress and strain.
        'compression' (default) suits a compression bar and matches simulate_compression.py.
        'tension' suits an SHTB and matches simulate_tension.py: stress and
        strain come out positive in tension, and the specimen is taken to
        deform when the bar faces move APART rather than together.

    Returns
    -------
    dict with keys
        strain, strain_rate, stress      -- compression POSITIVE
        stress_in, stress_out            -- from each face separately
        equilibrium                      -- |F1 - F2| / max|F1|, a quality metric
        contact                          -- bool mask, True while loaded

    Notes
    -----
    Strain comes from integrating the velocity difference, so it inherits the
    low-frequency floor set by `eta` in `separate`: a small DC error in the
    velocities integrates into a linear drift in strain. Start the record before
    the wave arrives, and check that the strain returns toward zero after the
    event if the specimen unloads.
    """
    if loading not in ('compression', 'tension'):
        raise ValueError("loading must be 'compression' or 'tension'")
    # s = +1 flips the sign of forces/velocities so that the loading sense of
    # interest comes out positive; everything below is written once for both.
    s = 1.0 if loading == 'compression' else -1.0

    t = np.asarray(t, float)
    force_in = np.asarray(force_in, float)
    force_out = np.asarray(force_out, float)
    # relative velocity of the two faces; positive = specimen being deformed
    # (closing in compression, opening in tension)
    closing = s * (np.asarray(vel_in, float) - np.asarray(vel_out, float))

    # gate the integral on the specimen actually being loaded
    mean_force = 0.5 * (force_in + force_out)
    peak_c = np.max(np.abs(mean_force))
    if contact_threshold > 0 and peak_c > 0:
        contact = -s * mean_force > contact_threshold * peak_c
    else:
        contact = np.ones_like(t, dtype=bool)

    # engineering strain, compression positive
    disp = _cumtrapz(np.where(contact, closing, 0.0), t)
    eng_strain = disp / length
    eng_rate = np.where(contact, closing, 0.0) / length

    # stresses from each face, positive in the chosen loading sense
    stress_in = -s * force_in / area
    stress_out = -s * force_out / area

    if true_measures:
        # current length L = L0 (1 - eng_strain); constant volume -> A = A0/(1-e)
        stretch = 1.0 - eng_strain
        if np.any(stretch <= 0):
            raise ValueError('engineering strain reached 1.0; specimen fully collapsed')
        strain = -np.log(stretch)
        rate = closing / (length * stretch)
        stress_in = stress_in * stretch
        stress_out = stress_out * stretch
    else:
        strain, rate = eng_strain, eng_rate

    peak = np.max(np.abs(force_in))
    return dict(strain=strain, strain_rate=rate,
                stress=0.5 * (stress_in + stress_out),
                stress_in=stress_in, stress_out=stress_out,
                contact=contact,
                equilibrium=np.abs(force_in - force_out) /
                            (peak if peak > 0 else 1.0))


def _cumtrapz(y, t):
    """Cumulative trapezoidal integral, same length as y, starting at 0."""
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def conditioning(f, positions, c0, eta, dispersion=None, attenuation=None):
    """
    Diagnostic: normalised system determinant, 1 = ideal, 0 = singular.

    Use this to check a gauge layout before committing to it. Dips mean the
    gauge spacings are commensurate with a half-wavelength at that frequency,
    where noise is amplified. Equal spacings are the worst case: positions
    {2.2, 1.2, 0.2} have spacings 1.0/1.0/2.0 and are exactly singular at every
    multiple of c0/2.
    """
    f = np.asarray(f, float)
    x = np.asarray(positions, float)
    xi = _wavenumber(f, c0, eta, dispersion, attenuation)
    xc = np.conj(xi)
    h1 = sum(np.exp(-1j * (xi - xc) * d) for d in x)
    h2 = sum(np.exp(+1j * (xi - xc) * d) for d in x)
    g = sum(np.exp(+1j * (xi + xc) * d) for d in x)
    return ((h1 * h2 - g * np.conj(g)) / (h1 * h2)).real
