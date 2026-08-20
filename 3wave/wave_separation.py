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

__all__ = ['separate', 'separate_field', 'backpropagate', 'bar_interface',
           'specimen_response', 'conditioning', 'single_wave_window']


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
        returns it. For a bar of length L and a gauge at distance d, that
        leaves a single-wave window lasting until t = (2L - d) / c0.

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
