"""
Load a MEASURED shot into the same dict shape that dump.npz produces.

    from experiment import load_experiment
    d = load_experiment('experiment_pc_bar')

This is the half of rig readiness NOTES.md left open: `--l-free-ref` made the
one length a rig cannot derive into an input, and this makes the record itself
into one. Everything downstream -- identify_bar_compression.py,
identify_attenuation.py, reconstruct_interface.py -- opens a real shot and a
simulated one with the same line, because both hand back the same keys.

--------------------------------------------------------------------------
Force is a perfectly good input, and E*A never enters
--------------------------------------------------------------------------
The file holds FORCE in kN. Nothing converts it to strain and nothing should.
`separate` is linear and its docstring says as much -- "strain, or any quantity
proportional to it -- force, volts" -- so with force in, P and M come back in
kN and

    F_interface = P + M

is the force at the impact face directly. E, A and rho appear NOWHERE in the
reconstruction; only c0, the gauge positions and eta do. That is worth knowing
because E and A are exactly the numbers a rig knows worst, and this deliverable
does not need them. They come back only for the E = rho c^2 closure, which is a
by-product and is labelled as one.

--------------------------------------------------------------------------
What a real record has that a simulated one does not
--------------------------------------------------------------------------
Three things, and all three break something if left alone:

  * A PRE-TRIGGER BASELINE. This file starts 1638 us before the shot. `separate`
    requires the signals quiescent at t[0] -- fine -- but `_rise_time` in
    identify_bar_compression.py takes a global argmax over |gradient|, and the
    edge-template width derived from it is meaningless if the record's own noise
    floor is in the running. The loader trims to just ahead of the first arrival
    and re-zeros t.
  * A DC OFFSET on each channel, from the amplifier rather than the bar.
    Measured here at -1.1e-5 and -6.1e-5 kN: small, and free to remove.
  * NO GROUND TRUTH. There is no c0, no true gauge position, no interface force
    to compare against. Those keys are ABSENT rather than filled with a guess,
    which is what lets the identification scripts print a dash instead of a
    fabricated error column. `L_free_out` is present because it is the one
    measured length the operator supplies, not because it is truth.

--------------------------------------------------------------------------
One instrumented bar
--------------------------------------------------------------------------
An uninstrumented bar is reported as `eps_in` with shape (0, N) and an empty
`pos_in`, rather than by omitting the keys. Downstream then asks how many
gauges a bar has, which is the question it actually wants answered, instead of
whether a key exists.
"""

import os

import numpy as np

import config

__all__ = ['load_experiment']

US_PER_MS = 1.0e3


def load_experiment(case, path=None):
    """
    Read one measured shot named by a config case.

    Parameters
    ----------
    case : str
        A member of config.EXPERIMENT_CASES.
    path : str, optional
        Override the data file named in the config, e.g. to run the same
        geometry against a second shot.

    Returns
    -------
    dict
        The dump.npz keys that a measured shot can honestly fill:

            t, dt, N              time base [ms], re-zeroed at the trim point
            t0_file               where that zero sits in the FILE's own time
                                  base [us], so results can be referred back
            eps_out, pos_out      (n_gauge, N) signals and their TAPE positions
            eps_in, pos_in        empty -- the input bar carries no gauge
            units                 'kN'; these are forces, not strains
            L_free_out, L_bar_out the supplied bar length [mm]
            L_free_in             the input bar's length, for its c = 2L/P only
            A_out, rho_out        geometry and an assumed density, for closures
            L_specimen            0.0 -- the bars are struck face to face
            loading, eta          as for a simulated dump
            case, cfg, source     provenance: which case, its config, which file

        Deliberately ABSENT: c0_in, c0_out, E_in, E_out, force_iface_*, spec_*.
        Nothing here knows them.
    """
    cfg = config.load(case)
    src = path if path is not None else cfg['file']
    if not os.path.isabs(src):
        src = os.path.join(os.path.dirname(config.DEFAULT_PATH), src)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{case}: {src} not found. config.toml names it as "file"; paths '
            'are taken relative to config.toml, not to the working directory.')

    raw = np.loadtxt(src)
    cols = dict(cfg['columns'])
    i_t = int(cols.pop('time'))
    names = list(cols)                       # insertion order = config order
    if raw.shape[1] <= max([i_t] + [cols[n] for n in names]):
        raise ValueError(f'{src}: {raw.shape[1]} columns, but the config asks '
                         f'for index {max([i_t] + list(cols.values()))}')

    t_us = raw[:, i_t]
    t_us_full = t_us.copy()
    sig = np.array([raw[:, cols[n]] for n in names], float)

    dt_us = float(np.mean(np.diff(t_us)))
    if not np.allclose(np.diff(t_us), dt_us, rtol=1e-4):
        raise ValueError(f'{src}: time column is not uniformly sampled; '
                         '`separate` requires a uniform base')

    trim = cfg.get('trim', {})
    if trim.get('baseline', True):
        sig = _debias(sig, t_us, float(trim.get('baseline_before', 0.0)))
    if 'start' in trim:
        sig, t_us = _cut(sig, t_us, float(trim['start']))
    else:
        sig, t_us = _trim(sig, t_us, dt_us,
                          float(trim.get('threshold', 0.05)),
                          float(trim.get('lead', 50.0)))

    dt = dt_us / US_PER_MS
    n = sig.shape[1]
    # Where the analysis t = 0 sits in the FILE's own time base. The trim moves
    # it, and without this every time printed downstream is offset from the
    # record the operator is looking at -- by 1638 us on the specimen shot.
    t0_file = float(t_us_full[len(t_us_full) - n])
    bar = cfg['bar']
    L = float(bar['length'])
    d = dict(
        t=np.arange(n) * dt, dt=dt, N=n,
        eps_out=sig, pos_out=np.asarray(cfg['gauges'], float),
        eps_in=np.zeros((0, n)), pos_in=np.zeros(0),
        units='kN',
        L_free_out=L, L_bar_out=L,
        L_free_in=float(cfg.get('L_free_in_ref', 0.0)) or None,
        A_out=0.25 * np.pi * float(bar['diameter']) ** 2,
        rho_out=float(bar.get('rho', 0.0)) or None,
        L_specimen=0.0,
        loading=str(cfg['loading']),
        eta=float(cfg['analysis']['eta']),
        case=case, cfg=cfg, source=src, t0_file=t0_file,
    )
    if d['eps_out'].shape[0] != len(d['pos_out']):
        raise ValueError(f'{case}: {d["eps_out"].shape[0]} channels but '
                         f'{len(d["pos_out"])} tape positions')
    return d


def _debias(sig, t_us, before=0.0):
    """
    Remove each channel's pre-trigger mean.

    `before` is where the pre-trigger window ends. Zero -- the scope's own
    trigger instant -- is the obvious default and needs no threshold of its own,
    but it is only right when nothing happens before the trigger. It is NOT
    right on a shot whose trigger fires late: 2026-08-20_PC_AFC.txt carries a
    slow rise from -1138 us onward, so averaging over t < 0 would subtract a
    fifth of a kN of real signal and call it an offset. Set
    `baseline_before = -1200` there and the true noise floor, 5.9e-04 kN,
    appears -- the same as every other channel on this rig.

    A record with too little pre-trigger to average is left alone rather than
    being de-biased against its own signal.
    """
    pre = t_us < before
    if pre.sum() < 16:
        return sig
    return sig - sig[:, pre].mean(axis=1, keepdims=True)


def _cut(sig, t_us, start):
    """
    Keep the record from an EXPLICIT time, and re-zero t.

    The alternative to _trim's arrival detection, and the right choice whenever
    the record is already quiescent where you want to start -- which is what
    `separate` actually requires. Arrival detection exists to throw away a long
    quiet lead-in; where that lead-in is not quiet, or where it is wanted,
    saying so beats tuning a threshold against it.

    Starting LATE is the trap. It looks tidy and it breaks the quiescence
    assumption: cutting 2026-08-20_PC_AFC.txt at 500 us, just before its main
    event, leaves the bar carrying a standing 0.13 kN and its own history, and
    the free-end null goes from 3.1e-02 to 1.9e-01. Start where the record is
    quiet, not where the interesting part begins.
    """
    i0 = int(np.searchsorted(t_us, start))
    if i0 >= len(t_us) - 1:
        raise ValueError(f'trim.start = {start} is at or past the end of the '
                         f'record ({t_us[-1]})')
    return sig[:, i0:].copy(), t_us[i0:] - t_us[i0]


def _trim(sig, t_us, dt_us, threshold, lead_us):
    """
    Cut the record back to `lead_us` ahead of the first arrival, and re-zero t.

    The first arrival is the first sample, on ANY channel, exceeding `threshold`
    of that channel's peak. Taking the earliest across channels rather than
    per-channel is deliberate: the channels must stay time-aligned, since every
    delay the identification measures is a difference between them.
    """
    peak = np.abs(sig).max(axis=1, keepdims=True)
    if not np.all(peak > 0):
        raise ValueError('a channel is identically zero')
    hit = np.abs(sig) > threshold * peak
    if not hit.any():
        raise ValueError(f'no channel reaches {threshold:g} of its own peak')
    first = int(np.min([int(np.argmax(h)) for h in hit]))
    i0 = max(0, first - int(round(lead_us / dt_us)))
    return sig[:, i0:].copy(), t_us[i0:] - t_us[i0]
