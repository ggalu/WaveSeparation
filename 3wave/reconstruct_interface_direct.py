"""
Reconstruct the force at the SHTB coupler straight from the record and the
tape -- no identification step in between.

    python3 reconstruct_interface_direct.py [--case experiment_tension_bar_2]
                                            [--headless]

Every other real-data script in this folder is a two-stage pipeline: an
`identify_bar_*.py` script reads the echo train and writes `c0` and the gauge
positions to `bar_identified.npz`, and a second script -- `reconstruct_
interface.py`, `bar_equilibrium.py` -- reads that file. This script collapses
the two stages into one. It never touches `bar_identified.npz`; everything it
needs comes from the data file named in `config.toml` and from the config
itself:

    gauges              tape distance of each gauge from the coupler [mm]
    input_bar/output_bar.diameter, .L_input/.L_output
    specimen.length     the coupler
    analysis.eta

--------------------------------------------------------------------------
c0 from the wave transit time, not from the echo train
--------------------------------------------------------------------------
`identify_bar_tension.py` gets `c0` from the assembly's free-end echo, which
needs the whole `Q = 2 L_free_ref/c0` machinery, a tape measurement to the far
end, and a record long enough to contain that echo. There is a much shorter
route to `c0` when a bar already carries two gauges: the same wave passes both,
`D` = their TAPE spacing apart, so

    c0 = D / (time between the two gauges seeing it)

`gauge_transit_time` gets that delay from the peak of the cross-correlation
between the two gauges' DERIVATIVES -- edges correlate more sharply than the
pulse shape itself, the same reasoning `identify_attenuation.py` uses for its
own transfer-function fit. It is deliberately NOT told which gauge the wave
reaches first: on this rig the input bar is loaded from its far end (the
far-from-coupler gauge sees it first) while the output bar is loaded from its
near end (the near-to-coupler gauge sees it first), so both signs of lag are
searched and the stronger peak wins.

This buys speed and independence at the price of precision: `D` here is the
TAPE spacing, good to a percent or so, where the echo-train identification
resolves it to a fraction of a millimetre by leaning on a long reference
baseline (see "Calibrating the bar" in README.md). Measured on this shot,
`gauge_transit_time` alone gives `c0` about 1-2 % away from
`identify_bar_tension.py`'s value -- consistent with the tape's own D error,
and the point of this script is that it never needed the echo train, the
striker pulse length, or a reference length to the free end to get there.

--------------------------------------------------------------------------
What is checked, and why not the other two
--------------------------------------------------------------------------
`reconstruct_interface.py`'s causality/unilateral/separation checks assume a
DRY CONTACT that can part -- the direct-impact PC bar it was written for. This
rig's coupler is a threaded joint: the two bars never part and the joint can
legitimately carry tension, so those checks do not apply here (this is the
same reasoning `reconstruct_interface.py` itself uses to gate its "tensile"
check off for a TENSION-loaded case). What this script checks instead:

    EQUILIBRIUM   force is continuous across a genuine joint: F_in and F_out,
                 reconstructed from EACH bar's own two gauges independently,
                 must agree. See bar_equilibrium.py, which runs the same
                 check from the identified numbers.
    FREE-END NULL the output bar's far end is a genuine free surface (the
                 input bar's is the anvil/striker, not free) -- see
                 identify_bar_tension.py's own free-end null test.

Both need no ground truth, which is what makes them worth running on a real
shot: nothing here is checked against a simulator.

--------------------------------------------------------------------------
Optimizing positions against a known c0
--------------------------------------------------------------------------
`--c0` inverts the usual direction. Ordinarily `c0 = D_tape / transit_time`
treats the tape as given and c0 as the unknown. Handed a c0 from elsewhere
(the echo-train identification, a handbook figure, a second shot) it makes
more sense to trust that number and the measured transit time -- which comes
from a sub-sample cross-correlation peak over a multi-millisecond record, far
more precise than a tape read to the nearest millimetre -- and let the
POSITIONS absorb the small disagreement instead:

    D_opt = c0 * transit_time

`D_opt` is then a single number, not two, so recovering two gauge positions
from it needs one more decision: where to anchor the pair. Lacking any
information about which of the two tape reads is better, the least-committal
choice is the one that changes them as little as possible -- minimize
(x_near - X_near_tape)^2 + (x_far - X_far_tape)^2 subject to the single linear
constraint x_far - x_near = D_opt. That is a textbook equality-constrained
least squares and it has a closed form: split the correction evenly,

    x_near = mid - D_opt/2,   x_far = mid + D_opt/2,   mid = (X_near + X_far)/2

which leaves the pair's tape MIDPOINT untouched and moves each gauge by half
of `D_opt - D_tape` in opposite directions -- consistent with "D is the
physical parameter; the individual x is not" (README, "Know the spacing,
rather than choose it").
"""
import argparse

import numpy as np

import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--case', default='experiment_tension_bar_2',
                 help='config case to reconstruct (default: %(default)s)')
_ap.add_argument('--c0-lo', type=float, default=1000.0, metavar='MM/MS',
                 help='lower bound of the wave-speed search used to find the '
                      'gauge-to-gauge transit time (default: %(default)s)')
_ap.add_argument('--c0-hi', type=float, default=6500.0, metavar='MM/MS',
                 help='upper bound of the same search (default: %(default)s)')
_ap.add_argument('--c0', type=float, default=None, metavar='MM/MS',
                 help='treat this as the TRUE wave speed instead of computing '
                      'one per bar from the transit time. The gauge positions '
                      'are then OPTIMIZED to be consistent with it -- see '
                      '"Optimizing positions against a known c0" below.')
HEADLESS, ARGS = plotting.init(parser=_ap)

import config
from experiment import load_experiment
from wave_separation import separate

CASE = ARGS.case
if CASE not in config.EXPERIMENT_CASES:
    raise SystemExit(f'{CASE!r} is not a measured shot; expected one of '
                     f'{config.EXPERIMENT_CASES}')

cfg = config.load(CASE)
d = load_experiment(CASE)
t, dt, N = d['t'], d['dt'], d['N']
eta = d['eta']
UNITS = d.get('units', 'strain')
SCALE = 1.0 if UNITS != 'strain' else 1e6
USYM = UNITS if UNITS != 'strain' else 'ustrain'

BARS = [b for b in ('in', 'out') if d[f'eps_{b}'].shape[0] >= 2]
if not BARS:
    raise SystemExit(f'[{CASE}] has fewer than two gauges on every bar -- '
                     'gauge_transit_time needs a pair on the SAME bar to '
                     'measure c0, and separate() needs a pair to run at all.')

print(__doc__.split('---')[0].strip())
print(f'\nrecord     : {d["source"]}')
print(f'bars       : {", ".join(BARS)} ({sum(d[f"eps_{b}"].shape[0] for b in BARS)} '
      f'gauges total), eta = {eta:g} /ms')
print(f'c0 search  : {ARGS.c0_lo:.0f} - {ARGS.c0_hi:.0f} mm/ms')
if ARGS.c0 is not None:
    print(f'c0 FIXED   : {ARGS.c0:.1f} mm/ms (assumed true) -- gauge positions '
          'will be optimized to match it, not the other way round')


# --------------------------------------------------------------------------
# c0 from the gauge-to-gauge wave transit time
# --------------------------------------------------------------------------
def _xcorr(a, b, nf):
    """c[k] = sum_j a[j+k] b[j] for k = 0 .. nf-1 (circular, k > n wraps to
    negative lag); zero-padded via nf so nothing wraps within the small window
    `gauge_transit_time` actually searches."""
    return np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)


def _refine(c, i):
    """Sub-sample offset of the peak at index i, by a 3-point parabola."""
    if i <= 0 or i >= len(c) - 1:
        return 0.0
    den = c[i - 1] - 2.0 * c[i] + c[i + 1]
    return 0.0 if den == 0 else 0.5 * (c[i - 1] - c[i + 1]) / den


def gauge_transit_time(s_near, s_far, dt, D, c0_lo, c0_hi):
    """
    Signed transit time between two gauges D apart on the same bar, from the
    peak of the cross-correlation of their derivatives.

    `s_near`/`s_far` label the pair by TAPE position only (closer to /
    further from the interface); which one the wave reaches first depends on
    where this bar is loaded from and is NOT assumed -- both signs of lag are
    searched, bounded to [D/c0_hi, D/c0_lo] so that an echo many bar-lengths
    later cannot be mistaken for the direct arrival.

    Returns
    -------
    float
        Positive if `s_near` arrives first, negative if `s_far` does.
    """
    ga = np.gradient(np.asarray(s_near, float), dt)
    gb = np.gradient(np.asarray(s_far, float), dt)
    n = len(ga) + len(gb) - 1
    nf = 1 << int(np.ceil(np.log2(n)))
    c = _xcorr(gb, ga, nf)   # peaks at k = +lag when s_near leads s_far

    lo = max(1, int(np.floor(D / c0_hi / dt)))
    hi = min(nf // 2 - 1, int(np.ceil(D / c0_lo / dt)))
    if hi <= lo:
        raise ValueError(f'transit-time search window is empty for D={D:.1f} '
                         f'mm: c0 in [{c0_lo:.0f}, {c0_hi:.0f}] mm/ms gives a '
                         f'lag of {lo}-{hi} samples at dt={dt*1e3:.3f} us')

    i_pos = lo + int(np.argmax(c[lo:hi + 1]))          # s_near first
    i_neg = nf - hi + int(np.argmax(c[nf - hi:nf - lo + 1]))   # s_far first
    if c[i_pos] >= c[i_neg]:
        return (i_pos + _refine(c, i_pos)) * dt
    return (i_neg - nf + _refine(c, i_neg)) * dt


def optimize_positions(pos, i_near, i_far, D_opt):
    """
    Move exactly the pair (i_near, i_far) so their spacing becomes `D_opt`,
    by the minimum-correction split derived in "Optimizing positions against
    a known c0" above: split the change evenly, leaving the pair's tape
    midpoint untouched.
    """
    mid = 0.5 * (pos[i_near] + pos[i_far])
    out = pos.copy()
    out[i_near] = mid - 0.5 * D_opt
    out[i_far] = mid + 0.5 * D_opt
    return out


C0, X, X_TAPE, F, P, M = {}, {}, {}, {}, {}, {}
for b in BARS:
    sig = list(d[f'eps_{b}'])
    pos = np.asarray(d[f'pos_{b}'], float)
    i_near, i_far = int(np.argmin(pos)), int(np.argmax(pos))
    D = float(pos[i_far] - pos[i_near])
    lag = gauge_transit_time(sig[i_near], sig[i_far], dt, D,
                             ARGS.c0_lo, ARGS.c0_hi)
    who = (f'gauge at {pos[i_near]:.0f} mm' if lag > 0 else
          f'gauge at {pos[i_far]:.0f} mm') + ' arrives first'
    print(f'\n{b:>3} bar: {len(sig)} gauges at '
          f'[{", ".join(f"{v:.1f}" for v in pos)}] mm (tape), D = {D:.1f} mm')
    print(f'       transit time {abs(lag)*1e3:.3f} us ({who})')

    X_TAPE[b] = pos
    if ARGS.c0 is not None:
        c0 = float(ARGS.c0)
        D_opt = c0 * abs(lag)
        pos_opt = optimize_positions(pos, i_near, i_far, D_opt)
        print(f'       c0 fixed at {c0:.1f} mm/ms -> D consistent with it is '
              f'{D_opt:.2f} mm (tape said {D:.2f}, {D_opt - D:+.2f} mm)')
        for k in (i_near, i_far):
            print(f'       gauge at {pos[k]:.2f} mm (tape) -> '
                  f'{pos_opt[k]:.2f} mm (optimized), {pos_opt[k]-pos[k]:+.2f} mm')
        pos = pos_opt
    else:
        c0 = D / abs(lag)
        print(f'       -> c0 = {c0:.1f} mm/ms')
    C0[b] = c0
    X[b] = pos
    p, m = separate(t, sig, pos, c0=c0, eta=eta)
    P[b], M[b], F[b] = p, m, p + m

# --------------------------------------------------------------------------
# equilibrium across the coupler -- needs both bars
# --------------------------------------------------------------------------
if 'in' in BARS and 'out' in BARS:
    peak = float(np.abs(F['in']).max())
    equilibrium = np.abs(F['in'] - F['out']) / (peak if peak > 0 else 1.0)
    NULL_WINDOW = float(cfg.get('null', {}).get('window', 0.75))
    amp = max(float(np.abs(F['in']).max()), float(np.abs(F['out']).max()))
    i0 = int(np.argmax((np.abs(F['in']) + np.abs(F['out'])) > 0.02 * amp))
    i1 = min(int(NULL_WINDOW * N), N - 1)
    win = slice(i0, i1)
    print(f'\n--- force equilibrium across the coupler, in vs out '
          f'---------------------')
    print(f'analysis window {t[i0]:.3f}-{t[i1]:.3f} ms')
    print(f'|F_in - F_out| / max|F_in| : mean {equilibrium[win].mean():.4e}, '
          f'max {equilibrium[win].max():.4e}')
    print('a genuine joint carries the same force on both sides; what is '
          'left is model\nerror -- most likely a small extra transit time '
          'through the coupler that this\nreconstruction, treating each bar '
          'as if it ran straight to the interface, does\nnot account for.')
else:
    equilibrium = None
    print(f'\nonly the {BARS[0]} bar has two gauges -- no equilibrium check '
          '(needs both).')

# --------------------------------------------------------------------------
# free-end null -- only the output bar's far end is genuinely free
# --------------------------------------------------------------------------
if 'out' in BARS:
    L_out = float(d['L_free_out'])
    L_free = L_out - X['out']
    NULL_TOL = float(cfg.get('null', {}).get('tol', 5.0e-3))
    NULL_WINDOW = float(cfg.get('null', {}).get('window', 0.75))
    p_free, m_free = separate(t, list(d['eps_out']), L_free, c0=C0['out'],
                              eta=eta)
    total = p_free + m_free
    amp_free = float(np.abs(p_free).max())
    j0 = int(np.argmax(np.abs(p_free) > 0.02 * amp_free))
    j1 = min(int(NULL_WINDOW * N), N)
    w = slice(j0, j1)
    null_rms = (float(np.sqrt(np.mean(total[w] ** 2)) / amp_free)
               if j1 > j0 else float('nan'))
    print(f'\n--- free-end null, output bar\'s far end (no ground truth) '
          '------------')
    print(f'residual |eps+ + eps-| : rms {null_rms:.2e} of peak, threshold '
          f'{NULL_TOL:.1e} -> {"PASS" if null_rms <= NULL_TOL else "FAIL"}')
else:
    null_rms = float('nan')
    p_free = m_free = total = None
    print('\nno output-bar pair -- no free-end null (needs a free far end).')

# --------------------------------------------------------------------------
# .dat
# --------------------------------------------------------------------------
T0 = float(d.get('t0_file', 0.0))
_STEM = f'interface_force_direct_{CASE}' + (
    f'_c0-{ARGS.c0:.0f}' if ARGS.c0 is not None else '')
DAT = f'{_STEM}.dat'
cols = [t * 1e3 + T0]
header = 'time[us]'
for b in BARS:
    cols += [F[b], P[b], M[b]]
    header += f'  F_{b}[{UNITS}]  P_{b}[{UNITS}]  M_{b}[{UNITS}]'
if equilibrium is not None:
    cols.append(equilibrium)
    header += '  equilibrium'
_POS_LBL = 'optimized' if ARGS.c0 is not None else 'tape'
np.savetxt(DAT, np.column_stack(cols),
          header=header + '\n'
                 f'time is the SOURCE FILE\'s own base (analysis t=0 sits at '
                 f'{T0:.1f} us there)\n'
                 + ', '.join(f'{b} bar: c0={C0[b]:.1f} mm/ms, x='
                            f'{[round(float(v), 1) for v in X[b]]} mm '
                            f'({_POS_LBL})'
                            for b in BARS)
                 + f', eta={eta:g}, x=0 is each bar\'s own face at the coupler')
print(f'\nwrote {DAT}')

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'
tt = t * 1e3 + T0

N_ROWS = 1 + 2 * len(BARS) + (1 if equilibrium is not None else 0) + \
        (1 if p_free is not None else 0)
fig, axes = plt.subplots(N_ROWS, 1, figsize=(11, 3.6 * N_ROWS), sharex=False)
fig.patch.set_facecolor(SURFACE)
row = 0

# --- what was measured -----------------------------------------------------
for b in BARS:
    for k, s in enumerate(d[f'eps_{b}']):
        axes[row].plot(tt, s * SCALE, lw=.9,
                       color=(BLUE, ORANGE)[k % 2],
                       label=f'{b}-{k} at {X[b][k]:.0f} mm')
axes[row].set_ylabel(f'Gauge signal ({USYM})')
axes[row].set_title(f'What was measured — {CASE}', loc='left', fontsize=11)
axes[row].legend(frameon=False, fontsize=8, labelcolor=MUTED, ncol=len(BARS)*2)
row += 1

# --- transit-time correlation, one panel per bar ---------------------------
for b in BARS:
    sig = list(d[f'eps_{b}'])
    pos = X[b]
    i_near, i_far = int(np.argmin(pos)), int(np.argmax(pos))
    D = float(pos[i_far] - pos[i_near])
    ga = np.gradient(sig[i_near], dt)
    gb = np.gradient(sig[i_far], dt)
    nf = 1 << int(np.ceil(np.log2(len(ga) + len(gb) - 1)))
    c = _xcorr(gb, ga, nf)
    lag_samp = abs(D / C0[b]) / dt
    span = int(np.ceil(1.5 * lag_samp)) + 5
    lags = np.arange(-span, span + 1)
    vals = np.array([c[k % nf] for k in lags])
    axes[row].plot(lags * dt * 1e3, vals / np.abs(vals).max(), color=INK, lw=1.0)
    axes[row].axvline(np.sign(lags[np.argmax(vals)]) * abs(D / C0[b]) * 1e3,
                      color=ORANGE, lw=1.1, ls='--')
    axes[row].set_ylabel('correlation (norm.)')
    axes[row].set_title(f'{b} bar — gauge-to-gauge transit time '
                        f'{abs(D/C0[b])*1e3:.2f} us over D = {D:.0f} mm -> '
                        f'c0 = {C0[b]:.1f} mm/ms', loc='left', fontsize=10)
    axes[row].set_xlabel('lag (us)')
    row += 1

# --- the waves at each bar's own face --------------------------------------
for b in BARS:
    axes[row].plot(tt, P[b] * SCALE, color=BLUE, lw=.9, label='$P$ (leaving)')
    axes[row].plot(tt, M[b] * SCALE, color=ORANGE, lw=.9, label='$M$ (returning)')
    axes[row].axhline(0, color=GRID, lw=.8)
    axes[row].set_ylabel(f'Wave ({USYM})')
    axes[row].set_title(f'{b} bar — separated waves at its own face touching '
                        'the coupler', loc='left', fontsize=10)
    axes[row].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')
    row += 1

# --- equilibrium -------------------------------------------------------------
if equilibrium is not None:
    axes[row].plot(tt, F['in'] * SCALE, color=BLUE, lw=.9, label='F_in')
    axes[row].plot(tt, F['out'] * SCALE, color=ORANGE, lw=.9, label='F_out')
    axes[row].axhline(0, color=GRID, lw=.8)
    axes[row].set_ylabel(f'Force ({UNITS})')
    axes[row].set_title(f'Force at each bar\'s own face — equilibrium residual '
                        f'mean {equilibrium[win].mean():.3e}, max '
                        f'{equilibrium[win].max():.3e}', loc='left', fontsize=10)
    axes[row].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')
    row += 1

# --- free-end null -----------------------------------------------------------
if p_free is not None:
    band = NULL_TOL * amp_free * SCALE
    axes[row].axhspan(-band, band, color=BLUE, alpha=.18,
                      label=f'pass threshold +-{NULL_TOL:.1e}')
    axes[row].plot(tt, total * SCALE, color=INK, lw=.9,
                   label=r'$\varepsilon_+ + \varepsilon_-$ at the free surface')
    axes[row].axhline(0, color=GRID, lw=.8)
    axes[row].set_ylabel(f'Signal ({USYM})')
    axes[row].set_title(f'Output bar\'s free surface — rms {null_rms:.2e} of '
                        f'peak, {"PASS" if null_rms <= NULL_TOL else "FAIL"}',
                        loc='left', fontsize=10)
    axes[row].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')
    _r = np.abs(total[w]).max() * SCALE if j1 > j0 else np.abs(total).max() * SCALE
    axes[row].set_ylim(-3 * _r, 3 * _r)
    row += 1

axes[-1].set_xlabel('Time (us)')
for ax in axes:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

_suptitle = (f'Force at the coupler, positions optimized for a fixed '
            f'c0 = {ARGS.c0:.1f} mm/ms — {CASE}' if ARGS.c0 is not None else
            f'Force at the coupler, reconstructed directly from tape + '
            f'transit-time c0 — {CASE}')
fig.suptitle(_suptitle, x=.006, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .975))
FIG = f'{_STEM}.png'
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'wrote {FIG}')

plotting.show_unless(HEADLESS)
