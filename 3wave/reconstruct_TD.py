"""
Reconstruct the FORCE at the specimen/bar interface using ONLY time-domain
information -- no eta, no attenuation, no dispersion.

    python3 identify_bar_tension.py cases/identifications/tension_bar_2
    python3 reconstruct_TD.py cases/analyses/SHTB_PC [--headless]

`reconstruct_interface.py` solves the two-gauge (or N-gauge) problem in the
Laplace/frequency domain: exp(-eta t), FFT, a phase-domain normal-equations
solve, inverse FFT. That is what makes attenuation alpha(f) and dispersion
c_p(f)/c0 possible to include at all -- they are frequency-domain objects.

This script answers a narrower question -- what does the SAME record give up
if none of that machinery is used at all? `wave_separation.separate_time_domain`
solves the identical two-wave problem as a pure time shift: eliminate one wave
between the two gauges' records, march a causal recursion forward, done. No
FFT, no eta to pick, no alpha(f)/c_p(f) to have identified in the first place.
It is exact for a lossless, non-dispersive bar and NOTHING else -- which is
also exactly what `separate(..., dispersion=None, attenuation=None)` assumes,
so the two methods are two different numerical routes to the same answer
under that assumption, not two different physical models. Comparing them is a
check on the FFT machinery, not a check on whether attenuation/dispersion
matter -- for that, run `reconstruct_interface.py --no-attenuation
--no-dispersion` instead and diff against its default.

--------------------------------------------------------------------------
Why this needs exactly two gauges
--------------------------------------------------------------------------
The time-domain recursion eliminates one unknown wave between two records by
a single time shift; there is no least-squares step to fold a third gauge
into. `separate` handles any number by solving normal equations instead --
that generality is exactly the machinery this script is built to do without.
A bar identified with more than two gauges cannot be reconstructed here; use
reconstruct_interface.py for it (with --no-attenuation --no-dispersion for
the closest lossless comparison).

--------------------------------------------------------------------------
The slider, on BOTH bars
--------------------------------------------------------------------------
As in reconstruct_interface.py, every bar gets its own slider -- moving the
input-bar reconstruction and the output-bar reconstruction independently, off
their respective interfaces, while the force-equilibrium panel (when both bars
are identified) tracks whichever one last moved. Here each move is nearly
free: `separate_time_domain_field` is nothing but linear interpolation into an
already-computed pair of 1-D arrays, where reconstruct_interface.py re-runs
`separate_field`'s FFT-based field synthesis on every step.

--------------------------------------------------------------------------
What the figure shows
--------------------------------------------------------------------------
Same layout as reconstruct_interface.py's, except row 0 of the out column:

    row 0, in column     what was measured
    row 0, out column    P, propagated to gauge out-0's own position, against
                          the RAW signal recorded there -- plotted only, no
                          check: the two are exactly p_x0 + m_x0 = sig[0], so
                          they agree for as long as m_x0 stays negligible and
                          diverge once it doesn't. Where that is is left for
                          the reader to see, not computed here.
    row 1, per bar        F = P + M at THAT bar's own face, with an offset slider
    row 2, left           force equilibrium across the interface, both bars overlaid
    row 2, right          the free-end null, at BAR's own far free surface

except the checks that need a resolved wavefront (causality, tensile
residual, "bars part") are computed and annotated only on BAR, exactly as in
reconstruct_interface.py -- they are properties of that bar's own boundary
conditions, not of which separation method was used to get F.
"""
import argparse

import numpy as np

from wave_separation_code import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('case',
                 help='an identification folder (its own shot) or an analysis '
                      'folder (a specimen shot, with the bars its `bars` '
                      'folder identified)')
_ap.add_argument('--bar', default=None,
                 help='which bar carries the checks (causality, tensile, '
                      'free-end null), when the identification covered more '
                      'than one. Default: the only one, or "out". Both bars '
                      'are reconstructed and get a slider regardless.')
HEADLESS, ARGS = plotting.init(parser=_ap)

from wave_separation_code import cases
from wave_separation_code import config
from wave_separation_code.wave_separation import (
    separate_time_domain, separate_time_domain_field, wavefront_time)

cfg = config.load(ARGS.case)
if cfg['kind'] == 'simulation':
    raise SystemExit(f'{ARGS.case} is a simulation; give an identification '
                     'or an analysis folder')
CASE = cfg['case']
SELF = cfg['kind'] == 'identification'   # its own shot, not a borrowed calibration
BARS_DIR = cases.rel(cases.bars_dir(cfg))
IDENT_FILE = f'{BARS_DIR}/{cases.IDENT_FILE}'
ID = cases.identification(cfg)
BARS = [str(b) for b in ID['bars']]
BAR = ARGS.bar or ('out' if 'out' in BARS else BARS[0])
if BAR not in BARS:
    raise SystemExit(f'{IDENT_FILE} covers {BARS}, not {BAR!r}')

d = cases.record(cfg)
t, dt, N = d['t'], d['dt'], d['N']
sig = list(d[f'eps_{BAR}'])
eta = d['eta']                                  # unused by the solve itself;
                                                 # kept only for the printed
                                                 # header, so a reader can see
                                                 # what the FFT route would
                                                 # have used here
UNITS = d.get('units', 'strain')
SCALE = 1.0 if UNITS != 'strain' else 1e6
USYM = UNITS if UNITS != 'strain' else 'ustrain'

# The fraction of a gauge's own peak that counts as its first arrival, for the
# quiescent lead-in the time-domain recursion needs to seed itself. This is
# the SAME quantity [<case>.trim].threshold already uses to find the real
# arrival for trimming this record -- not a new number invented here -- so a
# case tuned against pickup/noise at the trigger (see SHTB_PC, threshold =
# 0.05) gets that same tolerance in the recursion's own lead-in check.
ARRIVAL_FRAC = float(cfg.get('trim', {}).get('threshold', 0.05))

c0 = float(ID[f'c_{BAR}'])
L = float(ID[f'L_ref_{BAR}'])
R = float(ID[f'R_{BAR}'])                       # 2L/c, MEASURED, in ms
x_id = np.asarray(ID[f'x_{BAR}'], float)
x_tape = (np.asarray(ID[f'tape_{BAR}'], float) if f'tape_{BAR}' in ID.files
          else None)

if len(x_id) != 2:
    raise SystemExit(
        f'{IDENT_FILE} identified {len(x_id)} gauges on the {BAR} bar, but '
        'separate_time_domain needs exactly two: the causal recursion '
        'eliminates one wave between a PAIR of records, with no least-'
        'squares step to fold a third gauge into. Use reconstruct_interface.py '
        '(optionally --no-attenuation --no-dispersion for the closest '
        'lossless comparison) instead.')

# See reconstruct_interface.py for why this has to be READ, not reconstructed
# as L - x: L_ref_{b} is the assembly-wide L_free_ref under
# identify_bar_tension.py's default route, not either bar's own length to its
# own free end.
L_FREE = {}
for _b in BARS:
    if f'L_free_{_b}' in ID.files:
        L_FREE[_b] = np.asarray(ID[f'L_free_{_b}'], float)
    else:
        L_FREE[_b] = (float(ID[f'L_ref_{_b}'])
                      - np.asarray(ID[f'x_{_b}'], float))
        print(f'!! {IDENT_FILE} carries no L_free_{_b}; falling back to '
              f'L_ref - x, which\n!! is only that bar\'s free-end distance if '
              'L_ref is its OWN length. Re-run the\n!! identification to get '
              'the identified value.')

# See reconstruct_interface.py: on the SHTB the input bar's own far end is
# the anvil, reached only through whatever sat at the interface during the
# calibration shot -- not a path a specimen shot preserves.
NULL_VALID = not (BAR == 'in' and not SELF)

IMPACT = str(cfg.get('interface', 'impact')) == 'impact'
IFACE = ('impact interface' if IMPACT else
         f'{BAR}put-bar / specimen interface')
# See reconstruct_interface.py's checks() comment: UNILATERAL (F >= 0) assumes
# a dry contact, false of a bonded tension joint.
TENSION = str(d.get('loading', cfg.get('loading'))) == 'tension'

_L_cfg = float(cfg.get('bar', {}).get('length', L))
if abs(_L_cfg - L) > 1.0:
    raise SystemExit(
        f'{IDENT_FILE} identified a {L:.1f} mm bar but [{CASE}] describes one '
        f'{_L_cfg:.1f} mm long.\nThose are not the same bar. Re-run the '
        'identification for this rig, or fix the case.')


# --------------------------------------------------------------------------
# the OTHER bar, when the identification covered both
# --------------------------------------------------------------------------
OTHER = next((b for b in BARS if b != BAR), None)
if OTHER is not None:
    sig_o = list(d[f'eps_{OTHER}'])
    c0_o = float(ID[f'c_{OTHER}'])
    x_o = np.asarray(ID[f'x_{OTHER}'], float)
    if len(x_o) != 2:
        raise SystemExit(
            f'{IDENT_FILE} identified {len(x_o)} gauges on the {OTHER} bar, '
            'but separate_time_domain needs exactly two. Use '
            'reconstruct_interface.py instead.')

print(__doc__.split('---')[0].strip())
print(f'\nrecord     : {d.get("source", "dump.npz")}')
print(f'bar        : {BAR}, {L:.1f} mm, c0 = {c0:.2f} mm/ms, '
      f'2L/c = {R*1e3:.1f} us (measured)')
print(f'signals    : {len(sig)} gauges in {UNITS}, eta (FFT route only) = '
      f'{eta:g} /ms')
print(f'x = 0 is   : the {IFACE}')
if not SELF:
    print(f'reusing    : c0 and positions identified in '
          f'{BARS_DIR}.\n             Nothing is identified from '
          'THIS record -- they are properties of the bar.')
print('method     : separate_time_domain -- a time shift between two '
      'gauges, no FFT')
print('attenuation: ignored by construction -- this method assumes a '
      'lossless bar')
print('dispersion : ignored by construction -- this method assumes '
      'c_p = c0 at every frequency')
print(f'arrival    : {ARRIVAL_FRAC:g} of peak counts as first arrival, for '
      'the recursion\'s own quiescent lead-in check '
      '([{}].trim.threshold)'.format(CASE))


# --------------------------------------------------------------------------
# the reconstruction, and the checks on it
# --------------------------------------------------------------------------
def reconstruct(x, gsig=sig, gc0=c0):
    """P, M and F = P + M at the contact plane, purely from a time shift."""
    p, m = separate_time_domain(t, gsig, x, gc0, arrival_frac=ARRIVAL_FRAC)
    return p, m, p + m


def _echo_time(p):
    """When the free-end echo reaches x = 0. See reconstruct_interface.py."""
    return wavefront_time(t, p) + R


def _echo_rise(m, t_echo):
    """10-90 rise of the free-end echo. See reconstruct_interface.py."""
    a = np.abs(m)
    i_R = int(np.searchsorted(t, t_echo))
    lo, hi = i_R, min(len(a), int(np.searchsorted(t, t_echo + 0.5)))
    if hi - lo < 4:
        return 3.0 * float(np.mean(np.diff(t))), False
    top = float(np.max(a[lo:hi]))
    j90 = lo + int(np.argmax(a[lo:hi] > 0.9 * top))
    floor = int(np.searchsorted(t, t_echo - R))
    j10 = j90
    while j10 > floor and a[j10] > 0.1 * top:
        j10 -= 1
    return (max(float(t[j90] - t[j10]), 3.0 * float(np.mean(np.diff(t)))),
            j10 > floor)


def checks(p, m, F):
    """The three boundary conditions the contact plane itself provides.

    Identical to reconstruct_interface.py's -- these are properties of the
    bar's own boundary conditions, not of the separation method.
    """
    amp = float(np.abs(p).max())
    t_echo = _echo_time(p)
    echo_in_record = t_echo <= float(t[-1])
    i_on = int(np.searchsorted(t, t_echo - R))
    _hi = int(np.searchsorted(t, t_echo))
    _a = np.abs(p[:_hi]) if _hi > i_on + 2 else np.abs(p)
    _pk = float(_a.max())
    _j90 = int(np.argmax(_a > 0.9 * _pk))
    _j10 = int(np.argmax(_a > 0.1 * _pk))
    rise_p = float(t[_j90] - t[_j10])
    rise, rise_ok = _echo_rise(m, t_echo)
    i_pre = int(np.searchsorted(t, t_echo - rise))
    i_sep = int(np.searchsorted(t, t_echo + rise))
    i_end = int(0.95 * N)
    i_ten = i_sep if IMPACT else i_end
    causality = (float(np.abs(m[i_on:i_pre]).max() / amp)
                 if echo_in_record and i_pre > i_on + 2 else float('nan'))
    return dict(
        amp=amp, i_on=i_on, i_pre=i_pre, i_sep=i_sep, i_end=i_end, rise=rise,
        causality_ok=(echo_in_record and i_pre > i_on + 2),
        echo_in_record=echo_in_record, rise_ok=rise_ok,
        rise_p=rise_p,
        causality=causality,
        tensile=(float('nan') if TENSION else
                 float(max(0.0, -F[i_on:i_ten].min()) / amp)),
        after=(float(np.sqrt(np.mean(F[i_sep:i_end] ** 2)) / amp)
               if IMPACT and i_end > i_sep else float('nan')),
        peak=float(F.max()),
        t_echo=float(t_echo * 1e3),
        t_open=float(t[min(i_sep, N - 1)] * 1e3),
    )


def free_end(x):
    """The free-end null, purely from a time shift. See reconstruct_interface.py."""
    lf = L_FREE[BAR] + (x_id - np.asarray(x, float))
    p, m = separate_time_domain(t, sig, lf, c0, arrival_frac=ARRIVAL_FRAC)
    tot, amp = p + m, float(np.abs(p).max())
    w = slice(int(np.argmax(np.abs(p) > 0.02 * amp)),
              int(float(cfg.get('null', {}).get('window', 0.75)) * N))
    return dict(p=p, m=m, tot=tot, amp=amp, w=w, L_free=lf,
                rms=float(np.sqrt(np.mean(tot[w] ** 2)) / amp),
                max=float(np.abs(tot[w]).max() / amp))


SETS = [('identified', x_id)]
if x_tape is not None:
    SETS.append(('tape', x_tape))
RES = {}
NULL = {}
for name, x in SETS:
    p, m, F = reconstruct(x)
    RES[name] = dict(x=x, p=p, m=m, F=F, **checks(p, m, F))
    NULL[name] = free_end(x)
    RES[name]['null'] = NULL[name]['rms']
NULL_TOL = float(cfg.get('null', {}).get('tol', cfg.get('null_tol', 5.0e-3)))

if OTHER is not None:
    p_o, m_o, F_o = reconstruct(x_o, gsig=sig_o, gc0=c0_o)
    _r0 = RES[SETS[0][0]]
    _peak_eq = float(np.abs(_r0['F']).max())
    equilibrium = np.abs(_r0['F'] - F_o) / (_peak_eq if _peak_eq > 0 else 1.0)
    _win = slice(_r0['i_on'] if _r0['echo_in_record'] else 0, _r0['i_end'])

F_BY = {BAR: RES[SETS[0][0]]['F']}
if OTHER is not None:
    F_BY[OTHER] = F_o
DAT_BARS = [b for b in ('in', 'out') if b in F_BY]

# --------------------------------------------------------------------------
# out-0: the pure outgoing wave P, propagated from the interface to gauge
# out-0's own position, against the RAW signal that gauge actually recorded.
#
# `separate_time_domain_field` shifts P and M each by their own correctly
# signed delay, so this is exact -- not a plain shift of F = P + M (which
# would wrongly carry M's interface-side value along at P's delay). p_at_x0
# is pure incident wave, with no reflection; the raw signal sig[0] is
# p_at_x0 + m_at_x0. The two are identical for as long as m_at_x0 is
# negligible, and diverge once it isn't -- nothing here decides when that
# is, it is just plotted.
#
# Before their own first significant rise, out-0 and out-1 carry nothing but
# pickup noise -- the recursion has no way to know that and just marches
# forward on it, so P (and this comparison) inherits it too. Zeroing each
# channel below its own arrival (same ARRIVAL_FRAC used everywhere else in
# this script) before feeding it to the recursion removes that: with both
# inputs exactly 0 before they rise, P and M built from them are exactly 0
# there too, so p_at_x0 and out-0's own (zeroed) record agree to the bit
# until out-0 actually rises -- not just approximately, given a clean
# record. out-1 is plotted alongside, zeroed the same way, purely so the
# zeroing itself can be seen to land where it should: flat at 0 up to
# out-1's own (later, more distant) rise, not before or after it.
# --------------------------------------------------------------------------
def _zero_lead_in(s, frac):
    """`s`, with everything before its own first crossing of frac * peak
    forced to exactly 0. `frac` is ARRIVAL_FRAC -- the SAME quantity the
    causal recursion already uses to find this signal's own quiescent-lead-
    in boundary (see `_time_domain_pm`), not a new threshold invented here.
    """
    s = np.asarray(s, float)
    i0 = int(np.argmax(np.abs(s) > frac * np.abs(s).max()))
    out = s.copy()
    out[:i0] = 0.0
    return out


OUT0 = None
if BAR == 'out':
    _x0, _x1 = float(x_id[0]), float(x_id[1])
    _sig0z = _zero_lead_in(sig[0], ARRIVAL_FRAC)
    _sig1z = _zero_lead_in(sig[1], ARRIVAL_FRAC)
    _p_x0, _m_x0 = separate_time_domain_field(t, [_sig0z, _sig1z], x_id, c0,
                                              [_x0], arrival_frac=ARRIVAL_FRAC)
    OUT0 = dict(p=_p_x0[0], m=_m_x0[0], sig0=_sig0z, sig1=_sig1z,
               x0=_x0, x1=_x1)

print('\n--- interface force, and the checks that need no ground truth '
      '-------')
_last = f'{"after sep":>11}' if IMPACT else ''
print(f'{"positions":>11} {"D [mm]":>8} {"peak F":>9} {"free-end":>10} '
      f'{"causality":>11} {"tensile":>9}' + _last)
print(f'{"":>11} {"":>8} {"["+UNITS+"]":>9} {"null rms":>10} '
      f'{"M before":>11} {"F < 0":>9}' + (f'{"F rms":>11}' if IMPACT else ''))
for name, x in SETS:
    r = RES[name]
    Dg = abs(x[1] - x[0]) if len(x) > 1 else float('nan')
    _cau = f'{r["causality"]:11.3f}' if r['causality_ok'] else f'{"n/a":>11}'
    _ten = f'{"n/a":>9}' if TENSION else f'{r["tensile"]:9.3f}'
    print(f'{name:>11} {Dg:8.2f} {r["peak"]:9.3f} {r["null"]:10.2e} '
          + _cau + ' ' + _ten
          + (f'{r["after"]:11.3f}' if IMPACT else ''))
print('the right-hand columns are fractions of peak |P|. Zero is the '
      'ideal;\nwhat is left is model error, and its SIGN is known -- a contact '
      'that pulls or a\nwave that arrives early is not a measurement, it is the '
      'residual.')
if not NULL_VALID:
    print(f'the free-end column is NOT a check on this run: the {BAR} bar '
          'reaches the free\nend through the interface, and this is not the '
          'shot the calibration measured\nthat path on. Reported, not to be '
          'read as PASS or FAIL.')
else:
    print(f'the free-end null uses the IDENTIFIED distances to the {BAR} bar\'s '
          f'own free\nsurface, {L_FREE[BAR].min():.0f}-{L_FREE[BAR].max():.0f} '
          f'mm; threshold for this case is {NULL_TOL:.1e}.')
_r0 = RES[SETS[0][0]]
_T0 = float(d.get('t0_file', 0.0))
if _r0['echo_in_record']:
    print(f'the echo reaches x = 0 at {_r0["t_echo"] + _T0:.0f} us (source-file '
          'base) -- one round trip\nafter the wave LEFT it, not after the '
          'record started.')
else:
    print('the echo-timing search did NOT resolve on this record: P keeps '
          'rising past its\ninitial edge -- see reconstruct_interface.py for '
          'why. Only causality below is\naffected; the reconstruction itself '
          'does not depend on it.')
if _r0['rise_ok']:
    print(f'its own 10-90 rise there measures {_r0["rise"]*1e3:.0f} us after '
          f'crossing 2L = {2*L:.0f} mm of\nbar, and that is the clearance '
          'held either side.')
else:
    print('its edge could NOT be measured: |M| never falls back to 10 % of the '
          'echo peak\nbetween the wavefront and the echo, so the walk-back hit '
          'its bound. The\nclearance is that bound and the causality check is '
          'reported n/a above.')

if len(SETS) > 1:
    a, b = RES[SETS[0][0]], RES[SETS[1][0]]
    off = float(np.mean(a['x'] - b['x']))
    dpk = abs(a['peak'] - b['peak']) / max(abs(a['peak']), abs(b['peak']))
    rel = float(np.sqrt(np.mean((a['F'] - b['F']) ** 2))
                / max(np.abs(a['F']).max(), np.abs(b['F']).max()))
    print(f'\nthe two sets differ by a COMMON {off:+.2f} mm '
          f'({abs(a["x"][1]-a["x"][0]) - abs(b["x"][1]-b["x"][0]):+.2f} mm in D). '
          f'The forces they give\ndiffer by {dpk*100:.1f} % in peak and '
          f'{rel:.2e} relative L2.')

if OTHER is not None:
    print(f'\n--- force equilibrium across the interface, {BAR} vs {OTHER} bar '
          '-------')
    print(f'{OTHER} bar    : c0 = {c0_o:.2f} mm/ms, x = '
          f'[{", ".join(f"{v:.1f}" for v in x_o)}] mm ({len(sig_o)} gauges), '
          'time domain, lossless by construction')
    print(f'|F_{BAR}-F_{OTHER}|/max|F_{BAR}| : mean {equilibrium[_win].mean():.4e}, '
          f'max {equilibrium[_win].max():.4e}, over {t[_win.start]:.3f}-'
          f'{t[min(_win.stop, N-1)]:.3f} ms')
    print('a genuine joint carries the same force on both sides; what is left '
          'is model\nerror -- most likely a small extra transit time through '
          'the coupler that this\nreconstruction, treating each bar as if it '
          'ran straight to the interface, does\nnot account for.')

print('\n--- what the reconstruction says happened '
      '--------------------------------')
r = RES[SETS[0][0]]
print(f'  force at the {IFACE} peaks at {r["peak"]:.3f} {UNITS}')
if IMPACT:
    print(f'  free-end echo reaches the contact at {r["t_echo"]:.0f} us; the '
          f'echo is\n    TENSILE, the contact cannot carry it, and the bars part')
    print(f'  after {r["t_open"]:.0f} us the reconstructed force is '
          f'{r["after"]*100:.1f} % of peak -- i.e. zero')
else:
    print(f'  there is no sharp wavefront at all: the force at x = 0 takes '
          f'{r["rise_p"]*1e3:.0f} us to go\n    from 10 % to 90 % here')
    if r['causality_ok']:
        print(f'  nothing returns to x = 0 until the free-end echo at '
              f'{r["t_echo"]:.0f} us, and M holds\n    to {r["causality"]:.3f} '
              'of peak before it')
    else:
        print('  the causality check DOES NOT APPLY to this record (see above)')
    print('  the bars never part here, so there is no "after separation" to '
          'score; F >= 0\n    is checked over the WHOLE record instead')


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init
from matplotlib.widgets import Slider

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'
tt = t * 1e3
T0_FIG = float(d.get('t0_file', 0.0))
tt_f = tt + T0_FIG

PANEL = {BAR: dict(sig=sig, x=x_id, c0=c0, p=RES[SETS[0][0]]['p'],
                   m=RES[SETS[0][0]]['m'], primary=True)}
if OTHER is not None:
    PANEL[OTHER] = dict(sig=sig_o, x=x_o, c0=c0_o, p=p_o, m=m_o, primary=False)
COLS = [b for b in ('in', 'out') if b in PANEL]
NULL_COL = COLS.index(BAR)
EQ_COLI = next((i for i, b in enumerate(COLS) if b != BAR), None)

fig, axes = plt.subplots(3, len(COLS), figsize=(max(11.0, 9.5 * len(COLS)), 12),
                         sharex=True, squeeze=False)
fig.patch.set_facecolor(SURFACE)

# current_F/current_delta are the only state a slider callback needs beyond
# its own column: every slider writes into these, and whichever moved last is
# what the equilibrium panel (shared across columns) reflects.
current_F = dict(F_BY)
current_delta = {b: 0.0 for b in COLS}
COLINFO = {}

for col, bname in enumerate(COLS):
    pnl = PANEL[bname]

    # --- what went in, OR (out bar) out-0: reconstructed P vs raw signal ---
    ax0 = axes[0, col]
    if bname == 'out' and OUT0 is not None:
        ax0.plot(tt_f, OUT0['sig1'] * SCALE, lw=.9, color=MUTED,
                 label=f'signal out-1, measured at {OUT0["x1"]:.0f} mm '
                       '(lead-in zeroed)')
        ax0.plot(tt_f, OUT0['sig0'] * SCALE, lw=1.1, color=INK,
                 label=f'signal out-0, measured at {OUT0["x0"]:.0f} mm '
                       '(lead-in zeroed)')
        ax0.plot(tt_f, OUT0['p'] * SCALE, lw=1.0, color=BLUE, ls='--',
                 label='P at out-0 (pure outgoing wave)')
        ax0.plot(tt_f, OUT0['m'] * SCALE, lw=1.0, color=ORANGE, ls='--',
                 label='M at out-0 (pure returning wave)')
        ax0.set_ylabel(f'Force ({UNITS})')
        ax0.set_title('P, M at out-0 vs the raw signal there, both lead-in '
                      'zeroed', loc='left', fontsize=11)
        _g = max(np.abs(OUT0['sig0']).max(), np.abs(OUT0['sig1']).max(),
                 np.abs(OUT0['p']).max(), np.abs(OUT0['m']).max()) * SCALE
        ax0.set_ylim(-1.3 * _g, 1.35 * _g)
        ax0.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')
    else:
        for k, s in enumerate(pnl['sig']):
            ax0.plot(tt_f, s * SCALE, lw=.9, color=(BLUE, ORANGE, INK)[k % 3],
                     label=f'gauge {k} at {pnl["x"][k]:.0f} mm (identified)')
        ax0.set_ylabel(f'Gauge signal ({USYM})')
        ax0.set_title(f'What was measured — {len(pnl["sig"])} gauges on the '
                      f'{bname} bar', loc='left', fontsize=11)
        _g = max(np.abs(s_).max() for s_ in pnl['sig']) * SCALE
        ax0.set_ylim(-1.3 * _g, 1.35 * _g)
        ax0.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

    # --- F = P+M at that bar's own face, EVERY column gets P, M and F ------
    ax1 = axes[1, col]
    line_p, = ax1.plot(tt_f, pnl['p'] * SCALE, lw=.8, color=BLUE, alpha=.6,
                       label='$P$')
    line_m, = ax1.plot(tt_f, pnl['m'] * SCALE, lw=.8, color=ORANGE, alpha=.6,
                       label='$M$')
    if pnl['primary']:
        line_F = None
        for (name, _), col_c, ls in zip(SETS, (INK, BLUE), ('-', '--')):
            (_line,) = ax1.plot(tt_f, RES[name]['F'] * SCALE, color=col_c,
                                lw=1.0, ls=ls,
                                label=f'$F = P + M$, {name} positions')
            if name == SETS[0][0]:
                line_F = _line
    else:
        (line_F,) = ax1.plot(tt_f, F_BY[bname] * SCALE, color=INK, lw=1.0,
                             label='$F = P + M$, identified positions')
    ax1.axhline(0, color=GRID, lw=1.0)
    band = 0.03 * r['amp'] * SCALE
    ax1.axhspan(-band, band, color=BLUE, alpha=.15,
                label='±3 % of peak $|P|$')
    title = f'F = P + M at the {bname}put-bar/specimen interface (time domain)'
    if pnl['primary']:
        ax1.axvline(r['t_echo'] + T0_FIG, color=MUTED, lw=1.1, ls='--')
        ax1.annotate('free-end echo arrives, $2L/c$ after the wave left  ',
                     (r['t_echo'] + T0_FIG, 0), fontsize=9, color=MUTED,
                     va='bottom', ha='right')
        for _b in (r['t_echo'] + T0_FIG - r['rise'] * 1e3,
                   r['t_echo'] + T0_FIG + r['rise'] * 1e3):
            ax1.axvline(_b, color=GRID, lw=1.0, ls=':')
        if IMPACT:
            ax1.axvline(r['t_open'] + T0_FIG, color=ORANGE, lw=1.1, ls='--')
            ax1.annotate('  bars part', (r['t_open'] + T0_FIG,
                                         r['peak'] * SCALE * .6),
                         fontsize=9, color=MUTED)
        title += ('' if TENSION else
                  f' — cannot go negative: residual {r["tensile"]:.3f} '
                  'of peak')
        if r['causality_ok']:
            title += f', causality {r["causality"]:.3f}'
    ax1.set_ylabel(f'Interface force ({UNITS})')
    ax1.set_title(title, loc='left', fontsize=10)
    ax1.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

    COLINFO[bname] = dict(ax=ax1, bar=bname, sig=pnl['sig'], x=pnl['x'],
                          c0=pnl['c0'], line_p=line_p, line_m=line_m,
                          line_F=line_F, title_base=title)

# --- bottom left: force equilibrium across the interface -------------------
EQ = None
if EQ_COLI is not None:
    axeq = axes[2, EQ_COLI]
    _eq_lines = {}
    for _b, _c in (('in', BLUE), ('out', ORANGE)):
        (_eq_lines[_b],) = axeq.plot(
            tt_f, F_BY[_b] * SCALE, lw=1.0, color=_c,
            label=f'$F_{{{_b}}}$ at the {_b}put-bar / specimen face')
    (_eq_diff,) = axeq.plot(tt_f, (F_BY['in'] - F_BY['out']) * SCALE,
                            color=INK, lw=1.1,
                            label='$F_{in} - F_{out}$ (should be 0)')
    axeq.axhline(0, color=GRID, lw=1.0)
    axeq.axvspan(tt_f[_win.start], tt_f[min(_win.stop, N - 1)], color=GRID,
                 alpha=.35, label='mean/max window')
    axeq.set_xlabel('Time (us)')
    axeq.set_ylabel(f'Interface force ({UNITS})')

    def _eq_title(mean, max_, deltas):
        sfx = ''.join(f', {b} face {d:+.0f} mm off interface'
                     for b, d in deltas.items() if d != 0.0)
        return ('Force equilibrium across the specimen — two independent '
               f'solves of one force:\nmean {mean:.2e}, max {max_:.2e} of '
               f'peak $|F_{{{BAR}}}|${sfx}')

    axeq.set_title(_eq_title(equilibrium[_win].mean(), equilibrium[_win].max(),
                             current_delta),
                   loc='left', fontsize=10)
    axeq.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')
    EQ = dict(ax=axeq, line_in=_eq_lines['in'], line_out=_eq_lines['out'],
              line_diff=_eq_diff, title_fn=_eq_title)

# --- one offset slider PER COLUMN -------------------------------------------
# separate_time_domain_field is a pair of np.interp calls into already-
# computed arrays, so unlike reconstruct_interface.py's single FFT-backed
# slider there is no cost to giving every bar its own.
_sliders = []
for bname, info in COLINFO.items():
    sax = info['ax'].inset_axes((0.60, 0.88, 0.38, 0.07))
    sax.set_facecolor(SURFACE)
    slider = Slider(sax, 'x off interface (mm)', -200.0, 50.0, valinit=0.0,
                    valstep=1.0, color=BLUE)
    slider.label.set_fontsize(8); slider.label.set_color(MUTED)
    slider.valtext.set_fontsize(8); slider.valtext.set_color(MUTED)

    def _make_handler(info=info, bname=bname):
        def _on_offset(val):
            delta = float(val)
            p_f, m_f = separate_time_domain_field(
                t, info['sig'], info['x'], info['c0'], [delta],
                arrival_frac=ARRIVAL_FRAC)
            p1, m1 = p_f[0], m_f[0]
            F1 = p1 + m1
            info['line_p'].set_ydata(p1 * SCALE)
            info['line_m'].set_ydata(m1 * SCALE)
            info['line_F'].set_ydata(F1 * SCALE)
            where = 'into the bar' if delta > 0 else 'towards/through the specimen'
            suffix = f'  [{delta:+.0f} mm off the interface, {where}]' if delta else ''
            info['ax'].set_title(info['title_base'] + suffix, loc='left',
                                 fontsize=10)

            if EQ is not None:
                current_F[bname] = F1
                current_delta[bname] = delta
                diff = current_F['in'] - current_F['out']
                EQ['line_in'].set_ydata(current_F['in'] * SCALE)
                EQ['line_out'].set_ydata(current_F['out'] * SCALE)
                EQ['line_diff'].set_ydata(diff * SCALE)
                eq_frac = np.abs(diff) / (_peak_eq if _peak_eq > 0 else 1.0)
                EQ['ax'].set_title(EQ['title_fn'](eq_frac[_win].mean(),
                                                  eq_frac[_win].max(),
                                                  current_delta),
                                   loc='left', fontsize=10)
            fig.canvas.draw_idle()
        return _on_offset

    slider.on_changed(_make_handler())
    _sliders.append(slider)             # keep a reference; Slider needs one

# --- bottom right: the free-end null ---------------------------------------
axnull = axes[2, NULL_COL]
_nl = NULL[SETS[0][0]]
axnull.plot(tt_f, _nl['p'] * SCALE, lw=.9, color=BLUE, label='$P$')
axnull.plot(tt_f, _nl['m'] * SCALE, lw=.9, color=ORANGE, label='$M$')
axnull.plot(tt_f, _nl['tot'] * SCALE, color=INK, lw=1.1,
            label='$P + M$ (should be 0)')
axnull.axhline(0, color=GRID, lw=1.0)
axnull.axvspan(tt_f[_nl['w'].start], tt_f[min(_nl['w'].stop, N - 1)],
               color=GRID, alpha=.35, label='rms/max window')
axnull.set_xlabel('Time (us)')
axnull.set_ylabel(f'Stress at free end ({USYM})')
_verdict = (f'rms {_nl["rms"]:.2e} vs tol {NULL_TOL:.1e} -> '
            f'{"PASS" if _nl["rms"] <= NULL_TOL else "FAIL"}' if NULL_VALID else
            f'rms {_nl["rms"]:.2e} — NOT A CHECK: the {BAR} bar reaches the '
            'free end through the\ninterface, and this is not the shot the '
            'calibration measured that path on')
axnull.set_title(
    f'Free-end null (time domain): stress at the {BAR} bar\'s free surface, '
    f'{_nl["L_free"].min():.0f}-{_nl["L_free"].max():.0f} mm away —\n'
    + _verdict, loc='left', fontsize=10)
axnull.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

axes[0, 0].set_xlim(tt_f[0], tt_f[int(0.95 * N)])

if OTHER is not None:
    _Din = abs(PANEL['in']['x'][1] - PANEL['in']['x'][0])
    _Dout = abs(PANEL['out']['x'][1] - PANEL['out']['x'][0])
    _title = (f'Force at each bar\'s own interface, TIME DOMAIN ONLY — in: '
              f'{_Din:.0f} mm gauge spacing, out: {_Dout:.0f} mm'
              + ('' if SELF
                 else f'\ncalibrated on {BARS_DIR} — nothing is '
                      'identified from this record'))
else:
    _title = (f'Force at the {IFACE}, TIME DOMAIN ONLY, from two gauges '
              f'{abs(x_id[1]-x_id[0]):.0f} mm apart on the {BAR} bar'
              + ('' if SELF
                 else f'\ncalibrated on {BARS_DIR} — nothing is '
                      'identified from this record'))

for ax in axes.flat:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.suptitle(_title, x=.006, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .975))
_stem = cases.output(cfg, 'interface_force_TD')
FIG = f'{_stem}.png'
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'\nwrote {cases.rel(FIG)}')

DAT = f'{_stem}.dat'
T0 = float(d.get('t0_file', 0.0))
_cols = [tt + T0] + [F_BY[b] for b in DAT_BARS]
_header = 'time[us]  ' + '  '.join(
    f'F_{b}[{UNITS}]' for b in DAT_BARS)
_geom = '\n'.join(
    f'F_{b}: {b} bar, c0={float(ID[f"c_{b}"]):.3f} mm/ms, '
    f'x={[round(float(v), 2) for v in np.asarray(ID[f"x_{b}"], float)]} mm, '
    f'reconstructed at ITS OWN face (x=0), time domain only' for b in DAT_BARS)
np.savetxt(DAT, np.column_stack(_cols),
           header=_header + '\n'
                  f'time is the SOURCE FILE\'s own base (analysis t=0 sits at '
                  f'{T0:.1f} us there)\n'
                  + _geom + f'\nx=0 is the {IFACE}; no eta, no attenuation, '
                  'no dispersion')
print(f'wrote {cases.rel(DAT)}: {", ".join(["time"] + [f"F_{b}" for b in DAT_BARS])}')

plotting.show_unless(HEADLESS)
