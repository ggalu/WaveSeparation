"""
Reconstruct the FORCE at the impact interface, and check it against the physics
the rig itself guarantees.

    python3 identify_bar_compression.py --experiment experiment_pc_bar
    python3 reconstruct_interface.py [--headless]

This is what the calibration was for. `separate` puts the two travelling waves
at x = 0 -- the plane where the bars touch and no gauge can go -- and their sum
is the contact force:

    F(t) = P(t) + M(t)

--------------------------------------------------------------------------
No E, no A, no density
--------------------------------------------------------------------------
The gauge records are FORCE in kN and `separate` is linear, so P and M come back
in kN and F is the interface force outright. E*A never enters and neither does
rho. That matters: E and A are the numbers a rig knows worst, and the answer
does not depend on them. Only c0, the gauge positions and eta do -- and those
are exactly what the calibration shot measures.

--------------------------------------------------------------------------
Four checks, and three of them need no ground truth
--------------------------------------------------------------------------
A simulated shot can be checked against the simulator. A real one cannot be
checked against anything -- except itself. This rig offers three boundary
conditions that hold whatever the bar is made of:

  FREE END      eps_+ + eps_- = 0 at the far surface, at all times. The one the
                calibration scripts already run.
  CAUSALITY     M = 0 at the interface until the free-end echo can get back,
                i.e. for t < 2L/c. Nothing is travelling toward the contact
                before then, and if the reconstruction says otherwise it is
                leaking P into M.
  UNILATERAL    F >= 0 always. The bars are pressed together, not glued; a
                contact cannot pull. Any tensile excursion is model error with
                a known sign.
  SEPARATION    F -> 0 once the tensile echo reaches the contact, and stays
                there. The bars have parted and there is nothing left to
                transmit.

The last three are on the plane actually being reconstructed, which the
free-end null is not, and they are INDEPENDENT of it -- the contact and the far
surface are different boundaries. That is what makes it honest to fit the
attenuation against gauge magnitudes, screen it on the free end, and confirm it
here.

--------------------------------------------------------------------------
Two position sets, side by side
--------------------------------------------------------------------------
Both are run and both are reported, because on the PC shot they differ. The
identification recovers the gauge SPACING to 0.9 mm but puts both positions
~9 mm further from the impact face than the tape does -- a COMMON offset, from
the contact-end reflection at 2L/c not being an ideal free surface while the
striker is still in contact. Theory says a common offset is benign and the
spacing is what matters; running both turns that from a claim into a number.
"""
import argparse

import numpy as np

import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--case', default=None,
                 help='config case to reconstruct; default is whichever one '
                      'bar_identified.npz was written from.')
_ap.add_argument('--bar', default=None,
                 help='which bar, when the identification covered more than '
                      'one. Default: the only one, or "out".')
_ap.add_argument('--no-attenuation', action='store_true',
                 help='ignore the identified alpha(f) and reconstruct with a '
                      'lossless bar, for comparison.')
HEADLESS, ARGS = plotting.init(parser=_ap)

import config
from wave_separation import separate

IDENT_FILE = 'bar_identified.npz'

try:
    ID = np.load(IDENT_FILE, allow_pickle=True)
except FileNotFoundError:
    raise SystemExit(
        f'{IDENT_FILE} not found. Run the identification first:\n'
        '    python3 identify_bar_compression.py --experiment experiment_pc_bar')

CASE = ARGS.case or str(ID['case'])
BARS = [str(b) for b in ID['bars']]
BAR = ARGS.bar or ('out' if 'out' in BARS else BARS[0])
if BAR not in BARS:
    raise SystemExit(f'{IDENT_FILE} covers {BARS}, not {BAR!r}')

if CASE in config.EXPERIMENT_CASES:
    from experiment import load_experiment
    d = load_experiment(CASE)
else:
    from dump import load_dump
    d = load_dump()
cfg = config.load(CASE)
t, dt, N = d['t'], d['dt'], d['N']
sig = list(d[f'eps_{BAR}'])
eta = d['eta']
UNITS = d.get('units', 'strain')
SCALE = 1.0 if UNITS != 'strain' else 1e6
USYM = UNITS if UNITS != 'strain' else 'ustrain'

c0 = float(ID[f'c_{BAR}'])
L = float(ID[f'L_ref_{BAR}'])
R = float(ID[f'R_{BAR}'])                       # 2L/c, MEASURED, in ms
x_id = np.asarray(ID[f'x_{BAR}'], float)
x_tape = (np.asarray(ID[f'tape_{BAR}'], float) if f'tape_{BAR}' in ID.files
          else None)

# What sits at x = 0. The calibration shot has the two bars touching directly;
# a specimen shot has something in between. Only the REPORTING depends on it --
# the solve is identical, because `separate` was never told about boundaries.
IMPACT = str(cfg.get('interface', 'impact')) == 'impact'
IFACE = ('impact interface' if IMPACT else
         f'{BAR}put-bar / specimen interface')

# The identified numbers belong to a BAR. Reconstructing a different shot with
# them is the whole point of calibrating -- but it is only valid on the SAME
# bar, and a length mismatch is the cheap way to catch a case pointed at the
# wrong one.
_L_cfg = float(cfg.get('bar', {}).get('length', L))
if abs(_L_cfg - L) > 1.0:
    raise SystemExit(
        f'{IDENT_FILE} identified a {L:.1f} mm bar but [{CASE}] describes one '
        f'{_L_cfg:.1f} mm long.\nThose are not the same bar. Re-run the '
        'identification for this rig, or fix the case.')

ATT = None
if f'alpha_{BAR}' in ID.files and not ARGS.no_attenuation:
    ATT = (np.asarray(ID[f'alpha_f_{BAR}'], float),
           np.asarray(ID[f'alpha_{BAR}'], float))

print(__doc__.split('---')[0].strip())
print(f'\nrecord     : {d.get("source", "dump.npz")}')
print(f'bar        : {BAR}, {L:.1f} mm, c0 = {c0:.2f} mm/ms, '
      f'2L/c = {R*1e3:.1f} us (measured)')
print(f'signals    : {len(sig)} gauges in {UNITS}, eta = {eta:g} /ms')
print(f'x = 0 is   : the {IFACE}')
if CASE != str(ID['case']):
    print(f'reusing    : c0, positions and alpha(f) identified on '
          f'[{str(ID["case"])}].\n             Nothing is identified from THIS '
          'record -- they are properties of the bar.')
print(f'attenuation: ' + ('lossless (--no-attenuation)' if ARGS.no_attenuation
                          else 'none identified' if ATT is None else
                          f'alpha(f) up to {ATT[0][-1]:.0f} kHz, '
                          f'{ATT[1][-1]:.2e} /mm there'))


# --------------------------------------------------------------------------
# the reconstruction, and the checks on it
# --------------------------------------------------------------------------
def reconstruct(x):
    """P, M and F = P + M at the contact plane, for one set of gauge positions."""
    p, m = separate(t, sig, x, c0=c0, eta=eta, attenuation=ATT)
    return p, m, p + m


def _echo_time(p):
    """
    When the free-end echo reaches x = 0: one round trip after the wave LEFT it.

    `R = 2L/c` is a DELAY, not an instant, and the two coincide only when the
    record happens to start at the moment of loading. On the calibration shot it
    very nearly does, because the loader trims to just ahead of the first
    arrival. On the specimen shot it does not at all -- that record is kept whole
    from -1638 us and the wave does not leave x = 0 until ~2150 us into it, so
    measuring R from t[0] would look for the echo 1.5 round trips early.

    P is reconstructed AT x = 0, so P's own onset is the moment the wave left.
    """
    amp = float(np.abs(p).max())
    return float(t[int(np.argmax(np.abs(p) > 0.02 * amp))]) + R


def _echo_rise(m, t_echo):
    """
    10-90 rise of the free-end echo, measured on M itself, in the units of t.

    This sets the clearance every window below leaves around 2L/c, and it has to
    be MEASURED rather than picked. In aluminium the echo arrives as a step and
    a few microseconds would do. In polycarbonate it has crossed 2L = 2054 mm of
    lossy bar and its edge is spread over ~230 us; a clearance shorter than that
    puts the echo's own leading edge inside the "before the echo" window and
    reports it as a causality violation -- 0.198 against the 0.049 that is
    actually there.

    The plateau is taken after the arrival and the rise is walked back from it,
    so nothing about the striker or the bar length enters.
    """
    a = np.abs(m)
    i_R = int(np.searchsorted(t, t_echo))
    lo, hi = i_R, min(len(a), int(np.searchsorted(t, t_echo + 0.5)))
    if hi - lo < 4:
        return 3.0 * float(np.mean(np.diff(t)))
    top = float(np.max(a[lo:hi]))
    j90 = lo + int(np.argmax(a[lo:hi] > 0.9 * top))
    j10 = j90
    while j10 > 0 and a[j10] > 0.1 * top:
        j10 -= 1
    return max(float(t[j90] - t[j10]), 3.0 * float(np.mean(np.diff(t))))


def checks(p, m, F):
    """
    The three boundary conditions the contact plane itself provides.

    All are reported as a fraction of peak |P|, so they are comparable with each
    other and with the free-end null. The windows are set by the physics:

      onset    where P first rises -- before it, everything is zero and dividing
               by it measures noise.
      arrival  R = 2L/c, when the free-end echo reaches the contact. M must be
               quiet BEFORE it, and the contact opens AT it. The echo's own
               measured rise is held clear of both, so that neither check is
               scored against the edge it is waiting for.
      tail     the last 5 % is dropped: exp(+eta t) amplifies the record-end
               truncation, the same trap the free-end null has.
    """
    amp = float(np.abs(p).max())
    i_on = int(np.argmax(np.abs(p) > 0.02 * amp))
    t_echo = _echo_time(p)
    # How long the LOADING takes at x = 0, 10-90, measured over the loading
    # phase only -- i.e. before the echo returns and P stops being the whole
    # story. A different quantity from the echo's rise above: this one is what
    # the specimen did to the pulse, that one is what 2L of lossy bar did to it.
    _hi = int(np.searchsorted(t, t_echo))
    _a = np.abs(p[:_hi]) if _hi > i_on + 2 else np.abs(p)
    _pk = float(_a.max())
    _j90 = int(np.argmax(_a > 0.9 * _pk))
    _j10 = int(np.argmax(_a > 0.1 * _pk))
    rise_p = float(t[_j90] - t[_j10])
    rise = _echo_rise(m, t_echo)
    i_pre = int(np.searchsorted(t, t_echo - rise))    # M must be quiet before
    i_sep = int(np.searchsorted(t, t_echo + rise))    # contact open after
    i_end = int(0.95 * N)
    # With a specimen at x = 0 the bars never part, so there is no "after
    # separation" to score -- but F >= 0 still holds, for the WHOLE record,
    # since a dry compression interface cannot pull either way.
    i_ten = i_sep if IMPACT else i_end
    return dict(
        amp=amp, i_on=i_on, i_pre=i_pre, i_sep=i_sep, i_end=i_end, rise=rise,
        rise_p=rise_p,
        causality=float(np.abs(m[i_on:i_pre]).max() / amp),
        tensile=float(max(0.0, -F[i_on:i_ten].min()) / amp),
        after=(float(np.sqrt(np.mean(F[i_sep:i_end] ** 2)) / amp)
               if IMPACT and i_end > i_sep else float('nan')),
        peak=float(F.max()),
        t_echo=float(t_echo * 1e3),
        t_open=float(t[min(i_sep, N - 1)] * 1e3),
    )


def free_end(x):
    """The free-end null, from the same positions: L - x are distances from it."""
    p, m = separate(t, sig, L - np.asarray(x, float), c0=c0, eta=eta,
                    attenuation=ATT)
    tot, amp = p + m, float(np.abs(p).max())
    w = slice(int(np.argmax(np.abs(p) > 0.02 * amp)),
              int(float(cfg.get('null', {}).get('window', 0.75)) * N))
    return dict(p=p, m=m, tot=tot, amp=amp, w=w,
                rms=float(np.sqrt(np.mean(tot[w] ** 2)) / amp))


SETS = [('identified', x_id)]
if x_tape is not None:
    SETS.append(('tape', x_tape))
RES = {}
for name, x in SETS:
    p, m, F = reconstruct(x)
    RES[name] = dict(x=x, p=p, m=m, F=F, **checks(p, m, F))
    RES[name]['null'] = free_end(x)['rms']

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
    print(f'{name:>11} {Dg:8.2f} {r["peak"]:9.3f} {r["null"]:10.2e} '
          f'{r["causality"]:11.3f} {r["tensile"]:9.3f}'
          + (f'{r["after"]:11.3f}' if IMPACT else ''))
print('the right-hand columns are fractions of peak |P|. Zero is the '
      'ideal;\nwhat is left is model error, and its SIGN is known -- a contact '
      'that pulls or a\nwave that arrives early is not a measurement, it is the '
      'residual.')
_r0 = RES[SETS[0][0]]
print(f'the echo reaches x = 0 at {_r0["t_echo"]:.0f} us -- one round trip after '
      f'the wave LEFT it,\nnot after the record started. Its own 10-90 rise '
      f'there measures {_r0["rise"]*1e3:.0f} us after\ncrossing 2L = {2*L:.0f} mm '
      'of lossy bar, and that is the clearance held either side.')

if len(SETS) > 1:
    a, b = RES[SETS[0][0]], RES[SETS[1][0]]
    off = float(np.mean(a['x'] - b['x']))
    dpk = abs(a['peak'] - b['peak']) / max(abs(a['peak']), abs(b['peak']))
    rel = float(np.sqrt(np.mean((a['F'] - b['F']) ** 2))
                / max(np.abs(a['F']).max(), np.abs(b['F']).max()))
    print(f'\nthe two sets differ by a COMMON {off:+.2f} mm '
          f'({abs(a["x"][1]-a["x"][0]) - abs(b["x"][1]-b["x"][0]):+.2f} mm in D). '
          f'The forces they give\ndiffer by {dpk*100:.1f} % in peak and '
          f'{rel:.2e} relative L2 -- which is the measurement\nof how benign a '
          'common offset is, in place of the usual assertion that it is.')

print('\n--- what the reconstruction says happened '
      '--------------------------------')
r = RES[SETS[0][0]]
print(f'  force at the {IFACE} peaks at {r["peak"]:.3f} {UNITS}')
if IMPACT:
    print('  the striker\'s own release returns one striker round trip in and '
          'steps it DOWN,\n    not to zero: an unmatched striker gives a '
          'geometric staircase of ratio\n    (Z1-Z2)/(Z1+Z2) per round trip, '
          'not a rectangular pulse. Only a MATCHED\n    striker unloads to zero '
          'at 2L/c. See README, "Why the force does not go\n    to zero when '
          'the striker unloads".')
    print(f'  free-end echo reaches the contact at {r["t_echo"]:.0f} us; the '
          f'echo is\n    TENSILE, the contact cannot carry it, and the bars part')
    print(f'  after {r["t_open"]:.0f} us the reconstructed force is '
          f'{r["after"]*100:.1f} % of peak -- i.e. zero')
else:
    print(f'  there is no sharp wavefront at all: the force at x = 0 takes '
          f'{r["rise_p"]*1e3:.0f} us to go\n    from 10 % to 90 %, against the '
          '~20 us step this same rig delivers with no\n    specimen in the way. '
          'That is the specimen, not the bar -- and it is why\n    this record '
          'cannot be used to IDENTIFY anything: the edges are gone.')
    print(f'  nothing returns to x = 0 until the free-end echo at '
          f'{r["t_echo"]:.0f} us, and M holds\n    to {r["causality"]:.3f} of '
          'peak before it -- the record is clean over the whole loading')
    print('  the bars never part here, so there is no "after separation" to '
          'score; F >= 0\n    is checked over the WHOLE record instead')


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'
tt = t * 1e3
T0_FIG = float(d.get('t0_file', 0.0))
tt_f = tt + T0_FIG

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
fig.patch.set_facecolor(SURFACE)

# --- what went in ---------------------------------------------------------
for k, s in enumerate(sig):
    axes[0].plot(tt_f, s * SCALE, lw=.9,
                 color=(BLUE, ORANGE, INK)[k % 3],
                 label=f'gauge {k} at {x_id[k]:.0f} mm (identified)')
axes[0].set_ylabel(f'Gauge signal ({USYM})')
axes[0].set_title(f'What was measured — {len(sig)} gauges on the {BAR} bar',
                  loc='left', fontsize=11)
_g = max(np.abs(s_).max() for s_ in sig) * SCALE
axes[0].set_ylim(-1.3 * _g, 1.35 * _g)
axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

# --- the two waves at the contact ----------------------------------------
r = RES[SETS[0][0]]
axes[1].plot(tt_f, r['p'] * SCALE, color=BLUE, lw=.9,
             label=r'$P$  (leaving the contact, into the bar)')
axes[1].plot(tt_f, r['m'] * SCALE, color=ORANGE, lw=.9,
             label=r'$M$  (returning to the contact)')
axes[1].axvline(r['t_echo'] + T0_FIG, color=MUTED, lw=1.1, ls='--')
axes[1].annotate('  free-end echo arrives, $2L/c$ after the wave left',
                 (r['t_echo'] + T0_FIG, 0), fontsize=9, color=MUTED, va='bottom')
for _b in (r['t_echo'] + T0_FIG - r['rise'] * 1e3,
           r['t_echo'] + T0_FIG + r['rise'] * 1e3):
    axes[1].axvline(_b, color=GRID, lw=1.0, ls=':')
axes[1].axhline(0, color=GRID, lw=.8)
axes[1].set_ylabel(f'Wave ({USYM})')
axes[1].set_title('The two waves separated AT the contact plane — $M$ is flat '
                  'zero until the echo can get back: causality residual '
                  f'{r["causality"]:.3f} of peak $|P|$'
                  + ('' if ATT is not None else '  (LOSSLESS)'),
                  loc='left', fontsize=10)
# headroom, so the legend sits above the traces rather than across them
_pk = max(np.abs(r['p']).max(), np.abs(r['m']).max()) * SCALE
axes[1].set_ylim(-1.25 * _pk, 1.55 * _pk)
axes[1].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')

# --- the answer -----------------------------------------------------------
for (name, _), col, ls in zip(SETS, (INK, BLUE), ('-', '--')):
    axes[2].plot(tt_f, RES[name]['F'] * SCALE, color=col, lw=1.0, ls=ls,
                 label=f'$F = P + M$, {name} positions')
axes[2].axhline(0, color=GRID, lw=1.0)
band = 0.03 * r['amp'] * SCALE
axes[2].axhspan(-band, band, color=BLUE, alpha=.15,
                label='±3 % of peak $|P|$')
if IMPACT:
    axes[2].axvline(r['t_open'] + T0_FIG, color=ORANGE, lw=1.1, ls='--')
    axes[2].annotate('  bars part', (r['t_open'] + T0_FIG, r['peak'] * SCALE * .6),
                     fontsize=9, color=MUTED)
axes[2].set_xlabel('Time (us)')
axes[2].set_ylabel(f'Interface force ({UNITS})')
axes[2].set_title(f'THE ANSWER — force at the {IFACE}. It cannot go '
                  'negative (a dry contact does not pull): residual '
                  f'{r["tensile"]:.3f} of peak'
                  + ('' if ATT is not None else '  (LOSSLESS)'),
                  loc='left', fontsize=10)
axes[2].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')
axes[2].set_xlim(tt_f[0], tt_f[int(0.95 * N)])

for ax in axes:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.suptitle(f'Force at the {IFACE}, reconstructed from two gauges '
             f'{abs(x_id[1]-x_id[0]):.0f} mm apart on the {BAR} bar'
             + ('' if CASE == str(ID['case'])
                else f' calibrated on [{str(ID["case"])}]'),
             x=.006, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .975))
_stem = 'interface_force' + ('' if CASE == str(ID['case']) else f'_{CASE}')
FIG = (f'{_stem}.png' if ATT is not None else f'{_stem}_lossless.png')
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'\nwrote {FIG}')

DAT = (f'{_stem}.dat' if ATT is not None else f'{_stem}_lossless.dat')
T0 = float(d.get('t0_file', 0.0))
np.savetxt(DAT,
           np.column_stack([tt + T0, RES[SETS[0][0]]['F'], RES[SETS[0][0]]['p'],
                            RES[SETS[0][0]]['m']]),
           header=f'time[us]  F_interface[{UNITS}]  P[{UNITS}]  M[{UNITS}]\n'
                  f'time is the SOURCE FILE\'s own base (analysis t=0 sits at '
                  f'{T0:.1f} us there)\n'
                  f'{BAR} bar, c0={c0:.3f} mm/ms, x={[round(float(v), 2) for v in x_id]} mm, '
                  f'eta={eta:g}, x=0 is the {IFACE}')
print(f'wrote {DAT}')

plotting.show_unless(HEADLESS)
