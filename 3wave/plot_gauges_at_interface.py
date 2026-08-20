"""
Each gauge shifted to x = 0 ON ITS OWN, against the two-gauge separation.

    python3 identify_bar_compression.py --experiment experiment_pc_bar
    python3 plot_gauges_at_interface.py [--case CASE] [--headless]

One gauge gives one equation per frequency and there are two unknowns, so a
single gauge cannot separate anything. What it CAN do is be shifted to the
interface on the assumption that only one wave is passing it -- which on a
direct-impact bar is true for a while, because the loading wave is generated at
the interface and nothing comes back until the far free end returns it.

This plot is that assumption, drawn. `backpropagate` shifts each gauge to x = 0
by itself; `separate` uses both together. While the assumption holds the three
curves lie on top of each other, and the moment it stops holding the
single-gauge curves peel away -- each at its OWN time, because each gauge's
window ends when ITS echo arrives:

    gauge at distance d on a bar of length L:  valid until t = 2 (L - d) / c0

measured from the moment the wave left x = 0. That is the window in the
RECONSTRUCTION, which is the gauge record advanced by d/c0 -- the echo reaches
the gauge itself later, at (2L - d)/c0, and using that number instead overstates
the window by d/c0. See `backpropagate`.

Note which way round that is. The window is SHORTER for the gauge further from
the interface, because its echo has less far to travel back. The far gauge is
the first to go wrong, not the last.

What the picture is for:

  * it says how much of the record needed two gauges at all. Where the curves
    agree, one gauge would have done, and a classical single-gauge reduction is
    not wrong -- just unnecessary;
  * it shows what the classical reduction would have given past that point, and
    the answer is: something entirely plausible, and wrong by the whole of the
    neglected wave. That is the failure mode multi-gauge separation exists for,
    and it is silent;
  * it is a check on the identified numbers that uses neither boundary
    condition. Two gauges 371 mm apart, shifted independently, have no reason to
    agree unless c0 and alpha(f) are right.
"""
import argparse

import numpy as np

import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--case', default=None,
                 help='config case to plot; default is whichever one '
                      'bar_identified.npz was written from.')
_ap.add_argument('--bar', default=None, help='which bar. Default "out".')
_ap.add_argument('--no-attenuation', action='store_true',
                 help='shift with a lossless bar, i.e. a pure time delay.')
HEADLESS, ARGS = plotting.init(parser=_ap)

import config
from wave_separation import separate, backpropagate, wavefront_time

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
T0 = float(d.get('t0_file', 0.0))

c0 = float(ID[f'c_{BAR}'])
L = float(ID[f'L_ref_{BAR}'])
x = np.asarray(ID[f'x_{BAR}'], float)
ATT = None
if f'alpha_{BAR}' in ID.files and not ARGS.no_attenuation:
    ATT = (np.asarray(ID[f'alpha_f_{BAR}'], float),
           np.asarray(ID[f'alpha_{BAR}'], float))
IMPACT = str(cfg.get('interface', 'impact')) == 'impact'
IFACE = 'impact interface' if IMPACT else f'{BAR}put-bar / specimen interface'

# --------------------------------------------------------------------------
# the three reconstructions
# --------------------------------------------------------------------------
# Two gauges, the real thing: both waves, valid throughout.
P, M = separate(t, sig, x, c0=c0, eta=eta, attenuation=ATT)
F = P + M

# One gauge each, assuming the returning wave is absent. eta = 0 here: there is
# no determinant to regularise, so this is an exact shift (plus the attenuation
# correction) rather than a windowed solve, and it keeps the comparison about
# the ONE-WAVE assumption instead of about two different regularisations.
SINGLE = [backpropagate(t, s, xi, c0=c0, eta=0.0, attenuation=ATT,
                        direction='plus')[0]
          for s, xi in zip(sig, x)]

# When each gauge stops being alone: its own echo, (2L - d)/c0 after the wave
# left x = 0. Referred to the record by P's own onset, as elsewhere.
t_left = wavefront_time(t, P)
VALID = [t_left + 2.0 * (L - xi) / c0 for xi in x]

print(__doc__.split('---')[0].strip())
print(f'\nrecord    : {d.get("source", "dump.npz")}')
print(f'bar       : {BAR}, {L:.1f} mm, c0 = {c0:.2f} mm/ms'
      + ('' if ATT is None else f', alpha(f) to {ATT[0][-1]:.0f} kHz'))
print(f'x = 0 is  : the {IFACE}')
print(f'the wave leaves x = 0 at {(t_left + T0/1e3)*1e3:.0f} us '
      '(source-file time base)\n')

print(f'{"gauge":>7} {"x [mm]":>8} {"single-wave window ends":>25}')
print(f'{"":>7} {"":>8} {"t_left + 2(L-d)/c0":>25}')
for k, xi in enumerate(x):
    print(f'{BAR}-{k:<5} {xi:8.2f} {VALID[k]*1e3 + T0:22.0f} us')
print(f'\nthe FAR gauge expires FIRST: its echo has less bar to cross. '
      f'{abs(VALID[1]-VALID[0])*1e3:.0f} us\nseparates the two windows, which is '
      f'2 D / c0 for D = {abs(x[1]-x[0]):.1f} mm.')
print('\nno in-window error is quoted here on purpose. Both single-gauge curves '
      'ring at\nthe START of the record -- shifting a gauge to x = 0 moves its '
      'wavefront toward\nthe record boundary, and the far gauge moves it '
      f'{max(x)/c0*1e3:.0f} us -- and the echo edge leaks\nahead of its own '
      'arrival at the END. Any single number over the window is one of\nthose '
      'two artefacts, not the physics. Read the lower panel instead: it is flat '
      'in\nbetween, which is the whole claim.')

# Past the first window, quote what the single-gauge route would have claimed.
i_end = min(int(0.95 * N), len(t) - 1)
i_bad = int(np.searchsorted(t, min(VALID)))
if i_bad < i_end:
    w = slice(i_bad, i_end)
    print('\nAFTER the first window closes, the single-gauge curves are wrong '
          'by the whole\nof the wave they neglect:')
    for k in range(len(x)):
        print(f'  {BAR}-{k}: {np.sqrt(np.mean((SINGLE[k][w]-F[w])**2))/np.abs(F[w]).max():.2f} '
              'relative rms against the two-gauge answer')
    print('and neither of them looks wrong on the page. That is the point.')

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'
tt = t * 1e3 + T0

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=(3, 1.6)))
fig.patch.set_facecolor(SURFACE)

COL = (BLUE, ORANGE, '#5aa469')
for k in range(len(x)):
    axes[0].plot(tt, SINGLE[k] * SCALE, color=COL[k % 3], lw=1.0,
                 label=f'{BAR}-{k} at {x[k]:.0f} mm, shifted alone')
axes[0].plot(tt, F * SCALE, color=INK, lw=1.4, ls='--',
             label='both gauges, separated ($F = P + M$)')
for k in range(len(x)):
    axes[0].axvline(VALID[k] * 1e3 + T0, color=COL[k % 3], lw=1.1, ls=':')
    axes[0].annotate(f'  {BAR}-{k} window ends', (VALID[k] * 1e3 + T0,
                     np.abs(F).max() * SCALE * (0.97 - 0.09 * k)),
                     fontsize=8.5, color=COL[k % 3])
axes[0].axvspan(tt[0], min(VALID) * 1e3 + T0, color=GRID, alpha=.35, lw=0,
                label='one wave only — a single gauge is enough here')
axes[0].axhline(0, color=GRID, lw=.8)
axes[0].set_ylabel(f'Force at $x = 0$ ({UNITS})')
axes[0].set_title(f'Each gauge shifted to the {IFACE} on its own, against the '
                  'two-gauge separation', loc='left', fontsize=11)
axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')

for k in range(len(x)):
    axes[1].plot(tt, (SINGLE[k] - F) * SCALE, color=COL[k % 3], lw=1.0,
                 label=f'{BAR}-{k} − separated')
    axes[1].axvline(VALID[k] * 1e3 + T0, color=COL[k % 3], lw=1.1, ls=':')
axes[1].axvspan(tt[0], min(VALID) * 1e3 + T0, color=GRID, alpha=.35, lw=0)
axes[1].axhline(0, color=GRID, lw=.8)
axes[1].set_xlabel('Time (us)  — source-file time base')
axes[1].set_ylabel(f'Error ({UNITS})')
axes[1].set_title('The neglected wave, which is exactly the error. Flat while '
                  'the assumption holds.', loc='left', fontsize=10)
axes[1].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

for ax in axes:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
axes[1].set_xlim(tt[0], tt[i_end])

fig.suptitle('One gauge is enough — until it is not',
             x=.006, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .968))
FIG = 'gauges_at_interface' + ('' if CASE == str(ID['case']) else f'_{CASE}') + '.png'
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'\nwrote {FIG}')

plotting.show_unless(HEADLESS)
