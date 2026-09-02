"""
Force equilibrium across the coupler, from a real SHTB calibration shot.

    python3 identify_bar_tension.py --experiment experiment_tension_bar_2   # once
    python3 bar_equilibrium.py [--case experiment_tension_bar_2] [--headless]

Each bar is separated INDEPENDENTLY from its own two gauges, using the c0 and
positions identify_bar_tension.py --experiment already wrote to
bar_identified.npz, and turned into a force at its OWN face touching the
coupler:

    F_bar(t) = P_bar(t) + M_bar(t)

No E, A or rho enters -- the gauge records are already force (see
experiment.py's "Force in, force out" note). Force is continuous across a
genuine joint, so F_in and F_out reconstructed this way should agree.
|F_in - F_out| / max|F_in| is exactly wave_separation.specimen_response's own
`equilibrium` field -- reused here directly rather than through the rest of
that function's velocity/strain machinery, which assumes a deforming
specimen and does not apply to a rigid coupler.

drive_tension.py cannot do this: it is the SIMULATOR driver (writes dump.npz
from [tension]'s own model) and takes no data file. This reads a MEASURED
shot instead, through the same experiment.py path identify_bar_tension.py
--experiment uses.
"""
import argparse

import numpy as np

import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--case', default=None,
                 help='config case to reconstruct; default is whichever one '
                      'bar_identified.npz was written from.')
HEADLESS, ARGS = plotting.init(parser=_ap)

import config
from wave_separation import separate

IDENT_FILE = 'bar_identified.npz'
try:
    ID = np.load(IDENT_FILE, allow_pickle=True)
except FileNotFoundError:
    raise SystemExit(
        f'{IDENT_FILE} not found. Run the identification first:\n'
        '    python3 identify_bar_tension.py --experiment experiment_tension_bar_2')

CASE = ARGS.case or str(ID['case'])
BARS = [str(b) for b in ID['bars']]
if 'in' not in BARS or 'out' not in BARS:
    raise SystemExit(
        f'{IDENT_FILE} covers {BARS}, not both -- equilibrium needs a force at '
        'EACH bar\'s own face. Re-run the identification on a shot with two '
        'gauges on both bars.')

if CASE in config.EXPERIMENT_CASES:
    from experiment import load_experiment
    d = load_experiment(CASE)
else:
    from dump import load_dump
    d = load_dump()
cfg = config.load(CASE)
t, dt, N = d['t'], d['dt'], d['N']
eta = d['eta']
UNITS = d.get('units', 'strain')
SCALE = 1.0 if UNITS != 'strain' else 1e6
USYM = UNITS if UNITS != 'strain' else 'ustrain'

# --------------------------------------------------------------------------
# separate each bar on its own -- two independent 2-gauge solves, exactly as
# identify_bar_tension.py's Q/c0 apply to either half of the SAME assembly,
# but here each bar's own face (x = 0 there) is what is reconstructed, not
# the far free end.
# --------------------------------------------------------------------------
F = {}
for b in ('in', 'out'):
    sig = list(d[f'eps_{b}'])
    c0 = float(ID[f'c_{b}'])
    x = np.asarray(ID[f'x_{b}'], float)
    p, m = separate(t, sig, x, c0=c0, eta=eta)
    F[b] = p + m
    print(f'{b:>3} bar: c0={c0:.1f} mm/ms, x={[round(float(v), 1) for v in x]} mm, '
          f'{len(sig)} gauges')

peak = float(np.abs(F['in']).max())
equilibrium = np.abs(F['in'] - F['out']) / (peak if peak > 0 else 1.0)

# Window: clear of the record's quiescent start and of the tail the eta-window
# amplifies -- same convention as the free-end null test in
# identify_bar_tension.py / reconstruct_interface.py.
NULL_WINDOW = (float(cfg.get('null', {}).get('window', 0.75))
              if CASE in config.EXPERIMENT_CASES
              else float(cfg.get('null_window', 0.75)))
amp = max(float(np.abs(F['in']).max()), float(np.abs(F['out']).max()))
_i0 = int(np.argmax((np.abs(F['in']) + np.abs(F['out'])) > 0.02 * amp))
_i1 = min(int(NULL_WINDOW * N), N - 1)
win = slice(_i0, _i1)

print(f'\nanalysis window {t[_i0]:.3f}-{t[_i1]:.3f} ms')
print(f'force equilibrium |F_in-F_out|/max|F_in| : '
      f'mean {equilibrium[win].mean():.4e}, max {equilibrium[win].max():.4e}')

# --------------------------------------------------------------------------
# .dat
# --------------------------------------------------------------------------
T0 = float(d.get('t0_file', 0.0))
DAT = f'bar_equilibrium_{CASE}.dat'
np.savetxt(DAT, np.column_stack([t * 1e3 + T0, F['in'], F['out'], equilibrium]),
          header=f'time[us]  F_in[{UNITS}]  F_out[{UNITS}]  equilibrium\n'
                 f'time is the SOURCE FILE\'s own base (analysis t=0 sits at '
                 f'{T0:.1f} us there)')
print(f'wrote {DAT}')

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
tt = t * 1e3 + T0
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.patch.set_facecolor('#fcfcfb')

axes[0].plot(tt, F['in'] * SCALE, color=BLUE, lw=.9, label='F_in (input-bar face)')
axes[0].plot(tt, F['out'] * SCALE, color=ORANGE, lw=.9, label='F_out (output-bar face)')
axes[0].axhline(0, color=GRID, lw=.8)
axes[0].set_ylabel(f'Force ({USYM})')
axes[0].set_title(f'Force at each bar\'s own face touching the coupler — {CASE}',
                  loc='left', fontsize=11)
axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')

axes[1].plot(tt, equilibrium, color=INK, lw=.9)
for _b in (tt[_i0], tt[_i1]):
    axes[1].axvline(_b, color=ORANGE, lw=1.1, ls='--')
axes[1].set_xlabel('Time (us)')
axes[1].set_ylabel('|F_in - F_out| / max|F_in|')
axes[1].set_title(f'Equilibrium residual — mean {equilibrium[win].mean():.3e}, '
                  f'max {equilibrium[win].max():.3e} in the analysis window',
                  loc='left', fontsize=11)

for ax in axes:
    ax.set_facecolor('#fcfcfb'); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.tight_layout()
FIG = f'bar_equilibrium_{CASE}.png'
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'wrote {FIG}')

plotting.show_unless(HEADLESS)
