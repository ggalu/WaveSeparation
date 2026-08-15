"""
Full reduction: three strain gauges per bar -> specimen stress/strain response,
validated against the simulator's own specimen measurement.

Run drive_compression.py (compression) or drive_tension.py (SHTB) first to produce
dump.npz. Gauge locations and eta come from config.toml. Then:

    python3 reduce_specimen.py              # shows the figure in a window
    python3 reduce_specimen.py --headless   # writes the .png only, no window

The figure is always written to specimen_reconstruction.png either way.
--headless is also implied by MPL_HEADLESS=1 or by there being no display,
so the script is safe to run over ssh or from a batch job.
"""
import numpy as np

import plotting
HEADLESS = plotting.init(__doc__)   # picks the backend; must precede pyplot

from dump import load_dump
from wave_separation import separate, bar_interface, specimen_response

# --- load ------------------------------------------------------------------
# One file, and the gauge signals arrive with their exact positions already
# resolved -- see dump.py for the full field list.
d = load_dump()
E, A, c0, dt, t = d['E'], d['A'], d['c0'], d['dt'], d['t']
L_SPEC, A_SPEC = d['L_specimen'], d['A_specimen']
LOADING, ETA = d['loading'], d['eta']

# --- separate each bar, then reduce ---------------------------------------
# The input bar's interior lies toward global -x, the output bar's toward +x.
# eta comes from config.toml: stress is insensitive to it, but strain (which
# requires integration) degrades badly below ~0.5 /ms. See the regularisation
# notes in wave_separation.py.
p_in, m_in = separate(t, d['eps_in'], d['pos_in'], c0=c0, eta=ETA)
p_out, m_out = separate(t, d['eps_out'], d['pos_out'], c0=c0, eta=ETA)

# v0: separation cannot see rigid-body motion, so each bar's pre-impact velocity
# must be added back. Both come from the dump -- the direct-impact input bar
# arrives at 10 mm/ms, an SHTB's bars are both at rest.
F_in, v_in = bar_interface(p_in, m_in, E, A, c0, outward=-1, v0=d['v0_in'])
F_out, v_out = bar_interface(p_out, m_out, E, A, c0, outward=+1, v0=d['v0_out'])

res = specimen_response(t, F_in, v_in, F_out, v_out, L_SPEC, A_SPEC,
                        loading=LOADING)

# --- ground truth ----------------------------------------------------------
# The dump stores the specimen truth in the simulator's own sign convention:
# compression negative, tension positive. Flip only the compression case so both
# come out positive in their loading sense, matching specimen_response.
_sgn = -1.0 if LOADING == 'compression' else 1.0
sig_true, eps_true = _sgn * d['spec_stress'], _sgn * d['spec_strain']

# Analysis window, derived from the truth rather than hardcoded: from first
# loading to when the specimen stops deforming (peak strain), plus a little pad.
# Compression and tension have quite different event timings, and a bonded
# tension specimen stays loaded long after deformation ends.
_live = np.abs(sig_true) > 0.02 * np.abs(sig_true).max()
_i0 = int(np.argmax(_live))
_i1 = int(np.argmax(np.abs(eps_true))) + int(0.05 / dt)
_t0, _t1 = t[_i0], t[min(_i1, len(t) - 1)]
win = (t >= _t0) & (t <= _t1)
def relerr(a, b):
    return np.linalg.norm(a[win] - b[win]) / np.linalg.norm(b[win])

print(f'loading = {LOADING}, eta = {ETA} /ms, '
      f'{len(d["pos_in"])} gauges per bar at '
      f'{[f"{p:.1f}" for p in d["pos_in"]]} mm (input), '
      f'{[f"{p:.1f}" for p in d["pos_out"]]} mm (output)')
print(f'analysis window {_t0:.3f} - {_t1:.3f} ms\n')
print(f'peak specimen strain (true)      : {eps_true[win].max():.4f}')
print(f'peak specimen stress (true)      : {sig_true[win].max():.5f} GPa')
print(f'peak strain rate (reconstructed) : {res["strain_rate"][win].max():.1f} /ms'
      f'  = {res["strain_rate"][win].max()*1e3:.0f} /s')
print()
print(f'stress   rel L2 err vs truth         : {relerr(res["stress"], sig_true):.4e}')
print(f'strain   rel L2 err vs truth         : {relerr(res["strain"], eps_true):.4e}')
print(f'force equilibrium |F1-F2|/max|F1|    : mean {res["equilibrium"][win].mean():.4e},'
      f' max {res["equilibrium"][win].max():.4e}')
print()
print('strain error broken down in time:')
for lo, hi in zip(np.linspace(_t0, _t1, 6)[:-1], np.linspace(_t0, _t1, 6)[1:]):
    m = (t >= lo) & (t < hi)
    print(f'  {lo:.2f}-{hi:.2f} ms  strain abs err {np.abs(res["strain"][m]-eps_true[m]).max():.4e}'
          f'   (true strain reaches {eps_true[m].max():.4f})')

np.savetxt('specimen_reconstructed.dat',
           np.column_stack((t, res['stress'], res['strain'], res['strain_rate'])),
           header=f'time[ms]  stress[GPa]  strain[-]  strain_rate[1/ms]  ({LOADING} positive)')
print('\nwrote specimen_reconstructed.dat')

# --- plot ------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
c = res['contact']
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
fig.patch.set_facecolor('#fcfcfb')

axes[0].plot(eps_true[c], sig_true[c] * 1e3, color=INK, lw=2.6, alpha=.30,
             label='simulator (truth)')
axes[0].plot(res['strain'][c], res['stress'][c] * 1e3, color=BLUE, lw=1.6,
             label='reconstructed')
axes[0].set_xlabel('Engineering strain'); axes[0].set_ylabel('Stress (MPa)')
axes[0].set_title('Specimen response', loc='left', fontsize=11)

axes[1].plot(t, eps_true, color=INK, lw=2.6, alpha=.30, label='simulator (truth)')
axes[1].plot(t, res['strain'], color=BLUE, lw=1.6, label='reconstructed')
axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Engineering strain')
axes[1].set_title('Strain history', loc='left', fontsize=11)

axes[2].plot(t, res['stress_in'] * 1e3, color=BLUE, lw=1.4, label='input face $F_1$')
axes[2].plot(t, res['stress_out'] * 1e3, color=ORANGE, lw=1.4, label='output face $F_2$')
axes[2].plot(t, sig_true * 1e3, color=INK, lw=2.6, alpha=.30, label='simulator (truth)')
axes[2].set_xlabel('Time (ms)'); axes[2].set_ylabel('Stress (MPa)')
axes[2].set_title('Force equilibrium check', loc='left', fontsize=11)

for ax in axes:
    ax.set_facecolor('#fcfcfb'); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
axes[1].set_xlim(0, _t1); axes[2].set_xlim(0, _t1)

_n = len(d['pos_in'])
fig.suptitle(f'Specimen response recovered from {_n} strain gauge'
             f'{"s" if _n > 1 else ""} per bar ({LOADING})',
             x=.007, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig('specimen_reconstruction.png', dpi=140, facecolor=fig.get_facecolor())
print('wrote specimen_reconstruction.png')

plotting.show_unless(HEADLESS)
