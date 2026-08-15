"""
Plot the force seen at every strain-gauge location alongside the average force
carried by the specimen.

This is raw signal inspection -- it does NOT run the wave separation. Its job is
to show what the gauges actually record, and in particular WHEN the two
counter-propagating waves start to overlap at each gauge.

Run drive_compression.py or drive_tension.py first to produce dump.npz, then:
    python3 plot_forces.py              # shows the figure in a window
    python3 plot_forces.py --headless   # writes the .png only, no window

The figure is always written to gauge_forces.png either way. --headless is also
implied by MPL_HEADLESS=1 or by there being no display, so the script is safe to
run over ssh or from a batch job.

Works with either simulator: the loading sense comes from the dump, so
compression data is plotted compression-positive and tension data
tension-positive.
"""
import numpy as np

import plotting
HEADLESS = plotting.init(__doc__)     # picks the backend; must precede pyplot
import matplotlib.pyplot as plt

from dump import load_dump

# --- load ------------------------------------------------------------------
# see dump.py for the full definition of each entry
d = load_dump()
E, A, c0, t = d['E'], d['A'], d['c0'], d['t']
A_SPEC, LOADING = d['A_specimen'], d['loading']

# Sign conventions and labels follow the loading sense recorded in the dump.
TENSION = LOADING == 'tension'
SIGN = +1.0 if TENSION else -1.0              # make the loading sense positive
SENSE = 'tension +' if TENSION else 'compression +'
print(f'dump is {"TENSION (SHTB)" if TENSION else "COMPRESSION (direct impact)"}')

# A strain gauge measures strain, so the force it reports is E*A*eps. This is
# deliberately NOT the recorded element force: with damping != 0 that also
# carries the artificial-viscosity term, which no real gauge would see.
gauge_forces = {'in': SIGN * E * A * d['eps_in'],
                'out': SIGN * E * A * d['eps_out']}
gauge_pos = {'in': d['pos_in'], 'out': d['pos_out']}

# Force carried by the specimen, as the simulator measured it.
specimen_force = SIGN * d['spec_stress'] * A_SPEC

# --- colors (dataviz reference palette, categorical slots 1-3) -------------
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d3'

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
fig.patch.set_facecolor('#fcfcfb')

for ax, bar, label in zip(axes, ('in', 'out'), ('Input bar', 'Output bar')):
    ax.set_facecolor('#fcfcfb')
    for sig, exact, color in zip(gauge_forces[bar], gauge_pos[bar], SERIES):
        ax.plot(t, sig, color=color, lw=1.6, label=f'gauge @ {exact:.0f} mm')
    ax.plot(t, specimen_force, color=INK, lw=1.4, ls=(0, (5, 2)), alpha=.75,
            label='specimen (mean)')

    # round trip to the bar's far end: the reflection arrives back here.
    # Geometrically the same in both simulators; only the boundary condition
    # there differs (a free end, or the struck anvil on the SHTB input bar).
    L_free = d['L_free_in'] if bar == 'in' else d['L_free_out']
    t_end = 2 * L_free / c0
    ax.axvline(t_end, color=INK_MUTED, lw=1, ls=':', alpha=.7)
    ax.annotate('far-end reflection\nreaches interface', xy=(t_end, ax.get_ylim()[1]),
                xytext=(4, -4), textcoords='offset points', va='top',
                fontsize=8, color=INK_MUTED)

    ax.set_title(label, loc='left', fontsize=11, color=INK, pad=8)
    ax.set_ylabel(f'Force (kN, {SENSE})', fontsize=10, color=INK_MUTED)
    ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc='upper right')

axes[-1].set_xlabel('Time (ms)', fontsize=10, color=INK_MUTED)
axes[0].set_xlim(0, t[-1])
fig.suptitle('Force at each strain gauge vs. average specimen force',
             x=.125, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .97))
fig.savefig('gauge_forces.png', dpi=140, facecolor=fig.get_facecolor())
print('wrote gauge_forces.png')

# --- when does overlap actually begin at each gauge? ----------------------
print('\noverlap onset = arrival of the far-end reflection at the gauge')
for bar, Lfree in (('in', d['L_free_in']), ('out', d['L_free_out'])):
    print(f'  {bar} bar (far end {Lfree:.0f} mm from interface):')
    for exact in gauge_pos[bar]:
        print(f'    gauge @ {exact:6.1f} mm : outgoing {exact/c0:.3f} ms, '
              f'reflected {(2*Lfree - exact)/c0:.3f} ms, '
              f'loading ends {2*Lfree/c0:.3f} ms')

# last, so the timings above are readable before the window blocks
plotting.show_unless(HEADLESS)
