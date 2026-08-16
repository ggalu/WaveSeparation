"""
Lagrange (x-t) diagram of the SEPARATED waves, across the whole assembly.

    python3 drive_tension.py
    python3 lagrange_diagram.py [--headless]

Every other figure in this folder shows the separated waves as time series at
one plane. This one shows them as FIELDS: each bar is separated from its own
gauges, then the result is evaluated at several hundred stations along that bar,
giving eps_plus(x, t) and eps_minus(x, t) over the full record. The two families
of characteristics -- one running away from the specimen, one toward it -- are
then simply the two panels.

Nothing here needs the simulator's full field, so it runs on the ordinary
2.9 MB dump and takes a couple of seconds. What it does need is
`wave_separation.separate_field`, which propagates the P(w), M(w) SPECTRA to
each station. Re-transforming `separate`'s time-domain output instead is lossy
by 1.4e-01 -- see that function's docstring.

--------------------------------------------------------------------------
What the picture does and does not prove
--------------------------------------------------------------------------
With no dispersion, propagating a separated wave is an exact time shift, so
|eps_plus| is CONSTANT along x and its panel is a shear of one 1-D signal. Taken
alone the first two panels are a good way to see the method and a poor way to
check it.

Their SUM is not a shear, and it is where the physics is: it is the strain a
gauge at that station would actually have recorded. Three consequences are
visible, and this script prints all three as numbers rather than leaving them to
the eye:

  * at the four gauge stations the sum must reproduce the RECORDED strain, and
    does, to ~1e-14. Those are the only x where the field is data-constrained;
    everywhere else it is the model talking;
  * at a free surface the two waves must cancel. The output bar's far end is
    free, and the residual there is ~3e-04 of peak -- but only within ~1 mm of
    the surface, and only if the record's tail is windowed off (unwindowed it
    reads 6.5e-02, 200x worse, which is the same trap identify_bar_tension.py
    documents);
  * E*A*(eps_plus + eps_minus) at x = 0 is the interface force sep_test.py
    already validates against the simulator's own.

--------------------------------------------------------------------------
Where the reconstruction is not valid
--------------------------------------------------------------------------
`separate` assumes one uniform bar of speed c0 on the straight line from each
gauge to the plane being reconstructed. It assumes nothing at all about
boundary conditions. So the mask follows from the materials, not from the ends:

  * the ANVIL, beyond the input bar, is steel -- masked. The dump carries
    L_bar_in for exactly this: it is 3000 mm where L_free_in is 3020;
  * the SPECIMEN is not a bar and neither separation covers it -- masked;
  * the STRIKER overlaps the input bar over [20, 820] mm, but it is a separate
    chain touching the bar only through the anvil contact. The bar under it is
    plain aluminium and the reconstruction there is as good as anywhere else.
    It is tinted, NOT masked -- what is missing is the striker's own strain,
    which the simulator never records;
  * the last 20 mm of the input bar is likewise NOT masked. Its far boundary is
    an anvil rather than a free end, but separation was never told about
    boundaries, so nothing about the reconstruction changes there.
"""
import argparse
import textwrap

import numpy as np

import plotting

# argparse owns the whole command line and the backend must be picked before
# pyplot is imported, so both happen up here -- see plotting.py.
_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--x-step', type=float, default=8.0, metavar='MM',
                 help='spacing of the reconstruction stations [mm]. Snapped to '
                      'a whole number of elements. Default 8.')
_ap.add_argument('--n-rows', type=int, default=1400, metavar='N',
                 help='time rows in the image. The record is block-meaned down '
                      'to this. Default 1400.')
_ap.add_argument('--null-window', type=float, default=0.75, metavar='FRAC',
                 help='fraction of the record the printed metrics use. The '
                      'tail MUST be cut: exp(+eta t) amplifies the record-end '
                      'truncation. Default 0.75.')
_ap.add_argument('--chunk', type=int, default=64, metavar='N',
                 help='stations synthesised per FFT batch; bounds peak memory. '
                      'Default 64.')
HEADLESS, ARGS = plotting.init(parser=_ap)

from dump import load_dump
from wave_separation import separate_field

d = load_dump()
E, A, c0, dt, t = d['E'], d['A'], d['c0'], d['dt'], d['t']
dx, N, ETA, LOADING = d['dx'], d['N'], d['eta'], d['loading']
X_IN, X_OUT = d['X_IN'], d['X_OUT']
L_BAR_IN, L_BAR_OUT = d['L_bar_in'], d['L_bar_out']
L_FREE_IN, L_FREE_OUT = d['L_free_in'], d['L_free_out']

n_in = d['eps_in'].shape[0]
n_out = d['eps_out'].shape[0]
if n_in < 2 or n_out < 2:
    raise SystemExit(
        f'separation needs two gauges per bar; this dump has {n_in} on the '
        f'input bar and {n_out} on the output bar. Add a second entry to '
        '`gauges` in config.toml and re-run the driver. A single gauge leaves '
        'P and M under-determined -- see "A consequence worth knowing" in '
        'README.md.')

print(__doc__.split('---')[0].strip())
print(f'\nrecord     : {N} samples at {dt*1e3:.4f} us  ({t[-1]:.3f} ms), '
      f'{LOADING} positive')
print(f'assembly   : input bar {L_BAR_IN:.0f} mm | specimen '
      f'{d["L_specimen"]:.0f} mm | output bar {L_BAR_OUT:.0f} mm, '
      f'c0 = {c0:.1f} mm/ms')
if L_BAR_IN < L_FREE_IN or L_BAR_OUT < L_FREE_OUT:
    print(f'             non-bar material beyond the input bar '
          f'({L_FREE_IN - L_BAR_IN:.0f} mm) and the output bar '
          f'({L_FREE_OUT - L_BAR_OUT:.0f} mm) -- masked below')

# --------------------------------------------------------------------------
# the station grid: element centres, so nothing is interpolated
# --------------------------------------------------------------------------
X_TOTAL = X_OUT + L_FREE_OUT
n_elem = int(round(X_TOTAL / dx))
stride = max(1, int(round(ARGS.x_step / dx)))
cols = np.arange(0, n_elem, stride)
X = (cols + 0.5) * dx                      # global position of each station

in_bar = (X < X_IN) & (X >= X_IN - L_BAR_IN)
out_bar = (X > X_OUT) & (X <= X_OUT + L_BAR_OUT)
invalid = ~(in_bar | out_bar)
assert np.all(np.abs(X / dx - 0.5 - np.round(X / dx - 0.5)) < 1e-9), \
    'stations are not on element centres'

# Local coordinate of each station: distance from its own interface, positive
# INTO the bar. This is the sign that a Lagrange diagram gets wrong silently,
# which is why the shift check below exists.
x_loc = np.where(in_bar, X_IN - X, X - X_OUT)

q = max(1, N // max(1, ARGS.n_rows))
print(f'stations   : {in_bar.sum()} input + {out_bar.sum()} output '
      f'(every {stride*dx:.0f} mm), {invalid.sum()} masked')
print(f'time rows  : {N // q} (block mean of {q} samples; '
      f'the wavefront is ~380 samples wide)')

# --------------------------------------------------------------------------
# the two fields
# --------------------------------------------------------------------------
n_rows = N // q
eps_p = np.full((len(X), n_rows), np.nan)
eps_m = np.full((len(X), n_rows), np.nan)
t_img = None

for mask, sig, pos in ((in_bar, d['eps_in'], d['pos_in']),
                       (out_bar, d['eps_out'], d['pos_out'])):
    if not mask.any():
        continue
    p, m, t_img = separate_field(
        t, [sig[k] for k in range(sig.shape[0])], list(pos), c0=c0, eta=ETA,
        x=x_loc[mask], decimate=q, chunk=ARGS.chunk)
    eps_p[mask], eps_m[mask] = p, m

eps_sum = eps_p + eps_m

# --------------------------------------------------------------------------
# checks -- printed, not eyeballed. None of these needs the full field.
# --------------------------------------------------------------------------
print('\n--- checks '
      '----------------------------------------------------------------')

# (1) at a gauge the reconstruction must reproduce what was recorded. These are
#     the only stations where the field is constrained by data.
print('reconstruction vs the RECORDED strain, at the gauges:')
worst = 0.0
for bar, sig, pos, plane, sgn in (('in', d['eps_in'], d['pos_in'], X_IN, -1),
                                  ('out', d['eps_out'], d['pos_out'], X_OUT, +1)):
    sigs = [sig[k] for k in range(sig.shape[0])]
    p, m, _ = separate_field(t, sigs, list(pos), c0=c0, eta=ETA,
                             x=list(pos), decimate=1, chunk=ARGS.chunk)
    for k, xk in enumerate(pos):
        rel = np.linalg.norm(p[k] + m[k] - sig[k]) / np.linalg.norm(sig[k])
        worst = max(worst, rel)
        print(f'   {bar}-{k}  x_local {xk:7.1f} mm  (global {plane + sgn*xk:7.1f})'
              f'   rel L2 {rel:.2e}')
if worst > 1e-10:
    raise SystemExit(f'gauge round trip is {worst:.1e}; the separation or the '
                     'station mapping is wrong')

# (2) propagation is an exact time shift, so the field at c0*dt*k must equal the
#     field at 0 shifted by k samples. This is what catches a sign error in the
#     local->global mapping, which no other check here can see.
sigs = [d['eps_in'][k] for k in range(n_in)]
ks = (1, 13, 500)
p_s, _, _ = separate_field(t, sigs, list(d['pos_in']), c0=c0, eta=ETA,
                           x=[0.0] + [c0 * dt * k for k in ks], decimate=1)
shift = max(np.abs(p_s[j, k:] - p_s[0, :-k]).max() / np.abs(p_s[0]).max()
            for j, k in enumerate(ks, start=1))
print(f'\npropagation is an exact time shift          : {shift:.1e} '
      f'(over {ks[-1]} samples = {c0*dt*ks[-1]:.0f} mm)')
if shift > 1e-11:
    raise SystemExit('propagation is not a pure shift; check the sign of xi')

# (3) the free surface. Only the OUTPUT bar's far end is free -- the input bar
#     ends on the anvil -- so the input row is reported, never asserted.
w = slice(int(0.15 * N), int(ARGS.null_window * N))
print(f'\nfree-surface residual rms|eps+ + eps-| / peak|eps+|, '
      f'windowed to {ARGS.null_window:.2f} of the record:')
print(f'{"bar":>5} {"far end":>10} {"at the end":>12} {"10 mm in":>10} '
      f'{"unwindowed":>12}')
# Whether a bar's far end is a free surface is geometry, not a constant: the
# SHTB's input bar ends on the anvil, but the compression bar's ends free. A
# bar that reaches its own far end has nothing beyond it to reflect off.
for bar, sig, pos, L, L_free in (
        ('in', d['eps_in'], d['pos_in'], L_BAR_IN, L_FREE_IN),
        ('out', d['eps_out'], d['pos_out'], L_BAR_OUT, L_FREE_OUT)):
    sigs = [sig[k] for k in range(sig.shape[0])]
    p, m, _ = separate_field(t, sigs, list(pos), c0=c0, eta=ETA,
                             x=[L, L - 10.0], decimate=1, chunk=ARGS.chunk)
    pk = np.abs(p[0]).max()
    r_end = np.sqrt(np.mean((p[0] + m[0])[w] ** 2)) / pk
    r_10 = np.sqrt(np.mean((p[1] + m[1])[w] ** 2)) / pk
    r_full = np.sqrt(np.mean((p[0] + m[0]) ** 2)) / pk
    tag = 'free' if abs(L - L_free) < 0.5 * dx else 'not free'
    print(f'{bar:>5} {tag:>10} {r_end:12.2e} {r_10:10.2e} {r_full:12.2e}')
print('   a free end nulls; an end with something beyond it does not, and')
print('   should not -- the "not free" rows are reported, never asserted.')
print('   note the unwindowed column: exp(+eta t) amplifies the record-end')
print('   truncation, so the same good calibration reads ~200x worse there.')

# (4) tie to the quantity sep_test.py already validates against the simulator.
print('\ninterface force E*A*(eps+ + eps-) vs the simulator\'s own:')
for bar, sig, pos, truth in (('in', d['eps_in'], d['pos_in'], d['force_iface_in']),
                             ('out', d['eps_out'], d['pos_out'], d['force_iface_out'])):
    sigs = [sig[k] for k in range(sig.shape[0])]
    p, m, _ = separate_field(t, sigs, list(pos), c0=c0, eta=ETA, x=[0.0],
                             decimate=1, chunk=ARGS.chunk)
    F = E * A * (p[0] + m[0])
    print(f'   {bar:>3}  rel L2 {np.linalg.norm(F[w]-truth[w])/np.linalg.norm(truth[w]):.2e}')

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt              # backend already chosen
from matplotlib.colors import LinearSegmentedColormap, Normalize

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d3'
AQUA, SURFACE = '#1baf7a', '#fcfcfb'

# Diverging: two hues with a NEUTRAL midpoint, per the reference palette --
# blue for one sign, red for the other, grey #f0efec at zero. Equal step count
# per arm. A quiescent bar therefore reads as flat grey, and a MASKED region as
# a darker grey with hatching over it, so "nothing happening" and "no answer
# here" can never be confused -- including in greyscale, which is why the hatch
# is there as well as the tone.
CMAP = LinearSegmentedColormap.from_list('wave', [
    '#184f95', '#2a78d6', '#86b6ef', '#cde2fb',
    '#f0efec',
    '#f6c9c8', '#ee8a89', '#e34948', '#8f2322'])
CMAP = CMAP.copy()
CMAP.set_bad('#c9c8c3')

# One symmetric scale shared by all three panels: the comparison between them is
# the point, and per-panel limits would hide that the sum is genuinely largest.
# A robust percentile rather than the max, so a single ringing cell cannot crush
# the pulse.
_finite = np.abs(np.concatenate([eps_p[~invalid], eps_m[~invalid],
                                 eps_sum[~invalid]]))
VMAX = float(np.nanpercentile(_finite, 99.9)) * 1e6
NORM = Normalize(-VMAX, +VMAX)
_clipped = 100.0 * np.mean(_finite * 1e6 > VMAX)
print(f'\ncolour scale : +-{VMAX:.0f} ustrain (99.9th percentile), '
      f'{_clipped:.2f} % of cells clipped')

fig, axes = plt.subplots(1, 3, figsize=(14, 9.5), sharex=True, sharey=True,
                         layout='constrained')
fig.patch.set_facecolor(SURFACE)

_h = 0.5 * stride * dx                                  # half a cell
EXTENT = [X[0] - _h, X[-1] + _h, t_img[0], t_img[-1]]
PANELS = (
    (eps_p, r'(a)  $\varepsilon_+$ — away from the specimen'),
    (eps_m, r'(b)  $\varepsilon_-$ — toward the specimen'),
    (eps_sum, r'(c)  $\varepsilon_+ + \varepsilon_-$ — the total field'),
)
for ax, (field, title) in zip(axes, PANELS):
    im = ax.imshow(np.ma.masked_invalid(field * 1e6).T, origin='lower',
                   aspect='auto', extent=EXTENT, cmap=CMAP, norm=NORM,
                   interpolation='nearest')
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=8)

    # masked material, hatched so the distinction survives greyscale
    for lo, hi in ((0.0, X_IN - L_BAR_IN), (X_IN, X_OUT),
                   (X_OUT + L_BAR_OUT, X_TOTAL)):
        if hi > lo:
            ax.axvspan(lo, hi, facecolor='none', hatch='///', edgecolor=GRID,
                       linewidth=0.0, zorder=3)
    # the striker overlaps the bar but is NOT invalid -- tint only
    ax.axvspan(X_IN - L_BAR_IN, X_IN - L_BAR_IN + 800.0, facecolor=MUTED,
               alpha=.05, zorder=2)
    # The specimen is 10 mm on a 6030 mm axis -- narrower than one pixel, so a
    # span alone would vanish and the two halves would read as one field. They
    # are NOT: left and right are separate solves that share no data. Draw the
    # divide explicitly, in the surface colour so it reads as a cut.
    ax.axvline(0.5 * (X_IN + X_OUT), color=SURFACE, lw=3.0, zorder=4)
    for xv in (X_IN, X_OUT):
        ax.axvline(xv, color=MUTED, lw=.7, ls=':', alpha=.85, zorder=5)

axes[0].set_ylabel('Time (ms)')

for ax in axes:
    ax.set_xlabel('Position along the assembly (mm)')

# The gauges: the only four stations constrained by data. Marked once, on the
# panel where the claim is checked.
g_all = list(X_IN - np.asarray(d['pos_in'])) + list(X_OUT + np.asarray(d['pos_out']))
axes[2].plot(g_all, [t_img[0]] * len(g_all), marker='^', ms=7, ls='none',
             color=AQUA, clip_on=False, zorder=6)
axes[2].annotate('gauges — the only x constrained by data',
                 (g_all[0], t_img[0]), xytext=(0, 14),
                 textcoords='offset points', fontsize=8, color=MUTED, ha='center')

# Characteristic slope key. Put it in the output bar's quiet corner -- before
# the pulse has crossed the specimen nothing is happening there, so the key sits
# on blank surface rather than on data.
_x0 = X_OUT + 0.20 * L_BAR_OUT
_dx_ref = 0.55 * L_BAR_OUT
_t0 = t_img[0] + 0.04 * (t_img[-1] - t_img[0])
axes[0].plot([_x0, _x0 + _dx_ref], [_t0, _t0 + _dx_ref / c0], color=INK,
             lw=1.3, alpha=.6, zorder=6)
axes[0].annotate(f'a characteristic: slope 1/c₀ = {1000/c0:.3f} µs/mm',
                 (_x0, _t0), xytext=(3, -12), textcoords='offset points',
                 fontsize=8, color=MUTED, zorder=6)

for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

cb = fig.colorbar(im, ax=axes, location='right', shrink=.85, pad=.015,
                  aspect=42, extend='both')
cb.set_label(f'Strain (µstrain, {LOADING} positive)', color=MUTED, fontsize=10)
cb.ax.tick_params(colors=MUTED, labelsize=9)
cb.outline.set_visible(False)

fig.suptitle('Separated waves as a field: each bar reconstructed from its own '
             f'{n_in} gauges, at {stride*dx:.0f} mm stations',
             x=.007, ha='left', fontsize=13, color=INK)

# Everything the picture cannot say for itself goes in a footnote: there is no
# band of the plot blank in all three panels, and text over a wavefield is
# unreadable. Reserve the strip first, or constrained layout will overrun it.
_note = textwrap.fill(
    f'Layout: anvil 0–{X_IN-L_BAR_IN:.0f} | input bar '
    f'{X_IN-L_BAR_IN:.0f}–{X_IN:.0f} | specimen {X_IN:.0f}–{X_OUT:.0f} | '
    f'output bar {X_OUT:.0f}–{X_OUT+L_BAR_OUT:.0f} mm. '
    'Hatched = the two-wave model does not apply (anvil, specimen). Tinted = '
    'the striker rides over that stretch; the bar under it is ordinary and IS '
    'reconstructed, but the striker\'s own strain is never recorded. '
    'The two halves are INDEPENDENT solves — no gauge on one bar enters the '
    'other, and they meet only across the specimen. (a) and (b) are exact time '
    'shifts of a single signal each, so their amplitude is constant along x; '
    '(c) is not, and is where the physics is.', width=175)
# Hung BELOW the axes and picked up by bbox_inches='tight' at save time, rather
# than carved out of the figure with a layout rect -- constrained layout is
# already placing the suptitle, and a rect makes the two fight over that band.
fig.text(.007, -.012, _note, ha='left', va='top', fontsize=8.5, color=MUTED,
         linespacing=1.5, transform=fig.transFigure)

fig.savefig('lagrange_diagram.png', dpi=140, facecolor=fig.get_facecolor(),
            bbox_inches='tight')
print('\nwrote lagrange_diagram.png')

plotting.show_unless(HEADLESS)
