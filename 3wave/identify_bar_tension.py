"""
Identify gauge positions, gauge spacing and the bar wave speed from a
connected-bar calibration shot -- no specimen, both bars bolted together.

    python3 drive_calibration_tension.py
    python3 identify_bar_tension.py [--headless]

Runs on the rig's OWN striker. The 800 mm POM tube gives a 1097 us pulse against
a 2376 us assembly round trip, so echoes overlap the direct pulse and whole-pulse
matched filtering fails outright (10-50 % errors, measured). Everything below
therefore times EDGES on the differentiated record instead, which does not care
how long the pulse is.

--------------------------------------------------------------------------
What is and is not identifiable
--------------------------------------------------------------------------
A strain record is a function of time, and every arrival in it is some path
length divided by c0. The whole data set is therefore invariant under

    (all lengths, c0)  ->  (lambda * all lengths, lambda * c0)

No amount of timing breaks that: the experiment fixes every length only up to
one overall scale, and exactly ONE measured length has to be supplied.

This script asks for the least painful one: the distance from a single gauge --
the one the wave reaches first, which is the one furthest from the free end --
to the far free end. Call it xi_ref. Everything else is then leverage:

    d(c0)/c0 = d(D)/D = d(xi_ref)/xi_ref

so a tape measurement good to +-2 mm over the ~3.5 m reference baseline lands D
to +-0.23 mm on a 400 mm spacing. THE POINT IS THE RATIO xi_ref/D: a sloppy
measurement on a long baseline buys a sharp one on a short baseline, which is
exactly the trade you want, because the short baseline is the one you cannot
measure and the long one is the one you can.

No assumption is made that the two bars are instrumented symmetrically. The
script MEASURES the asymmetry of each nominal pair instead, and reports it.

That trade is better still because of what the reduction actually consumes. In
`separate` the positions enter only as

    xi * x_k = (w - i eta) * x_k / c0

so the result depends on the TRANSIT TIMES x_k/c0 and on nothing else --
verified: scaling positions and c0 together by any factor changes the separated
waves by 4e-14 relative. c0 alone is still needed, but only in `bar_interface`,
where it converts strain to particle velocity LINEARLY.

DENSITY IS NOT IDENTIFIABLE from strain records, at any scale. Strain is
dimensionless and time is all the record carries, so the shot fixes c0 =
sqrt(E/rho) and never E and rho separately. Breaking that needs one absolute
force or mass measurement; weighing the bar is the easy one. The reduction never
wants rho -- it wants E*A, a force scale -- so calibrate E*A directly, from a
static load or from striker momentum, and treat rho as a by-product.

--------------------------------------------------------------------------
How the record is read
--------------------------------------------------------------------------
With the bars joined and no specimen the assembly is one uniform bar with a
reflector at each end. A gauge xi from the free end and a from the anvil end
sees, in the DERIVATIVE of its record,

    delay 0          + edge   the pulse arriving
    delay P          - edge   its own trailing edge (P = striker pulse length)
    delay 2 xi / c0  - edge   the free-end echo arriving, INVERTED
    delay 2 xi/c0+P  + edge   that echo's trailing edge

P is the same at every gauge and 2 xi / c0 is not, which is how the two negative
edges are told apart with nothing assumed about the striker. Where they happen
to land within one edge width of each other the gauge is simply dropped from the
c0 average -- its position still comes through, from the gauge-to-gauge lag.

Only the FREE end is ever used. The anvil end is not a clean reflector: the
anvil is a lumped mass rather than a termination, and it reflects like a free
end displaced outward -- measured at +257 mm on this rig, against the 349 mm its
added mass m/(rho A) would suggest, so it cannot be modelled away either. A
round-trip estimate of c0 built on it comes out 4.1 % low. On a compression SHPB
struck directly on a genuinely free end that route is available and needs no
reference length at all.
"""
import numpy as np

import plotting
HEADLESS = plotting.init(__doc__)   # picks the backend; must precede pyplot

import config
from dump import load_dump

# --------------------------------------------------------------------------
# The measurements a tape and a scale supply. Bar lengths are easy; the gauge
# positions are not, and are never read here -- they are what is recovered.
# --------------------------------------------------------------------------
cfg = config.load('calibration_tension')
L_OUTPUT = cfg['bar']['L_output']            # output-bar face -> free end [mm]
# A bolted or threaded coupling has a thickness, and the model carries one
# element of it. Input-bar distances are quoted from the input face, that much
# further from the free end. Zero for bars butted directly together.
L_JOINT = cfg['specimen']['length']
L_ASSEMBLY = cfg['bar']['L_input'] + L_JOINT + L_OUTPUT
DIAMETER = cfg['bar']['diameter']
AREA = 0.25 * np.pi * DIAMETER ** 2

# Tape precision on the reference length, used only to quote the error bar the
# result inherits. Set it to what you can actually measure.
XI_REF_TOLERANCE = 2.0                       # [mm]


# --------------------------------------------------------------------------
# signal processing
# --------------------------------------------------------------------------
def _xcorr(a, b):
    """c[k] = sum_j a[j+k] b[j] for k >= 0, by FFT, zero-padded so nothing wraps."""
    n = len(a) + len(b) - 1
    nf = 1 << int(np.ceil(np.log2(n)))
    c = np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)
    return c[:len(a)]


def _refine(y, i):
    """Sub-sample offset of the extremum near index i, by a 3-point parabola."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    d = y[i - 1] - 2.0 * y[i] + y[i + 1]
    return 0.0 if d == 0 else 0.5 * (y[i - 1] - y[i + 1]) / d


def _extremum(c, lo, hi, sign):
    """Sub-sample index of the strongest peak of the given sign in [lo, hi)."""
    lo, hi = max(0, int(lo)), min(len(c), int(hi))
    seg = c[lo:hi] * sign
    i = lo + int(np.argmax(seg))
    return i + _refine(c * sign, i), c[i]


def _rise_index(g, frac=0.3):
    """First sample of the leading edge, from the differentiated record."""
    return int(np.argmax(np.abs(g) > frac * np.abs(g).max()))


# --------------------------------------------------------------------------
# load, differentiate
# --------------------------------------------------------------------------
d = load_dump()
t, dt, N = d['t'], d['dt'], d['N']
n_in = d['eps_in'].shape[0]
names = [f'in-{k}' for k in range(n_in)] + \
        [f'out-{k}' for k in range(d['eps_out'].shape[0])]
signals = [d['eps_in'][k] for k in range(n_in)] + \
          [d['eps_out'][k] for k in range(d['eps_out'].shape[0])]
grads = [np.gradient(s, dt) for s in signals]
true_pos = list(d['pos_in']) + list(d['pos_out'])   # gauge -> its own bar face

print(__doc__.split('---')[0].strip())
print(f'\nrecord            : {N} samples at {dt*1e3:.4f} us  ({t[-1]:.3f} ms)')
print(f'tape measurements : assembly {L_ASSEMBLY:.1f} mm, output bar '
      f'{L_OUTPUT:.1f} mm, joint {L_JOINT:.1f} mm, diameter {DIAMETER:.2f} mm')
print(f'gauges            : {len(names)}, positions NOT read from config')

# The leading edge is ~59 us wide (10-90 %) on this rig. The template spans a
# little more than that: long enough for a sharp correlation peak, short enough
# that two edges 0.1 ms apart still resolve into two peaks.
EDGE_MS = 0.12
n_t = int(round(EDGE_MS / dt))

# --------------------------------------------------------------------------
# arrival of the direct pulse at each gauge, and the lag between gauges
# --------------------------------------------------------------------------
# One common template, taken from whichever gauge the wave reaches first, so
# every lag is measured against the same feature.
i_first = int(np.argmin([_rise_index(g) for g in grads]))
ir = _rise_index(grads[i_first])
TEMPLATE = grads[i_first][ir - n_t // 4: ir + 3 * n_t // 4]

arrival = []
for g in grads:
    c = _xcorr(g, TEMPLATE)
    i, _ = _extremum(c, 0, len(c), +1)
    arrival.append(i * dt)
arrival = np.array(arrival)

REF = int(np.argmin(arrival))     # earliest arrival = furthest from the free end
lag = arrival - arrival[REF]      # >= 0, = (xi_ref - xi_k) / c0

print(f'\nreference gauge   : {names[REF]} (earliest arrival -> longest '
      f'baseline to the free end)')

# --------------------------------------------------------------------------
# the two negative edges: the striker's trailing edge, and the free-end echo
# --------------------------------------------------------------------------
def candidates(g, t_direct, n_want=4):
    """Delays of the strongest negative edges after the direct arrival."""
    c = _xcorr(g, TEMPLATE)
    work = c.copy()
    work[:int((t_direct / dt) + 1.5 * n_t)] = 0.0
    out = []
    for _ in range(n_want):
        i, v = _extremum(work, 0, len(work), -1)
        out.append((i * dt - t_direct, v))
        # exclude only half a template either side, or a close second edge
        # (the trailing edge and the echo can be ~0.1 ms apart) is swallowed
        work[max(0, int(i) - n_t // 2): int(i) + n_t // 2] = 0.0
    return sorted(out)


cands = [candidates(g, a) for g, a in zip(grads, arrival)]

# P is the delay common to EVERY gauge; the echo delay is not. Nothing about the
# striker has to be known for this -- it falls out of the comparison.
# A gauge whose echo happens to land within an edge width of P shows one merged
# peak instead of two, so P is taken by MAJORITY rather than by unanimity.
TOL = 0.6 * EDGE_MS
best, P = 0, None
for cand, _ in [c for cs in cands for c in cs]:
    hits = [min((x for x, _ in cs), key=lambda x: abs(x - cand))
            for cs in cands if any(abs(x - cand) < TOL for x, _ in cs)]
    if len(hits) > best:
        best, P = len(hits), float(np.median(hits))
if P is None or best < 2:
    raise SystemExit('no pulse length shared by at least two gauges; '
                     'check the record')

tau = []
for cs in cands:
    pick = [x for x, _ in cs if abs(x - P) > TOL]
    tau.append(pick[0] if pick else np.nan)
tau = np.array(tau)

print(f'striker pulse P   : {P*1e3:.1f} us (the delay shared by {best} of '
      f'{len(names)} gauges)')

print('\n--- edges, per gauge '
      '------------------------------------------------------')
print(f'{"gauge":>7} {"peak":>10} {"arrival":>9} {"lag vs ref":>11} '
      f'{"2xi/c0":>10}')
print(f'{"":>7} {"[ustrain]":>10} {"[ms]":>9} {"[us]":>11} {"[ms]":>10}')
for k, nm in enumerate(names):
    tk = '  merged' if np.isnan(tau[k]) else f'{tau[k]:10.5f}'
    print(f'{nm:>7} {np.abs(signals[k]).max()*1e6:10.1f} {arrival[k]:9.4f} '
          f'{lag[k]*1e3:11.4f} {tk}')

# --------------------------------------------------------------------------
# c0 from the ONE measured length
# --------------------------------------------------------------------------
# xi_k = xi_ref - c0 * lag_k, and tau_k = 2 xi_k / c0, so
#     Q_k = tau_k + 2 lag_k = 2 xi_ref / c0
# is the same for every gauge. Gauges whose two negative edges merged drop out;
# the rest are averaged, and their spread is a genuine consistency check.
# XI_REF is THE tape measurement -- the one length the experiment cannot supply
# itself. On the rig you measure it once, from the reference gauge to the far
# free end, to whatever precision you can manage. Here it stands in for that
# reading and is taken from the model geometry at this single point in the
# script; nothing else below consults the true geometry except the error columns.
X_TOTAL = d['X_OUT'] + d['L_free_out']
_x_ref = (d['X_IN'] - d['pos_in'][REF]) if REF < n_in else \
         (d['X_OUT'] + d['pos_out'][REF - n_in])
XI_REF = X_TOTAL - _x_ref

# Q must be the same at every gauge, so a gauge whose two negative edges merged
# into one -- and whose tau is therefore some later echo -- shows up as a gross
# outlier and is thrown out. With three or more gauges this needs no threshold
# tuning: the good ones agree to a fraction of a microsecond.
Q = tau + 2.0 * lag
ok = ~np.isnan(Q)
ok &= np.abs(Q - np.median(Q[ok])) < 0.01 * np.median(Q[ok])
if ok.sum() < 1:
    raise SystemExit('no gauge gave a usable free-end echo')
c0_id = 2.0 * XI_REF / np.mean(Q[ok])

print('\n--- wave speed '
      '------------------------------------------------------------')
print(f'reference length xi_ref ({names[REF]} -> free end) : {XI_REF:.1f} '
      f'+- {XI_REF_TOLERANCE:.1f} mm  (tape)')
print(f'{"gauge":>7} {"Q = 2xi_ref/c0 [ms]":>22}')
for k, nm in enumerate(names):
    note = f'{Q[k]:22.5f}' if ok[k] else (
        f'{"dropped: edges merged":>22}' if np.isnan(Q[k]) else
        f'{Q[k]:16.5f} rejected')
    print(f'{nm:>7} {note}')
print(f'  mean {np.mean(Q[ok]):.5f} ms over {ok.sum()} gauges, '
      f'spread {np.ptp(Q[ok])*1e3:.3f} us ({np.ptp(Q[ok])/np.mean(Q[ok]):.1e})')
print(f'\nc0 = 2 xi_ref / Q  : {c0_id:.3f} mm/ms')
print(f'c0 true            : {d["c0"]:.3f} mm/ms   rel err {(c0_id/d["c0"]-1):+.2e}')
print(f'tape contributes   : +-{XI_REF_TOLERANCE/XI_REF:.1e} '
      f'(+-{c0_id*XI_REF_TOLERANCE/XI_REF:.2f} mm/ms), which dominates '
      f'everything else')

# --------------------------------------------------------------------------
# positions -- from the lags, so every gauge gets one, merged edges or not
# --------------------------------------------------------------------------
xi = XI_REF - c0_id * lag
id_pos = np.where(np.arange(len(names)) >= n_in,
                  L_OUTPUT - xi, xi - L_OUTPUT - L_JOINT)

print('\n--- gauge positions '
      '-------------------------------------------------------')
print(f'{"gauge":>7} {"xi (to free end)":>18} {"x (from face)":>15} '
      f'{"true":>9} {"error":>9}')
for k, nm in enumerate(names):
    print(f'{nm:>7} {xi[k]:18.2f} {id_pos[k]:15.2f} {true_pos[k]:9.2f} '
          f'{id_pos[k]-true_pos[k]:+9.3f}')

print('\n--- gauge spacing D '
      '-------------------------------------------------------')
print(f'{"bar":>7} {"lag [us]":>12} {"D = c0 dt":>12} {"true":>9} {"error":>9}')
for bar, off, cnt in (('in', 0, n_in), ('out', n_in, len(names) - n_in)):
    if cnt < 2:
        continue
    dl = abs(lag[off + 1] - lag[off])
    D_id = c0_id * dl
    D_true = abs(true_pos[off + 1] - true_pos[off])
    print(f'{bar:>7} {dl*1e3:12.4f} {D_id:12.3f} {D_true:9.2f} '
          f'{D_id-D_true:+9.3f}')

# The errors above are what the TIMING costs; xi_ref is exact in a simulation
# and will not be on a rig. Everything scales with it linearly, so the tape
# error simply multiplies through -- and the ratio xi_ref/D is the leverage.
_f = XI_REF_TOLERANCE / XI_REF
print(f'\nwith a tape good to +-{XI_REF_TOLERANCE:.1f} mm on the '
      f'{XI_REF:.0f} mm reference baseline, every length above carries a '
      f'further\n+-{_f:.1e} relative: +-{c0_id*_f:.1f} mm/ms on c0 and '
      f'+-{c0_id*abs(lag[1]-lag[0])*_f:.2f} mm on D. The leverage is the ratio '
      f'xi_ref/D = {XI_REF/(c0_id*abs(lag[1]-lag[0])):.1f}:\na sloppy '
      f'measurement on a long baseline buys a sharp one on a short baseline.')

# --------------------------------------------------------------------------
# the symmetry that was NOT assumed, measured instead
# --------------------------------------------------------------------------
n_pair = min(n_in, len(names) - n_in)
if n_pair:
    print('\n--- how well matched are the two bars? '
          '------------------------------------')
    print('nothing above assumed the pairs are symmetric. This is what they '
          'actually are:\n')
    print(f'{"pair":>7} {"input x":>10} {"output x":>10} {"asymmetry":>11} '
          f'{"true":>9}')
    for k in range(n_pair):
        a, b = id_pos[k], id_pos[n_in + k]
        ta, tb = true_pos[k], true_pos[n_in + k]
        print(f'{k:>7} {a:10.2f} {b:10.2f} {a-b:+11.3f} {ta-tb:+9.3f}')

# --------------------------------------------------------------------------
# transit times -- what separate() actually consumes
# --------------------------------------------------------------------------
print('\n--- transit times, which is what separate() really needs '
      '-------------')
print('separate() depends on x_k/c0 only, so these are scale-free: an error in\n'
      'xi_ref cancels between the c0 and the position derived from it.\n')
print(f'{"gauge":>7} {"x/c0 [us]":>12} {"true [us]":>12} {"rel err":>10}')
for k, nm in enumerate(names):
    a, b = id_pos[k] / c0_id, true_pos[k] / d['c0']
    print(f'{nm:>7} {a*1e3:12.4f} {b*1e3:12.4f} {a/b-1:+10.2e}')

# --------------------------------------------------------------------------
# density: NOT identifiable from the records; closed with the bar's mass
# --------------------------------------------------------------------------
m_bar = d['rho'] * AREA * L_ASSEMBLY
E_id = (m_bar / (AREA * L_ASSEMBLY)) * c0_id ** 2

print('\n--- density and modulus '
      '---------------------------------------------------')
print('NOT identifiable from strain records: they fix c0 = sqrt(E/rho) and no '
      'more.\nClosed here with one extra measurement, the bar mass. In the lab '
      'that is an\nindependent weighing; HERE it is computed back from the '
      "simulator's own rho,\nso the rho line is circular. The E line is not: "
      'it uses the IDENTIFIED c0.\n')
print(f'bar mass (weighed)      : {m_bar*1e3:.1f} g')
print(f'rho = m / (A L)         : {m_bar/(AREA*L_ASSEMBLY):.4e} kg/mm^3   '
      f'(circular)')
print(f'E   = rho c0^2          : {E_id:.3f} GPa   '
      f'(true {d["E"]:.3f}, rel err {E_id/d["E"]-1:+.1e})')
print(f'E*A (the force scale)   : {E_id*AREA:.1f} kN   '
      f'(true {d["E"]*d["A"]:.1f})')

print('\n--- ready to use '
      '----------------------------------------------------------')
print(f'  c0 = {c0_id:.3f}')
print(f'  gauges = [{", ".join(f"{p:.2f}" for p in id_pos[:n_in])}]'
      '    # input bar, mm from its face')
print(f'  gauges = [{", ".join(f"{p:.2f}" for p in id_pos[n_in:])}]'
      '    # output bar')

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.patch.set_facecolor('#fcfcfb')

k = REF
axes[0].plot(t, signals[k] * 1e6, color=BLUE, lw=.9)
axes[0].set_ylabel('Strain (ustrain)')
axes[0].set_title(f'Calibration shot at gauge {names[k]} — 1097 us pulse, so '
                  'the echo overlaps the direct pulse', loc='left', fontsize=11)

axes[1].plot(t, _xcorr(grads[k], TEMPLATE) / np.abs(_xcorr(grads[k], TEMPLATE)).max(),
             color=INK, lw=.9)
for lab, tt, col in (('arrival', arrival[k], ORANGE),
                     ('trailing edge P', arrival[k] + P, MUTED),
                     ('free-end echo', arrival[k] + tau[k], ORANGE)):
    if not np.isnan(tt):
        axes[1].axvline(tt, color=col, lw=1.1, ls='--')
        axes[1].annotate(lab, (tt, 1.0), rotation=90, fontsize=8,
                         color=MUTED, va='top', ha='right')
axes[1].axhline(0, color=GRID, lw=.8)
axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Edge filter (norm.)')
axes[1].set_title('Differentiated record, matched against the leading edge — '
                  'edges are timed, not pulses', loc='left', fontsize=11)
axes[1].set_xlim(0, min(t[-1], arrival[k] + 2.2 * P))

for ax in axes:
    ax.set_facecolor('#fcfcfb'); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.tight_layout()
fig.savefig('bar_identification_tension.png', dpi=140, facecolor=fig.get_facecolor())
print('\nwrote bar_identification_tension.png')

plotting.show_unless(HEADLESS)
