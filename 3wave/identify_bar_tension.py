"""
Identify gauge positions, gauge spacing and the bar wave speed from a
connected-bar calibration shot -- no specimen, both bars bolted together.

    python3 drive_calibration_tension.py
    python3 identify_bar_tension.py [--headless]
    python3 identify_bar_tension.py --xi-ref 3679.5 --xi-ref-tol 2.0

Runs on the rig's OWN striker. The 800 mm POM tube gives a 1097 us pulse against
a 2435 us assembly round trip, so echoes overlap the direct pulse and whole-pulse
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
by preference the one the wave reaches first, which is the one furthest from the
free end -- to the far free end. Call it xi_ref. It comes from

    xi_ref, xi_ref_gauge, xi_ref_tol   in [calibration_tension] of config.toml
    --xi-ref / --xi-ref-tol            overriding those, to sweep sensitivity

and if none of them is set the script falls back to the MODEL's geometry, which
a simulation can supply and a rig cannot. That fallback is the self-check mode;
supplying xi_ref is what makes this an instrument. Any gauge may carry the tape
-- xi_k = xi_ref (1 - 2 lag_k / Q) inverts to refer it back to the reference
gauge -- but a short baseline divides the tolerance up by the same factor, so
measure the longest one you can reach.

Everything else is then leverage:

    d(c0)/c0 = d(D)/D = d(xi_ref)/xi_ref

so a tape measurement good to +-2 mm over the ~3.7 m reference baseline lands D
to +-0.22 mm on a 400 mm spacing. THE POINT IS THE RATIO xi_ref/D: a sloppy
measurement on a long baseline buys a sharp one on a short baseline, which is
exactly the trade you want, because the short baseline is the one you cannot
measure and the long one is the one you can.

That leverage does NOT extend to the gauge positions x. Those are xi plus a
constant the tape error never touches, so they inherit an ABSOLUTE band of order
the tape error itself -- +-1.3 to +-2.0 mm here, reported per gauge in the
+-tape column. It is benign, because a common offset mostly moves where the wave
is reconstructed rather than distorting it, and because D is what the reduction
leans on; but it is not the small relative number, and this script used to imply
that it was.

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
import argparse

import numpy as np

import plotting

# argparse has to own the whole command line, and the backend has to be chosen
# before pyplot is imported, so both happen here at the top -- see plotting.py.
_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--xi-ref', type=float, metavar='MM',
                 help='reference length, gauge -> far free end [mm]. Overrides '
                      'xi_ref in config.toml. This is THE measured length the '
                      'shot cannot supply itself.')
_ap.add_argument('--xi-ref-tol', type=float, metavar='MM',
                 help='what the tape is good to [mm]. Overrides xi_ref_tol. '
                      'Propagated to every result below.')
HEADLESS, ARGS = plotting.init(parser=_ap)   # picks the backend; precedes pyplot

import config
from dump import load_dump
from wave_separation import separate

# --------------------------------------------------------------------------
# The measurements a tape and a scale supply. Bar lengths are easy; the gauge
# positions are not, and are never read here -- they are what is recovered.
# --------------------------------------------------------------------------
cfg = config.load('calibration_tension')
L_OUTPUT = cfg['bar']['L_output']            # output-bar face -> free end [mm]
# A bolted or threaded coupling has a thickness -- 150 mm on this rig -- and the
# model carries it as the "specimen". Input-bar distances are quoted from the
# input face, that much further from the free end. Zero for bars butted directly
# together.
#
# This length must be RIGHT, and so must the coupler's material. The coupler is
# bar stock at bar diameter, so it is acoustically invisible and only its LENGTH
# enters; a coupler of different impedance biases every result in proportion to
# L_JOINT * (1/c_bar - 1/c_joint), and does so invisibly -- the Q check below
# cannot see it. See "What a mismatched coupler costs" in README.md.
L_JOINT = cfg['specimen']['length']
L_ASSEMBLY = cfg['bar']['L_input'] + L_JOINT + L_OUTPUT
DIAMETER = cfg['bar']['diameter']
AREA = 0.25 * np.pi * DIAMETER ** 2

# THE tape measurement, and the only quantity here the experiment cannot supply
# itself -- see "What is and is not identifiable" above. config.toml is its
# durable home; the flags exist so its influence can be swept without editing
# the file. Absent from both, the script falls back to the model's own geometry
# further down, which makes it a self-check rather than an instrument.
XI_REF_CFG = ARGS.xi_ref if ARGS.xi_ref is not None else cfg.get('xi_ref')
XI_REF_GAUGE = cfg.get('xi_ref_gauge')       # None = "whichever is reference"
XI_REF_TOLERANCE = (ARGS.xi_ref_tol if ARGS.xi_ref_tol is not None
                    else cfg.get('xi_ref_tol', 2.0))    # [mm]


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
# Q must be the same at every gauge, so a gauge whose two negative edges merged
# into one -- and whose tau is therefore some later echo -- shows up as a gross
# outlier and is thrown out. With three or more gauges this needs no threshold
# tuning: the good ones agree to a fraction of a microsecond.
#
# Q is resolved BEFORE xi_ref because referring a tape reading taken at some
# other gauge back to the reference gauge needs it.
Q = tau + 2.0 * lag
ok = ~np.isnan(Q)
ok &= np.abs(Q - np.median(Q[ok])) < 0.01 * np.median(Q[ok])
if ok.sum() < 1:
    raise SystemExit('no gauge gave a usable free-end echo')
Q_MEAN = float(np.mean(Q[ok]))

# XI_REF is THE tape measurement -- the one length the experiment cannot supply
# itself. On the rig you measure it once, from a gauge to the far free end, to
# whatever precision you can manage, and put it in config.toml (or pass
# --xi-ref). The model's own geometry is the FALLBACK: a simulation can supply
# it and a rig cannot, so relying on it makes this a self-check rather than an
# instrument. Nothing else below consults the true geometry except the error
# columns.
X_TOTAL = d['X_OUT'] + d['L_free_out']
_x_ref = (d['X_IN'] - d['pos_in'][REF]) if REF < n_in else \
         (d['X_OUT'] + d['pos_out'][REF - n_in])
XI_REF_TRUE = X_TOTAL - _x_ref

if XI_REF_CFG is None:
    XI_REF = XI_REF_TRUE
    XI_REF_SRC = 'model geometry -- no xi_ref configured, so this is a SELF-CHECK'
elif XI_REF_GAUGE in (None, names[REF]):
    XI_REF = XI_REF_CFG
    XI_REF_SRC = f'tape to {names[REF]}'
elif XI_REF_GAUGE in names:
    # The tape may have reached any gauge, not the one the record happens to
    # pick as reference. xi_k = xi_ref (1 - 2 lag_k / Q) inverts to give xi_ref
    # from whichever gauge was actually measured -- at the cost of dividing the
    # tolerance by that same factor, so a short baseline is a worse buy.
    _k = names.index(XI_REF_GAUGE)
    _scale = 1.0 - 2.0 * lag[_k] / Q_MEAN
    if _scale <= 0:
        raise SystemExit(f'xi_ref_gauge {XI_REF_GAUGE!r} gives a non-positive '
                         'baseline; check the record')
    XI_REF = XI_REF_CFG / _scale
    XI_REF_TOLERANCE = XI_REF_TOLERANCE / _scale
    XI_REF_SRC = (f'tape to {XI_REF_GAUGE} ({XI_REF_CFG:.1f} mm), referred to '
                  f'{names[REF]} by /{_scale:.5f}')
else:
    raise SystemExit(f'xi_ref_gauge {XI_REF_GAUGE!r} is not one of {names}')

# The configured value depends on the gauge layout and the bar lengths, so it
# goes stale the moment either changes. In a simulation the true answer is right
# there; say so rather than quietly identifying the wrong bar.
if XI_REF_CFG is not None and abs(XI_REF - XI_REF_TRUE) > XI_REF_TOLERANCE:
    print(f'\n!! WARNING: xi_ref resolves to {XI_REF:.1f} mm but the model '
          f'geometry says {XI_REF_TRUE:.1f} mm,\n!! a slip of '
          f'{XI_REF - XI_REF_TRUE:+.1f} mm, outside the +-'
          f'{XI_REF_TOLERANCE:.1f} mm tolerance. Has the gauge layout or a\n'
          f'!! bar length changed since xi_ref was measured?')

c0_id = 2.0 * XI_REF / Q_MEAN

print('\n--- wave speed '
      '------------------------------------------------------------')
print(f'reference length xi_ref ({names[REF]} -> free end) : {XI_REF:.1f} '
      f'+- {XI_REF_TOLERANCE:.1f} mm')
print(f'  source          : {XI_REF_SRC}')
print(f'{"gauge":>7} {"Q = 2xi_ref/c0 [ms]":>22}')
for k, nm in enumerate(names):
    note = f'{Q[k]:22.5f}' if ok[k] else (
        f'{"dropped: edges merged":>22}' if np.isnan(Q[k]) else
        f'{Q[k]:16.5f} rejected')
    print(f'{nm:>7} {note}')
print(f'  mean {Q_MEAN:.5f} ms over {ok.sum()} gauges, '
      f'spread {np.ptp(Q[ok])*1e3:.3f} us ({np.ptp(Q[ok])/Q_MEAN:.1e})')
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

# What the tape costs each position. xi_k = xi_ref (1 - 2 lag_k / Q), so
# d(xi_k) = (xi_k / xi_ref) d(xi_ref); and x is xi plus a CONSTANT (L_OUTPUT,
# L_JOINT) that the tape error does not touch, so the position inherits that as
# an ABSOLUTE band rather than a relative one. It is much the largest entry on
# this table, and the only one that is not the timing's fault.
xi_band = XI_REF_TOLERANCE * xi / XI_REF

print('\n--- gauge positions '
      '-------------------------------------------------------')
print(f'{"gauge":>7} {"xi (to free end)":>18} {"x (from face)":>15} '
      f'{"+-tape":>8} {"true":>9} {"error":>9}')
for k, nm in enumerate(names):
    print(f'{nm:>7} {xi[k]:18.2f} {id_pos[k]:15.2f} {xi_band[k]:8.2f} '
          f'{true_pos[k]:9.2f} {id_pos[k]-true_pos[k]:+9.3f}')

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

# The errors in the "error" columns are what the TIMING costs, with xi_ref taken
# as exact. The tape error is separate and adds on top -- but NOT uniformly, and
# the distinction matters:
#
#   c0, D, xi   scale with xi_ref, so they carry a RELATIVE band. D is the one
#               the reduction actually leans on, and the ratio xi_ref/D is the
#               leverage that makes it small.
#   x (positions) are xi plus a constant, so they carry an ABSOLUTE band of
#               order the tape error itself -- ~10x worse in relative terms, and
#               NOT improved by the leverage. It is benign anyway, because a
#               common offset mostly shifts where the wave is reconstructed
#               rather than distorting it, but it is not the +-_f the older
#               version of this message implied.
_f = XI_REF_TOLERANCE / XI_REF
_D = c0_id * abs(lag[1] - lag[0])
print(f'\nwith a tape good to +-{XI_REF_TOLERANCE:.1f} mm on the '
      f'{XI_REF:.0f} mm reference baseline:\n'
      f'  c0, D, xi   +-{_f:.1e} relative: +-{c0_id*_f:.1f} mm/ms on c0, '
      f'+-{_D*_f:.2f} mm on D\n'
      f'  positions   +-{xi_band.min():.2f} to +-{xi_band.max():.2f} mm '
      f'ABSOLUTE (the +-tape column above)\n'
      f'The leverage on D is the ratio xi_ref/D = {XI_REF/_D:.1f}: a sloppy '
      f'measurement on a long\nbaseline buys a sharp one on a short baseline. '
      f'It does not help the positions.')

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
print('separate() depends on x_k/c0 only. Note these do NOT inherit the tape\n'
      'error as a small relative number: x is xi plus a constant the tape does\n'
      'not scale, so a tape error moves them absolutely. What IS scale-free is\n'
      'xi/c0 -- the free-end null below tests exactly that, and nothing else.\n')
print(f'{"gauge":>7} {"x/c0 [us]":>12} {"true [us]":>12} {"rel err":>10}')
for k, nm in enumerate(names):
    a, b = id_pos[k] / c0_id, true_pos[k] / d['c0']
    print(f'{nm:>7} {a*1e3:12.4f} {b*1e3:12.4f} {a/b-1:+10.2e}')

# --------------------------------------------------------------------------
# free-end null test -- the only check here that needs no ground truth
# --------------------------------------------------------------------------
# The far end of the output bar is a free surface, so the stress there is zero
# at all times and the two travelling waves must cancel:
#
#     eps_plus + eps_minus = 0     at xi = 0
#
# Reconstruct AT that surface -- hand `separate` the identified xi, which are
# distances from it -- and the boundary condition becomes a residual that should
# vanish. Nothing here consults the true geometry, which is what makes this the
# one validation that survives contact with a real rig, where there is no truth
# to compare c0 or the positions against.
#
# It also responds to a mismatched coupler, which the Q spread provably cannot:
# the coupler's extra transit time enters Q identically at every gauge and
# cancels out of the spread, but it does not cancel here. Do not oversell that:
# a 150 mm coupler at 0.9 rho moves this residual only from 1.2e-3 to 3.5e-3,
# because the null constrains xi/c0 on baselines of METRES, where the coupler's
# bias is relatively small. The damage downstream lands on x/c0 instead, over
# baselines of ~130 mm, where the same absolute error is 20x larger in relative
# terms -- which is why the reduction degrades more than this number suggests.
# Treat a FAIL as conclusive and a PASS as weak evidence.
#
# What it CANNOT do is break the scale degeneracy. Scaling xi and c0 together
# leaves the residual identical to seven digits (verified), because the test
# constrains the transit times xi/c0 and nothing else -- which is exactly the
# quantity separate() consumes, and exactly the quantity xi_ref cannot fix.
NULL_WINDOW = cfg.get('null_window', 0.75)
NULL_TOL = cfg.get('null_tol', 5.0e-3)

p_free, m_free = separate(t, signals, xi, c0=c0_id, eta=d['eta'])
_total = p_free + m_free
_amp = np.abs(p_free).max()

# The tail MUST be cut. The exponential window that regularises separate()
# amplifies the truncation at the end of the record, and over the FULL record
# the residual comes out ~100x larger than it really is -- 1.2e-01 against
# 1.2e-03 on a calibration that is in fact good. Start at the first arrival at
# the free end; stop before the truncation.
_i0 = int(np.argmax(np.abs(p_free) > 0.02 * _amp))
_i1 = int(NULL_WINDOW * N)
_w = slice(_i0, _i1)
null_rms = float(np.sqrt(np.mean(_total[_w] ** 2)) / _amp)
null_max = float(np.abs(_total[_w]).max() / _amp)

print('\n--- free-end null test (no ground truth used) '
      '-----------------------------')
print(f'reconstructed at the free surface from all {len(names)} gauges, '
      f'{t[_i0]:.2f}-{t[_i1]:.2f} ms')
print(f'peak |eps+|            : {_amp*1e6:.1f} ustrain')
print(f'residual |eps+ + eps-| : rms {null_rms:.2e}, max {null_max:.2e} '
      '(relative to peak |eps+|)')
print(f'threshold              : {NULL_TOL:.1e}   ->  '
      f'{"PASS" if null_rms <= NULL_TOL else "FAIL"}')
if null_rms > NULL_TOL:
    print('  The free surface does not come out stress-free, so the transit\n'
          '  times xi/c0 are wrong. Most likely: a coupler that is not bar\n'
          '  material, or the wrong coupler length. Note xi_ref is NOT the\n'
          '  suspect -- this test is blind to it.')

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
# Four panels, not two. The top pair share the edge-timing time window; the
# bottom pair show the free-end null over its own, much longer, window -- so
# sharex is set per pair rather than across the figure.
fig, axes = plt.subplots(4, 1, figsize=(13, 13))
fig.patch.set_facecolor('#fcfcfb')
axes[1].sharex(axes[0])
axes[3].sharex(axes[2])

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

# --- the free-end null, made visible ---------------------------------------
# Two panels rather than one with two y-scales: the waves are ~1000 ustrain and
# their sum is ~1, so a shared axis would render the sum as a flat line on zero
# and prove nothing. Separate panels let each be read at its own scale.
_sig = 1e6                                     # strain -> ustrain
axes[2].plot(t, p_free * _sig, color=BLUE, lw=.9, label=r'$\varepsilon_+$ (incident)')
axes[2].plot(t, m_free * _sig, color=ORANGE, lw=.9, label=r'$\varepsilon_-$ (reflected)')
axes[2].set_ylabel('Strain (ustrain)')
axes[2].set_title('Waves reconstructed AT the free surface — a free end inverts, '
                  'so these should be mirror images', loc='left', fontsize=11)
# upper LEFT: the record is quiescent before the first arrival, so the legend
# cannot collide with a trace there. Upper right is where the waves peak.
axes[2].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')

_band = NULL_TOL * _amp * _sig
axes[3].axhspan(-_band, _band, color=BLUE, alpha=.18,
                label=f'pass threshold ±{NULL_TOL:.1e} × peak |ε₊|')
axes[3].plot(t, _total * _sig, color=INK, lw=.9,
             label=r'$\varepsilon_+ + \varepsilon_-$  (= stress / E)')
axes[3].axhline(0, color=GRID, lw=.8)
for _b in (t[_i0], t[_i1]):                    # the window the rms is taken over
    axes[3].axvline(_b, color=ORANGE, lw=1.1, ls='--')
# bottom, not top: the legend owns the top-left corner of this panel
axes[3].annotate('analysis window', (t[_i0], 0.04), xytext=(4, 0),
                 textcoords='offset points', fontsize=8, color=MUTED,
                 xycoords=('data', 'axes fraction'), va='bottom')
axes[3].set_xlabel('Time (ms)'); axes[3].set_ylabel('Strain (ustrain)')
axes[3].set_title(f'Free-surface stress — rms {null_rms:.2e} of peak |ε₊|, '
                  f'{"PASS" if null_rms <= NULL_TOL else "FAIL"}. Outside the '
                  'window the record truncates', loc='left', fontsize=11)
axes[3].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='upper left')
# Scale to the residual inside the window, not to the truncation spike outside
# it, which is ~200x larger and would flatten everything worth seeing.
_r = np.abs(_total[_w]).max() * _sig
axes[3].set_ylim(-3 * _r, 3 * _r)
axes[2].set_xlim(0, t[-1]); axes[3].set_xlim(0, t[-1])

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
