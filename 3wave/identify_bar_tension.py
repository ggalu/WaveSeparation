# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-09-03 09:38:06
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-09-03 11:01:52
"""
Identify gauge positions, gauge spacing and the bar wave speed from a
connected-bar calibration shot -- no specimen, both bars bolted together.

    python3 drive_calibration_tension.py
    python3 identify_bar_tension.py [--headless]
    python3 identify_bar_tension.py --l-free-ref 3679.5 --l-free-ref-tol 2.0

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
free end -- to the far free end. Call it L_free_ref. It comes from

    L_free_ref, L_free_ref_gauge, L_free_ref_tol
                                 in [calibration_tension] of config.toml
    --l-free-ref / --l-free-ref-tol
                                 overriding those, to sweep sensitivity

and if none of them is set the script falls back to the MODEL's geometry, which
a simulation can supply and a rig cannot. That fallback is the self-check mode;
supplying L_free_ref is what makes this an instrument. Any gauge may carry the
tape -- L_free_k = L_free_ref (1 - 2 lag_k / Q) inverts to refer it back to the
reference gauge -- but a short baseline divides the tolerance up by the same
factor, so measure the longest one you can reach.

L_free is a DISTANCE, in mm, measured from the free surface: the same family as
the dump's L_free_in / L_free_out, which are that distance for a bar face rather
than for a gauge. It is NOT the complex wavenumber xi = (w - i eta)/c_p of the
separation theory, which is a different quantity with different units and lives
in wave_separation.py. The two used to share the name xi here, which is why the
next paragraph but one spells the distinction out.

--------------------------------------------------------------------------
Which tape number carries the scale: [.c0_route]
--------------------------------------------------------------------------
ONE length must be imported; WHICH one is a choice, and c0_route in
config.toml makes it explicit rather than implicit:

  "joint"          (default) c0 = 2 L_free_ref / mean(Q), and every position
                   from L_free_k = L_free_ref - c0 lag_k. Consumes ONLY
                   L_free_ref and the timings -- the configured gauge list is
                   never read. Averages over every gauge that passes the 1 %
                   outlier check, so any ONE bad echo is rejected. Its
                   weakness is that it reaches the free end from an in-bar
                   reference, crossing the threaded joint twice: whatever the
                   coupler costs in transit time is absorbed into c0.

  "out_echo"       c0 = mean of the two out-bar gauges' own free-end echoes,
                   2 (L_output - x_tape_k) / tau_k. Never crosses the joint --
                   but it READS THE OUT-BAR TAPE POSITIONS, the very quantities
                   this script reports as identified, and then places every
                   gauge from a DIFFERENT anchor, L_free_ref. Two anchors that
                   disagree do not cancel: on experiment_tension_bar_2 they
                   disagree by 68 mm, and the out gauges come back 58 and 69 mm
                   away from the same tape that set c0. Kept so the older
                   results stay reproducible. Do not choose it.

  "out_echo_diff"  c0 = 2 D_out / (tau_out0 - tau_out1), from the DIFFERENCE of
                   the two out-bar round trips, and every L_free from that
                   gauge's own round trip, L_free_k = c0 tau_k / 2. The one
                   tape number it consumes is the out-bar gauge SPACING: no
                   L_output, no L_free_ref, no joint length. Exactly ONE anchor,
                   so the out-bar positions come back at their own tape as a
                   real check rather than by construction -- both at -3.7 mm
                   here, which says L_output is 2780.7 mm and not the
                   configured 2777.

                   Same exposure as "out_echo" in one respect: it needs both
                   out-bar echoes and has no redundancy if one is mis-detected.
                   Check rows (3) and (4) of the c0 table against each other
                   before selecting it.

Under "out_echo_diff" the in bar is a separate acoustic leg, because the
joint's ACOUSTIC length is not its tape length -- 91.0 mm against 23.0 mm here,
13.3 us of excess transit. The timings cannot split that delay from the in-bar
gauges' own offsets, so one in-bar tape position anchors that leg and the
excess is reported as L_joint_eff instead of being smeared over every in-bar
position. The other two routes smear it: it is why "joint" puts in-1 at 138 mm
against a 119 mm tape.

Everything else is then leverage:

    d(c0)/c0 = d(D)/D = d(L_free_ref)/L_free_ref

so a tape measurement good to +-2 mm over the ~3.7 m reference baseline lands D
to +-0.22 mm on a 400 mm spacing. THE POINT IS THE RATIO L_free_ref/D: a sloppy
measurement on a long baseline buys a sharp one on a short baseline, which is
exactly the trade you want, because the short baseline is the one you cannot
measure and the long one is the one you can.

That leverage does NOT extend to the gauge positions x. Those are L_free plus a
constant the tape error never touches, so they inherit an ABSOLUTE band of order
the tape error itself -- +-1.3 to +-2.0 mm here, reported per gauge in the
+-tape column. It is benign, because a common offset mostly moves where the wave
is reconstructed rather than distorting it, and because D is what the reduction
leans on; but it is not the small relative number, and this script used to imply
that it was.

No assumption is made that the two bars are instrumented symmetrically. The
script MEASURES the asymmetry of each nominal pair instead, and reports it.

That trade is better still because of what the reduction actually consumes. In
`separate` a position x_k reaches the answer only through the phase of

    xi * x_k = (w - i eta) * x_k / c0

where xi is the COMPLEX WAVENUMBER of the separation -- 1/length, the (w - i
eta)/c_p of wave_separation.py -- and not any length identified here. That is
the one place the symbol is used in this sense, and it is why the distances
recovered below are called L_free rather than xi.

So the result depends on the TRANSIT TIMES x_k/c0 and on nothing else --
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
reflector at each end. A gauge L_free from the free end and a from the anvil end
sees, in the DERIVATIVE of its record,

    delay 0                + edge   the pulse arriving
    delay P                - edge   its own trailing edge (P = striker pulse)
    delay 2 L_free / c0    - edge   the free-end echo arriving, INVERTED
    delay 2 L_free/c0 + P  + edge   that echo's trailing edge

P is the same at every gauge and 2 L_free / c0 is not, which is how the two
negative edges are told apart with nothing assumed about the striker. Where they
happen to land within one edge width of each other the gauge is simply dropped
from the c0 average -- its position still comes through, from the gauge-to-gauge
lag.

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
# dest is spelled out because argparse would otherwise lower-case the L, and
# the capital is what marks this as a length rather than the separation's xi.
_ap.add_argument('--l-free-ref', type=float, metavar='MM', dest='L_free_ref',
                 help='reference length, gauge -> far free end [mm]. Overrides '
                      'L_free_ref in config.toml. This is THE measured length '
                      'the shot cannot supply itself.')
_ap.add_argument('--l-free-ref-tol', type=float, metavar='MM',
                 dest='L_free_ref_tol',
                 help='what the tape is good to [mm]. Overrides L_free_ref_tol. '
                      'Propagated to every result below.')
_ap.add_argument('--experiment', metavar='CASE', default=None,
                 help='identify a MEASURED shot named by a config case (e.g. '
                      'experiment_tension_bar) instead of the simulated '
                      'calibration dump. There is no ground truth then, so the '
                      'true/error columns print a dash.')
HEADLESS, ARGS = plotting.init(parser=_ap)   # picks the backend; precedes pyplot

import config
from dump import load_dump
from wave_separation import separate

CASE = ARGS.experiment or 'calibration_tension'
EXPERIMENT = CASE in config.EXPERIMENT_CASES

# --------------------------------------------------------------------------
# The measurements a tape and a scale supply. Bar lengths are easy; the gauge
# positions are not, and are never read here -- they are what is recovered.
# --------------------------------------------------------------------------
print("*********************")
print("*** CASE: ", CASE)
print("*********************")
cfg = config.load(CASE)

# The whole identification models the bolted-together assembly as ONE uniform
# bar of speed c0 -- that is what makes the echo train readable at all. So the
# two bar tables must agree here, even though config.toml keeps them separate
# for the compression case's sake. Refuse rather than quietly average them.
# A measured shot's bar tables need not carry E/rho at all (only length and
# diameter matter to the identification), so only keys present on BOTH sides
# are compared.
_IN_BAR, _OUT_BAR = cfg['input_bar'], cfg['output_bar']
for _k in ('E', 'rho', 'diameter'):
    if _k in _IN_BAR and _k in _OUT_BAR and _IN_BAR[_k] != _OUT_BAR[_k]:
        raise SystemExit(
            f"[{CASE}.input_bar] and [{CASE}.output_bar] "
            f"disagree on {_k!r} ({_IN_BAR[_k]} vs {_OUT_BAR[_k]}).\n"
            "This script identifies ONE uniform bar from its echo train; two "
            "different bars\nwould need a different method entirely. Make the "
            "two tables match in config.toml.")

L_OUTPUT = _OUT_BAR['L_output']              # output-bar face -> free end [mm]
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
L_ASSEMBLY = _IN_BAR['L_input'] + L_JOINT + L_OUTPUT
DIAMETER = _IN_BAR['diameter']
AREA = 0.25 * np.pi * DIAMETER ** 2

# THE tape measurement, and the only quantity here the experiment cannot supply
# itself -- see "What is and is not identifiable" above. config.toml is its
# durable home; the flags exist so its influence can be swept without editing
# the file. Absent from both, the script falls back to the model's own geometry
# further down, which makes it a self-check rather than an instrument.
L_FREE_REF_CFG = (ARGS.L_free_ref if ARGS.L_free_ref is not None
                  else cfg.get('L_free_ref'))
L_FREE_REF_GAUGE = cfg.get('L_free_ref_gauge')   # None = "whichever is ref"
L_FREE_REF_TOL = (ARGS.L_free_ref_tol if ARGS.L_free_ref_tol is not None
                  else cfg.get('L_free_ref_tol', 2.0))    # [mm]

# WHICH tape number carries the scale -- see "[.c0_route]" in the docstring
# above for what each one consumes and what it costs. "joint" is the default
# because it degrades gracefully: it averages over every gauge that survives
# the 1 % outlier check, so one bad echo cannot set c0 by itself. Opt a case
# into "out_echo_diff" once its two out-bar echoes are known to agree.
C0_ROUTES = ('joint', 'out_echo', 'out_echo_diff')
C0_ROUTE = cfg.get('c0_route')
if C0_ROUTE is None:
    # The boolean this key replaced. It only ever selected "out_echo", which
    # is the route that mixes two anchors -- map it, but say so, because the
    # mapping preserves a result that is wrong rather than fixing it.
    C0_ROUTE = 'out_echo' if cfg.get('use_only_out_for_c0', False) else 'joint'
    if 'use_only_out_for_c0' in cfg:
        print(f"[{CASE}] use_only_out_for_c0 is superseded by c0_route; "
              f"reading it as c0_route = {C0_ROUTE!r}.\n"
              "  That route anchors c0 on the out-bar tape positions and the "
              "gauge positions on\n  L_free_ref -- two anchors, which do not "
              "cancel. 'out_echo_diff' is the fixed one.")
if C0_ROUTE not in C0_ROUTES:
    raise SystemExit(f'[{CASE}] c0_route = {C0_ROUTE!r} is not one of '
                     f'{list(C0_ROUTES)}')

# What ONE tape gauge position is good to [mm]. Only "out_echo_diff" consumes
# it: the scale it imports is the out-bar SPACING, a difference of two such
# readings, and the in bar's leg is anchored on a third. The other routes
# import L_free_ref instead and carry L_free_ref_tol.
GAUGE_TOL = float(cfg.get('gauge_tol', 2.0))


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


def _rise_index(g, frac=0.3, hi=None):
    """First sample of the leading edge, from the differentiated record."""
    a = np.abs(g) if hi is None else np.abs(g[:hi])
    return int(np.argmax(a > frac * a.max()))


def _rise_time(g, dt):
    """
    10-90 % rise time of the leading edge, in ms.

    EDGE_MS is hardcoded to the SIMULATED rig's ~59 us edge everywhere below
    except for a measured (EXPERIMENT) shot, where it is measured from the
    record instead -- a real bar's edge need not match the model's.
    """
    a = np.abs(g)
    pk = a.max()
    i_pk = int(np.argmax(a))
    i_lo = int(np.argmax(a[:i_pk + 1] > 0.1 * pk)) if i_pk else 0
    return max((i_pk - i_lo), 1) * dt


# --------------------------------------------------------------------------
# load, differentiate
# --------------------------------------------------------------------------
if EXPERIMENT:
    from experiment import load_experiment
    d = load_experiment(CASE)
else:
    d = load_dump()
if d['loading'] != 'tension':
    raise SystemExit(
        f"this dump is a {d['loading']} shot; identify_bar_tension.py reads the "
        "SHTB's\nassembly echo train. Run drive_calibration_tension.py, or use "
        "identify_bar_compression.py\nif you meant the direct-impact rig.")
t, dt, N = d['t'], d['dt'], d['N']
n_in = d['eps_in'].shape[0]
names = [f'in-{k}' for k in range(n_in)] + \
        [f'out-{k}' for k in range(d['eps_out'].shape[0])]
signals = [d['eps_in'][k] for k in range(n_in)] + \
          [d['eps_out'][k] for k in range(d['eps_out'].shape[0])]
grads = [np.gradient(s, dt) for s in signals]
true_pos = list(d['pos_in']) + list(d['pos_out'])   # gauge -> its own bar face
_ref_lbl = 'tape' if EXPERIMENT else 'true'
# A dump carries strain; a measured record carries whatever the conditioner
# was calibrated in (force, here) -- print and scale accordingly rather than
# assuming ustrain.
UNITS = d.get('units', 'strain')
SCALE, USYM = (1.0, UNITS) if EXPERIMENT else (1e6, 'ustrain')
PKFMT = '10.4g' if EXPERIMENT else '10.1f'

print(__doc__.split('---')[0].strip())
if EXPERIMENT:
    print(f'\nMEASURED shot     : {d["source"]}')
    print('no ground truth   : the true/error columns print a dash. The only '
          'checks are\n                    internal -- the Q spread across '
          'gauges, and the free-end null.')
print(f'\nrecord            : {N} samples at {dt*1e3:.4f} us  ({t[-1]:.3f} ms)')
print(f'tape measurements : assembly {L_ASSEMBLY:.1f} mm, output bar '
      f'{L_OUTPUT:.1f} mm, joint {L_JOINT:.1f} mm, diameter {DIAMETER:.2f} mm')
print(f'gauges            : {len(names)}, positions NOT read from config')

# An amplifier rail is not signal. HI[k] is the last usable sample index for
# gauge k -- unbounded (N) for a simulated dump, or a channel that never
# clips -- and every search below stays clear of it, so a clip transition (the
# sharpest thing left in a record once real edges are gone) cannot be mistaken
# for one.
if EXPERIMENT:
    _CLIP_MARGIN = 0.05                          # ms, vs. a rail's ~1-sample edge
    HI = np.array([
        N if np.isnan(co) else max(1, int((co - _CLIP_MARGIN) / dt))
        for co in d['clip_onset']])
    for nm, co, hi in zip(names, d['clip_onset'], HI):
        if not np.isnan(co):
            print(f'{"":18}  {nm} clips from {co*1e3:.1f} us '
                  f'(t={t[hi]:.3f} ms kept)')
else:
    HI = np.full(len(names), N)

# The leading edge is ~59 us wide (10-90 %) on the SIMULATED rig; that width is
# tuned to the lumped-mass model, not to a real bar, so a MEASURED shot
# self-measures it instead -- from whichever gauge is sharpest, bounded clear
# of any clipping. The template spans a little more than that: long enough for
# a sharp correlation peak, short enough that two edges 0.1 ms apart still
# resolve into two peaks.
if EXPERIMENT:
    EDGE_MS = max(3.0 * min(_rise_time(g[:hi], dt)
                            for g, hi in zip(grads, HI)), 4 * dt)
else:
    EDGE_MS = 0.12
n_t = int(round(EDGE_MS / dt))

# --------------------------------------------------------------------------
# arrival of the direct pulse at each gauge, and the lag between gauges
# --------------------------------------------------------------------------
# One common template, taken from whichever gauge the wave reaches first, so
# every lag is measured against the same feature.
i_first = int(np.argmin([_rise_index(g, hi=HI[k]) for k, g in enumerate(grads)]))
ir = _rise_index(grads[i_first], hi=HI[i_first])
TEMPLATE = grads[i_first][ir - n_t // 4: ir + 3 * n_t // 4]

arrival = []
for k, g in enumerate(grads):
    c = _xcorr(g, TEMPLATE)
    i, _ = _extremum(c, 0, HI[k], +1)
    arrival.append(i * dt)
arrival = np.array(arrival)

REF = int(np.argmin(arrival))     # earliest arrival = furthest from the free end
lag = arrival - arrival[REF]      # >= 0, = (L_free_ref - L_free_k) / c0

print(f'\nreference gauge   : {names[REF]} (earliest arrival -> longest '
      f'baseline to the free end)')

# --------------------------------------------------------------------------
# the two negative edges: the striker's trailing edge, and the free-end echo
# --------------------------------------------------------------------------
def candidates(g, t_direct, hi=None, n_want=4):
    """Delays of the strongest negative edges after the direct arrival."""
    c = _xcorr(g, TEMPLATE)
    work = c.copy()
    work[:int((t_direct / dt) + 1.5 * n_t)] = 0.0
    if hi is not None:
        work[hi:] = 0.0     # a clipped correlation output, zeroed after the
                            # fact -- not fed back into anything
    out = []
    for _ in range(n_want):
        i, v = _extremum(work, 0, len(work), -1)
        out.append((i * dt - t_direct, v))
        # exclude only half a template either side, or a close second edge
        # (the trailing edge and the echo can be ~0.1 ms apart) is swallowed
        work[max(0, int(i) - n_t // 2): int(i) + n_t // 2] = 0.0
    return sorted(out)


cands = [candidates(g, a, hi=HI[k])
        for k, (g, a) in enumerate(zip(grads, arrival))]

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
    msg = 'no pulse length shared by at least two gauges; check the record'
    if EXPERIMENT:
        rows = '\n'.join(
            f'  {nm:>7}  clips from '
            + ('never' if np.isnan(co) else f'{co*1e3:.1f} us')
            for nm, co in zip(names, d['clip_onset']))
        msg += ('\n\nEvery search above was kept clear of each channel\'s own '
               'clipped tail:\n' + rows + '\n\nIf the clip onset sits earlier '
               'than the striker pulse length or the free-end\necho would be '
               'expected on this geometry, the record may simply not contain '
               'them in\nclean form at any gauge -- that is a property of '
               'this shot, not a bug here.')
    raise SystemExit(msg)

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
      f'{"2L_free/c0":>10}')
print(f'{"":>7} {"[" + USYM + "]":>10} {"[ms]":>9} {"[us]":>11} {"[ms]":>10}')
for k, nm in enumerate(names):
    tk = '  merged' if np.isnan(tau[k]) else f'{tau[k]:10.5f}'
    print(f'{nm:>7} {np.abs(signals[k]).max()*SCALE:{PKFMT}} {arrival[k]:9.4f} '
          f'{lag[k]*1e3:11.4f} {tk}')

# --------------------------------------------------------------------------
# c0 from the ONE measured length
# --------------------------------------------------------------------------
# L_free_k = L_free_ref - c0 * lag_k, and tau_k = 2 L_free_k / c0, so
#     Q_k = tau_k + 2 lag_k = 2 L_free_ref / c0
# is the same for every gauge. Gauges whose two negative edges merged drop out;
# the rest are averaged, and their spread is a genuine consistency check.
# Q must be the same at every gauge, so a gauge whose two negative edges merged
# into one -- and whose tau is therefore some later echo -- shows up as a gross
# outlier and is thrown out. With three or more gauges this needs no threshold
# tuning: the good ones agree to a fraction of a microsecond.
#
# Q is resolved BEFORE L_free_ref because referring a tape reading taken at some
# other gauge back to the reference gauge needs it.
Q = tau + 2.0 * lag
ok = ~np.isnan(Q)
ok &= np.abs(Q - np.median(Q[ok])) < 0.01 * np.median(Q[ok])
if ok.sum() < 1:
    msg = 'no gauge gave a usable free-end echo'
    if EXPERIMENT:
        rows = '\n'.join(
            f'  {nm:>7}  tau={tau[k]*1e3:8.1f} us  Q={Q[k]*1e3:9.1f} us  '
            + ('clips from ' + f'{d["clip_onset"][k]*1e3:.1f} us'
               if not np.isnan(d['clip_onset'][k]) else 'never clips')
            for k, nm in enumerate(names))
        msg += ('\n\nEach gauge found A negative edge after P and called it '
               'the free-end echo,\nbut Q = tau + 2*lag disagrees between '
               f'gauges (should be identical):\n{rows}\n\nEvery search stayed '
               'clear of each channel\'s own clipped tail; a spread this large '
               'means\nwhat was found there is not a shared echo, not that '
               'the search leaked into\nclipped data. If the geometry puts '
               'the true free-end echo later than every\nchannel\'s clip '
               'onset, it simply is not in this record in usable form.')
    raise SystemExit(msg)
Q_MEAN = float(np.mean(Q[ok]))

# L_FREE_REF is THE tape measurement -- the one length the experiment cannot
# supply itself. On the rig you measure it once, from a gauge to the far free
# end, to whatever precision you can manage, and put it in config.toml (or pass
# --l-free-ref). The model's own geometry is the FALLBACK: a simulation can
# supply it and a rig cannot, so relying on it makes this a self-check rather
# than an instrument. Nothing else below consults the true geometry except the
# error columns.
if EXPERIMENT:
    # load_experiment carries no absolute mesh coordinates (X_IN/X_OUT are a
    # simulator bookkeeping detail); the tape positions and the assembly
    # geometry give the same answer algebraically: L_free_ref_true is the
    # reference gauge's own distance to the free end, going the long way round
    # through the joint for an input-bar reference.
    L_FREE_REF_TRUE = (d['pos_in'][REF] + L_JOINT + L_OUTPUT if REF < n_in
                       else L_OUTPUT - d['pos_out'][REF - n_in])
else:
    X_TOTAL = d['X_OUT'] + d['L_free_out']
    _x_ref = (d['X_IN'] - d['pos_in'][REF]) if REF < n_in else \
             (d['X_OUT'] + d['pos_out'][REF - n_in])
    L_FREE_REF_TRUE = X_TOTAL - _x_ref

if L_FREE_REF_CFG is None:
    L_FREE_REF = L_FREE_REF_TRUE
elif L_FREE_REF_GAUGE in (None, names[REF]):
    L_FREE_REF = L_FREE_REF_CFG
elif L_FREE_REF_GAUGE in names:
    # The tape may have reached any gauge, not the one the record happens to
    # pick as reference. L_free_k = L_free_ref (1 - 2 lag_k / Q) inverts to
    # give L_free_ref from whichever gauge was actually measured -- at the cost
    # of dividing the tolerance by that same factor, so a short baseline is a
    # worse buy.
    _k = names.index(L_FREE_REF_GAUGE)
    _scale = 1.0 - 2.0 * lag[_k] / Q_MEAN
    if _scale <= 0:
        raise SystemExit(f'L_free_ref_gauge {L_FREE_REF_GAUGE!r} gives a '
                         'non-positive baseline; check the record')
    L_FREE_REF = L_FREE_REF_CFG / _scale
    L_FREE_REF_TOL = L_FREE_REF_TOL / _scale
else:
    raise SystemExit(
        f'L_free_ref_gauge {L_FREE_REF_GAUGE!r} is not one of {names}')

# The configured value depends on the gauge layout and the bar lengths, so it
# goes stale the moment either changes. In a simulation the true answer is right
# there; say so rather than quietly identifying the wrong bar.
if (L_FREE_REF_CFG is not None
        and abs(L_FREE_REF - L_FREE_REF_TRUE) > L_FREE_REF_TOL):
    print(f'\n!! WARNING: L_free_ref resolves to {L_FREE_REF:.1f} mm but the '
          f'model\n!! geometry says {L_FREE_REF_TRUE:.1f} mm, a slip of '
          f'{L_FREE_REF - L_FREE_REF_TRUE:+.1f} mm, outside the +-'
          f'{L_FREE_REF_TOL:.1f} mm tolerance. Has the gauge layout or a\n'
          f'!! bar length changed since L_free_ref was measured?')

# --------------------------------------------------------------------------
# c0 -- from the output bar alone (diagnostic table, always) -- never
# through the joint
# --------------------------------------------------------------------------
# out-0 and out-1 sit entirely on the output bar: their own free-end echoes
# never touch the joint, and neither does the direct-arrival lag on either
# bar's own two gauges -- the joint's delay, wherever it physically sits, is
# common to both legs of a direct-arrival lag and cancels in the difference.
# Four independent, coupler-free measurements below, none of them consuming
# L_free_ref at all; distance is whatever the wave actually travelled between
# t1 and t2 -- ONE-WAY for a direct-arrival lag, ROUND-TRIP (2x tape L_free)
# for an echo -- so c0 = distance / dt in every row, no hidden factor of 2.
# Printed regardless of USE_ONLY_OUT_FOR_C0, as a cross-check either way.
_OUT0, _OUT1 = n_in, n_in + 1
print('\n--- c0, from the output bar alone -- never crosses the coupler '
      '----------')
print(f'{"method":>34} {"distance":>10} {"t1":>9} {"t2":>9} '
      f'{"dt":>9} {"c0":>10}')
print(f'{"":>34} {"[mm]":>10} {"[ms]":>9} {"[ms]":>9} '
      f'{"[us]":>9} {"[mm/ms]":>10}')

c0_direct = {}
for i, (bar, off, cnt) in enumerate(
        (('in', 0, n_in), ('out', n_in, len(names) - n_in)), start=1):
    if cnt < 2:
        continue
    dt_d = abs(arrival[off + 1] - arrival[off])
    D_tape = abs(true_pos[off + 1] - true_pos[off])
    c0_direct[bar] = D_tape / dt_d
    print(f'{f"({i}) direct arrival, {bar}-0->{bar}-1":>34} {D_tape:10.2f} '
          f'{arrival[off]:9.4f} {arrival[off + 1]:9.4f} '
          f'{dt_d*1e3:9.4f} {c0_direct[bar]:10.3f}')

c0_echo_out = {}
if len(names) - n_in >= 2:
    for i, (k, nm) in enumerate(((_OUT0, 'out-0'), (_OUT1, 'out-1')), start=3):
        lbl = f'({i}) {nm} free-end echo'
        if np.isnan(tau[k]):
            print(f'{lbl:>34} {"-- edges merged, not usable --":>48}')
            continue
        L_free_tape_k = L_OUTPUT - true_pos[k]
        t_echo = arrival[k] + tau[k]
        c0v = 2.0 * L_free_tape_k / tau[k]
        c0_echo_out[nm] = c0v
        print(f'{lbl:>34} {2.0*L_free_tape_k:10.2f} {arrival[k]:9.4f} '
              f'{t_echo:9.4f} {tau[k]*1e3:9.4f} {c0v:10.3f}')

# (5) The two round trips DIFFERENCED. What is left is 2 D_out of travel, so
# the only tape number in it is the out-bar gauge spacing -- L_output cancels
# with the free end, and L_free_ref never enters. This is the one row that
# imports nothing but a length between two gauges 1 m apart, which is why
# c0_route = "out_echo_diff" is built on it.
c0_echo_diff = D_OUT_TAPE = None
if len(names) - n_in >= 2 and not (np.isnan(tau[_OUT0]) or np.isnan(tau[_OUT1])):
    D_OUT_TAPE = abs(true_pos[_OUT1] - true_pos[_OUT0])
    _dtau = abs(tau[_OUT0] - tau[_OUT1])
    c0_echo_diff = 2.0 * D_OUT_TAPE / _dtau
    print(f'{"(5) out-0 - out-1 echo difference":>34} {2.0*D_OUT_TAPE:10.2f} '
          f'{tau[_OUT1]:9.4f} {tau[_OUT0]:9.4f} {_dtau*1e3:9.4f} '
          f'{c0_echo_diff:10.3f}')
    print('  (5) differences two ROUND TRIPS, so its t1/t2 are those intervals '
          'and not\n      absolute times. Rows (2) and (5) measure the same '
          f'{D_OUT_TAPE:.0f} mm over different paths\n      and should agree; '
          f'they differ by {abs(c0_echo_diff - c0_direct["out"]):.1f} mm/ms '
          f'({abs(c0_echo_diff/c0_direct["out"] - 1):.1e}) here, which is the '
          'edge\n      changing shape with distance, not geometry.')

# --------------------------------------------------------------------------
# c0 -- FINAL, picked by [.c0_route]
# --------------------------------------------------------------------------
if C0_ROUTE == 'out_echo_diff':
    # Row (5): the two round trips differenced. One anchor, the out-bar gauge
    # spacing, and no other length -- which is the whole point of this route.
    if c0_echo_diff is None:
        raise SystemExit(
            'c0_route = "out_echo_diff" needs BOTH out-bar free-end echoes '
            '(row 5 of the table above); at least one is not usable here')
    c0_id = c0_echo_diff
    print(f'\nc0 (FINAL, c0_route="out_echo_diff") = 2 D_out / (tau_out0 - '
          f'tau_out1) = {c0_id:.3f} mm/ms')
    print(f'  anchored on                : the out-bar tape SPACING, '
          f'{D_OUT_TAPE:.1f} mm -- no L_output, no L_free_ref')
    _spread = abs(c0_echo_out['out-0'] - c0_echo_out['out-1'])
    print(f'  cross-check, rows (3)/(4)  : {c0_echo_out["out-0"]:.1f} and '
          f'{c0_echo_out["out-1"]:.1f} mm/ms, spread {_spread:.3f} '
          f'({_spread / c0_id:.2e} relative)')
    print(f'  tape contributes           : +-{np.sqrt(2)*GAUGE_TOL/D_OUT_TAPE:.1e}'
          f' relative (+-{c0_id*np.sqrt(2)*GAUGE_TOL/D_OUT_TAPE:.1f} mm/ms), '
          f'from +-{GAUGE_TOL:.1f} mm on each of the two gauge positions')
    if not EXPERIMENT:
        print(f'  c0 true                    : {d["c0_in"]:.3f} mm/ms   '
              f'rel err {(c0_id/d["c0_in"]-1):+.2e}')
elif C0_ROUTE == 'out_echo':
    # The output-bar-only route: average of ONLY the two out-bar echoes
    # (rows 3-4 above). Superseded, and kept only so older results stay
    # reproducible: each echo reads a tape gauge POSITION, so c0 lands on a
    # different anchor from the positions below, and the two do not cancel.
    # It also has the same lack of redundancy as row (5) -- only two numbers,
    # so nothing catches ONE of them being mis-detected (see
    # calibration_tension, which hits exactly that).
    if len(c0_echo_out) < 2:
        raise SystemExit(
            'c0_route = "out_echo" needs both out-0 and out-1 free-end '
            'echoes to set c0 -- see the table above for which '
            'one is not usable')
    c0_id = float(np.mean(list(c0_echo_out.values())))
    _out_spread = abs(c0_echo_out['out-0'] - c0_echo_out['out-1'])
    print(f'\nc0 (FINAL, c0_route="out_echo") = mean(out-0, out-1 echo) '
          f'= {c0_id:.3f} mm/ms')
    print('  !! c0 is anchored on the out-bar TAPE POSITIONS here while the '
          'positions below\n  !! are anchored on L_free_ref. Two anchors. '
          'Use c0_route = "out_echo_diff".')
    print(f'  out-0 vs out-1 echo spread : {_out_spread:.3f} mm/ms '
          f'({_out_spread / c0_id:.2e} relative)')
    if not EXPERIMENT:
        print(f'  c0 true                    : {d["c0_in"]:.3f} mm/ms   '
              f'rel err {(c0_id/d["c0_in"]-1):+.2e}')
else:
    # The classic route: Q = tau + 2 lag is the same at every gauge, so
    # L_free_ref anchors c0 = 2 L_free_ref / Q averaged over however many of
    # the up to 4 gauges pass the 1 % outlier check above (see "Q" -- this
    # crosses the threaded joint twice from whichever gauge is REF, but is
    # robust to any ONE gauge's echo being bad, unlike the output-bar-only
    # route above).
    print(f'\n--- c0, via L_free_ref through the joint '
          f'(c0_route="joint") ---------------')
    print(f'L_free_ref ({names[REF]} -> free end) : {L_FREE_REF:.1f} '
          f'+- {L_FREE_REF_TOL:.1f} mm')
    print(f'{"gauge":>7} {"Q = 2 L_free_ref/c0 [ms]":>24}')
    for k, nm in enumerate(names):
        note = f'{Q[k]:24.5f}' if ok[k] else (
            f'{"dropped: edges merged":>24}' if np.isnan(Q[k]) else
            f'{Q[k]:15.5f} rejected')
        print(f'{nm:>7} {note}')
    print(f'  mean {Q_MEAN:.5f} ms over {ok.sum()} gauges, '
          f'spread {np.ptp(Q[ok])*1e3:.3f} us ({np.ptp(Q[ok])/Q_MEAN:.1e})')
    c0_id = 2.0 * L_FREE_REF / Q_MEAN
    print(f'\nc0 (FINAL, c0_route="joint") = 2 L_free_ref / Q '
          f'= {c0_id:.3f} mm/ms')
    if not EXPERIMENT:
        print(f'  c0 true                    : {d["c0_in"]:.3f} mm/ms   '
              f'rel err {(c0_id/d["c0_in"]-1):+.2e}')
    print(f'  tape contributes           : +-{L_FREE_REF_TOL/L_FREE_REF:.1e} '
          f'(+-{c0_id*L_FREE_REF_TOL/L_FREE_REF:.2f} mm/ms)')

# --------------------------------------------------------------------------
# positions -- every gauge gets one, merged edges or not
# --------------------------------------------------------------------------
_IS_OUT = np.arange(len(names)) >= n_in
L_JOINT_EFF = L_OUT_IMPLIED = None

if C0_ROUTE == 'out_echo_diff':
    # Each gauge's own round trip IS its distance to the free end: tau_k =
    # 2 L_free_k / c0, nothing else in it. A gauge whose two negative edges
    # merged has no usable tau, so it falls back to the same quantity built
    # from the shared Q instead -- L_free_k = c0 (Q/2 - lag_k), which is the
    # identity Q = tau + 2 lag rearranged, not a second anchor.
    L_free = np.where(np.isnan(tau), c0_id * (Q_MEAN / 2.0 - lag),
                      c0_id * tau / 2.0)

    # The OUT bar reaches the free end without crossing anything, so its
    # positions need only the configured L_output -- and the two gauges then
    # give the same L_output back independently, which is a real check on it.
    L_OUT_IMPLIED = float(np.mean([L_free[k] + true_pos[k]
                                   for k in range(n_in, len(names))]))

    # The IN bar does not. Everything on it reaches the free end THROUGH the
    # joint, whose acoustic length is not its tape length, and the timings
    # cannot separate that delay from the in-bar gauges' own offsets: one
    # in-bar length has to be imported, for exactly the reason one length has
    # to be imported overall. The reference gauge's tape position is the one
    # to spend, being the longest in-bar baseline. What comes back is the
    # joint's EFFECTIVE length, reported rather than smeared.
    if n_in:
        _anchor = int(np.argmin(arrival[:n_in]))
        L_JOINT_EFF = float(L_free[_anchor] - true_pos[_anchor] - L_OUTPUT)
    L_JOINT_USED = L_JOINT if L_JOINT_EFF is None else L_JOINT_EFF
    id_pos = np.where(_IS_OUT, L_OUTPUT - L_free,
                      L_free - L_OUTPUT - L_JOINT_USED)
else:
    L_JOINT_USED = L_JOINT
    L_free = L_FREE_REF - c0_id * lag
    id_pos = np.where(_IS_OUT, L_OUTPUT - L_free, L_free - L_OUTPUT - L_JOINT)

if C0_ROUTE == 'out_echo_diff':
    print('\n--- what the out-bar anchor implies for the rest of the assembly '
          '--------')
    print('nothing below was used to identify anything -- these are the '
          'configured lengths\nmeasured against the one anchor, which is what '
          'makes them checks.\n')
    print(f'{"quantity":>24} {"identified":>12} {"config":>10} {"slip":>10}')
    print(f'{"L_output (out bar)":>24} {L_OUT_IMPLIED:12.2f} {L_OUTPUT:10.2f} '
          f'{L_OUT_IMPLIED - L_OUTPUT:+10.2f}')
    _lfr_id = float(L_free[REF])
    print(f'{f"L_free_ref ({names[REF]})":>24} {_lfr_id:12.2f} '
          f'{L_FREE_REF:10.2f} {_lfr_id - L_FREE_REF:+10.2f}')
    if L_JOINT_EFF is not None:
        print(f'{"joint, acoustic":>24} {L_JOINT_EFF:12.2f} {L_JOINT:10.2f} '
              f'{L_JOINT_EFF - L_JOINT:+10.2f}')
        print(f'\nthe joint costs {(L_JOINT_EFF - L_JOINT)/c0_id*1e3:+.1f} us '
              f'more than {L_JOINT:.0f} mm of bar stock would. It is anchored '
              f'on\n{names[_anchor]}\'s tape position ({true_pos[_anchor]:.1f} '
              'mm), the only in-bar length imported; the in-bar\nSPACING below '
              'is independent of that choice, the in-bar POSITIONS are not.')
        if abs(L_JOINT_EFF - L_JOINT) > 3.0 * GAUGE_TOL:
            print('!! That is not a tape error. Either the coupler is not '
                  'acoustically bar stock,\n!! or a bar length is wrong -- '
                  'and every number referred through it, L_free_ref\n!! '
                  'included, inherits the difference. See "What a mismatched '
                  'coupler costs" in\n!! README.md.')

# --------------------------------------------------------------------------
# position override -- an EMPIRICAL fallback, not a measurement
# --------------------------------------------------------------------------
# [.position_override] names gauges whose arrival-lag position is not trusted
# -- investigated and not explained by anything this script or
# identify_attenuation.py can measure (NOTES.md, thread 10 addendum: neither
# alpha(f), nor c_p(f), nor a direct near/far transfer-function fit reproduces
# the gap). Overriding trades a value this script computed for one a tape
# supplied; OVERRIDDEN records which BAR that touches, so the attenuation
# section below knows not to fit that bar's own (now geometry-inconsistent)
# pair -- id_pos would no longer agree with the arrival timing that pair's
# alpha/c_p fit assumes.
POS_OVERRIDE = dict(cfg.get('position_override', {}))
OVERRIDDEN = set()
if POS_OVERRIDE:
    print('\n--- position override, from config -- NOT identified '
          '---------------')
    print('tape used as the more defensible of the two, not because it has '
          'been shown\ncorrect -- see [.position_override] in config.toml '
          'for why.')
    for nm, val in POS_OVERRIDE.items():
        if nm not in names:
            raise SystemExit(f'position_override names {nm!r}, not one of '
                             f'{names}')
        k = names.index(nm)
        print(f'{nm:>7}  identified {id_pos[k]:8.2f} mm -> override '
              f'{float(val):8.2f} mm  ({id_pos[k]-float(val):+.2f} mm)')
        id_pos[k] = float(val)
        L_free[k] = (id_pos[k] + L_OUTPUT + L_JOINT_USED if k < n_in
                     else L_OUTPUT - id_pos[k])
        OVERRIDDEN.add(names[k].split('-')[0])

# What the tape costs each position -- whichever tape number this route
# actually imported.
if C0_ROUTE == 'out_echo_diff':
    # The scale rides on the out-bar spacing, so d(c0)/c0 = d(D_out)/D_out with
    # d(D_out) = sqrt(2) GAUGE_TOL, two independent readings differenced. An
    # out-bar position is L_output minus a length that scales with it, so it
    # takes the whole band; an in-bar position is anchored on the reference
    # gauge's own tape reading, so it takes that reading's error outright plus
    # the scale error on its DISTANCE from that anchor -- which is why the in
    # bar's near gauge is the better-placed one here, not the worse.
    _REL_TOL = np.sqrt(2.0) * GAUGE_TOL / D_OUT_TAPE
    L_free_band = np.where(
        _IS_OUT, _REL_TOL * L_free,
        _REL_TOL * np.abs(L_free - L_free[_anchor]) + GAUGE_TOL) \
        if n_in else _REL_TOL * L_free
else:
    # L_free_k = L_free_ref (1 - 2 lag_k / Q), so d(L_free_k) =
    # (L_free_k / L_free_ref) d(L_free_ref); and x is L_free plus a CONSTANT
    # (L_OUTPUT, L_JOINT) that the tape error does not touch, so the position
    # inherits that as an ABSOLUTE band rather than a relative one. It is much
    # the largest entry on this table, and the only one that is not the
    # timing's fault.
    L_free_band = L_FREE_REF_TOL * L_free / L_FREE_REF

print('\n--- gauge positions '
      '-------------------------------------------------------')
print(f'{"gauge":>7} {"L_free (to end)":>18} {"x (from face)":>15} '
      f'{"+-tape":>8} {_ref_lbl:>9} {"error":>9}')
_ANCHOR_K = _anchor if (C0_ROUTE == 'out_echo_diff' and n_in) else None
for k, nm in enumerate(names):
    note = '  <- anchor: this position IS the tape reading' \
        if k == _ANCHOR_K else ''
    print(f'{nm:>7} {L_free[k]:18.2f} {id_pos[k]:15.2f} {L_free_band[k]:8.2f} '
          f'{true_pos[k]:9.2f} {id_pos[k]-true_pos[k]:+9.3f}{note}')
if _ANCHOR_K is not None:
    print(f'the {names[_ANCHOR_K]} row is not a measurement: the in bar reaches '
          'the free end only\nthrough the joint, so one in-bar length is '
          'imported to fix that leg (see the\njoint line above). Every OTHER '
          'row here is a measurement.')

print('\n--- gauge spacing D '
      '-------------------------------------------------------')
print(f'{"bar":>7} {"lag [us]":>12} {"D = c0 dt":>12} {"D (used)":>10} '
      f'{_ref_lbl:>9} {"error":>9}')
for bar, off, cnt in (('in', 0, n_in), ('out', n_in, len(names) - n_in)):
    if cnt < 2:
        continue
    dl = abs(lag[off + 1] - lag[off])
    D_lag = c0_id * dl                      # from raw timing, always printed
    D_used = abs(id_pos[off + 1] - id_pos[off])   # what separate() actually gets
    D_true = abs(true_pos[off + 1] - true_pos[off])
    note = '  (overridden)' if bar in OVERRIDDEN else (
        '  (anchor -- +0 by construction)'
        if bar == 'out' and C0_ROUTE == 'out_echo_diff' else '')
    print(f'{bar:>7} {dl*1e3:12.4f} {D_lag:12.3f} {D_used:10.3f} {D_true:9.2f} '
          f'{D_used-D_true:+9.3f}{note}')
if C0_ROUTE == 'out_echo_diff':
    # The out row's two D columns come from the two timings that disagree by
    # 1.2 % (rows 2 and 5 above): the direct-arrival lag, and the echo
    # difference that set c0. Nothing is wrong with the geometry between them.
    print('D = c0 dt is the DIRECT-arrival lag; D (used) on the out bar is the '
          'imported\nspacing that set c0, from the ECHO difference. Their gap '
          'is rows (2) vs (5).')

# The errors in the "error" columns are what the TIMING costs, with the
# imported length taken as exact. That length's own tape error is separate and
# adds on top -- and WHICH quantities it reaches depends on which length was
# imported, so the summary below is written per route rather than once.
if C0_ROUTE == 'out_echo_diff':
    # Here the scale IS the out-bar spacing, so nothing escapes it: c0 scales
    # with it, and so does every L_free, and so therefore does every OTHER
    # spacing -- the out bar's own D is the one exception, being the imported
    # number itself. This is a worse leverage ratio than L_free_ref/D buys
    # (1077 mm of baseline instead of 3730), which is the price of the single
    # anchor. It is not a large price: the timing inconsistency it exposes was
    # 68 mm.
    _d_in = (_REL_TOL * abs(L_free[1] - L_free[0]) if n_in >= 2 else
             float('nan'))
    print(f'\nwith each tape gauge position good to +-{GAUGE_TOL:.1f} mm, so '
          f'the {D_OUT_TAPE:.0f} mm out-bar\nspacing to '
          f'+-{np.sqrt(2)*GAUGE_TOL:.1f} mm:\n'
          f'  c0             : +-{_REL_TOL:.1e} relative '
          f'(+-{c0_id*_REL_TOL:.2f} mm/ms)\n'
          f'  D (out)        : the imported number itself, +-'
          f'{np.sqrt(2)*GAUGE_TOL:.1f} mm\n'
          f'  D (in)         : +-{_d_in:.2f} mm -- it scales with c0, but '
          'carries no anchor error\n'
          f'  positions      : +-{L_free_band.min():.2f} to '
          f'+-{L_free_band.max():.2f} mm ABSOLUTE (the +-tape column above)\n'
          '  L_free_ref     : not consumed at all -- it is a CHECK under this '
          'route, above')
else:
    # D = L_free_1 - L_free_0 and L_FREE_REF cancels out of that difference
    # regardless of which c0 route was used, so D never carries it. c0 itself
    # only escapes it under "out_echo" -- otherwise c0 = 2 L_free_ref/Q is
    # directly proportional to L_free_ref. Only the ABSOLUTE positions always
    # carry it, through the L_FREE_REF additive constant in
    # L_free_k = L_FREE_REF - c0 * lag_k -- a tape error there shifts every
    # position by close to the same amount, benign for the same reason as
    # always: a common offset mostly moves where the wave is reconstructed
    # rather than distorting it.
    _c0_tape = ('unaffected -- does not depend on L_free_ref under '
               'c0_route="out_echo"' if C0_ROUTE == 'out_echo' else
               f'+-{L_FREE_REF_TOL/L_FREE_REF:.1e} relative '
               f'(+-{c0_id*L_FREE_REF_TOL/L_FREE_REF:.2f} mm/ms)')
    print(f'\nwith a tape (L_free_ref) good to +-{L_FREE_REF_TOL:.1f} mm on the '
          f'{L_FREE_REF:.0f} mm reference baseline:\n'
          f'  c0             : {_c0_tape}\n'
          f'  D              : unaffected -- L_free_ref cancels out of any '
          'D = L_free_i - L_free_j\n'
          f'  positions      : +-{L_free_band.min():.2f} to '
          f'+-{L_free_band.max():.2f} mm ABSOLUTE (the +-tape column above)')

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
          f'{_ref_lbl:>9}')
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
      'error as a small relative number: x is L_free plus a constant the tape\n'
      'does not scale, so a tape error moves them absolutely'
      + (' -- and on the in\nbar, under this route, x is measured from an '
         'anchor whose own tape error it\ntakes outright.'
         if C0_ROUTE == 'out_echo_diff' else '.')
      + ' What IS\nscale-free is L_free/c0 -- the free-end null below tests '
        'exactly that,\nand nothing else.\n')
print(f'{"gauge":>7} {"x/c0 [us]":>12} {f"{_ref_lbl} [us]":>12} '
      f'{"rel err":>10}')
for k, nm in enumerate(names):
    a = id_pos[k] / c0_id
    if EXPERIMENT:
        print(f'{nm:>7} {a*1e3:12.4f} {"—":>12} {"—":>10}')
    else:
        b = true_pos[k] / d['c0_in']
        print(f'{nm:>7} {a*1e3:12.4f} {b*1e3:12.4f} {a/b-1:+10.2e}')

# --------------------------------------------------------------------------
# attenuation and dispersion -- from the OUTPUT bar's two gauges, ONLY
# --------------------------------------------------------------------------
# Same method as identify_bar_compression.py's polycarbonate case: fit_
# attenuation's per-band transfer function gives alpha(f) from magnitude and,
# when handed c0, a phase-velocity table c_p(f)/c0 from phase -- both from
# the SAME two-gauge spectra, no boundary condition involved. A metal bar is
# expected to be lossless (alpha ~ 0), but Pochhammer-Chree dispersion is a
# property of the CYLINDER, not the material, and is not: with c_p = c0
# assumed at every frequency, the wavefront edge on the SHTB tension bar's
# out-0/out-1 pair (120 / 1200 mm) leaks a small transient into P and M right
# where the edge passes the far gauge. See NOTES.md, open thread 10.
#
# The `in` bar is never fit here, on this rig or any other: its own gauge
# pair runs opposite to propagation (the wave reaches the larger-x gauge
# first), which loses the phase fit outright -- see identify_attenuation's
# direction note -- and [.position_override], when it touches `in`, leaves
# its id_pos disagreeing with the arrival timing the fit assumes anyway.
# alpha(f) and c_p(f) are properties of the CYLINDER (Pochhammer-Chree), not
# of which half of it a gauge sits on -- this script already assumes ONE
# uniform bar end to end (the diameter check above refuses to run otherwise)
# -- so the output bar's fit is simply carried over rather than re-derived.
ATT = {}
if EXPERIMENT and 'attenuation' in cfg:
    from identify_attenuation import fit_attenuation
    ac = cfg['attenuation']
    print('\n--- attenuation and dispersion, from the output bar\'s two '
          'gauges alone ------')
    off, cnt = n_in, len(names) - n_in
    if cnt < 2:
        print('out not measurable: fewer than 2 gauges on the output bar')
    elif 'out' in OVERRIDDEN:
        print('out not fit: its position was overridden above, so alpha(f)/'
              'c_p(f)\n      would divide an arrival-lag delay by a '
              'tape-supplied distance')
    else:
        try:
            a = fit_attenuation(t, signals[off:off + cnt], id_pos[off:off + cnt],
                                arrival[off:off + cnt],
                                arrival[off:off + cnt] + tau[off:off + cnt],
                                f_lo=float(ac.get('f_lo', 2.0)),
                                f_hi=float(ac.get('f_hi', 50.0)),
                                snr=float(ac.get('snr', 0.005)),
                                c0=c0_id)
        except ValueError as exc:
            print(f'out not measurable: {exc}')
        else:
            ATT['out'] = a
            fb, ab = a['table']
            mid = len(fb) // 2
            print(f'{"out":>5} single-wave window {a["span"]*1e3:.0f} us, '
                  f'{a["pairs"]} gauge pair(s), band {a["f_lo"]:.0f}-'
                  f'{a["f_hi"]:.0f} kHz')
            print(f'{"":>5} alpha at {fb[mid]:.0f} / {a["f_hi"]:.0f} kHz: '
                  f'{ab[mid]:.2e} / {ab[-1]:.2e} /mm')
            if a['dispersion_table'] is None:
                # This bar's only pair runs opposite to propagation -- see
                # identify_attenuation's direction note. A single reversed
                # pair cannot be corrected for, only detected.
                print(f'{"":>5} c_p/c0: not measurable -- the only gauge '
                      'pair here runs opposite to propagation')
            else:
                fbd, cpr = a['dispersion_table']
                print(f'{"":>5} c_p/c0 at {fbd[mid]:.0f} / {a["f_hi"]:.0f} '
                      f'kHz: {cpr[mid]:.4f} / {cpr[-1]:.4f}  (scalar c_p = '
                      f'{a["c_p"]:.1f} mm/ms against c0 = {c0_id:.1f})')
                print(f'{"":>5} far gauge predicted from near, relative L2: '
                      f'lossless {a["misfit_lossless"]:.2e}, alpha only '
                      f'{a["misfit"]:.2e}, alpha+dispersion '
                      f'{a["misfit_dispersion"]:.2e}')
            ATT['in'] = a
            print(f'{"in":>5} alpha(f), c_p/c0: same as out above -- one '
                  'cylinder, not fit separately')

# --------------------------------------------------------------------------
# free-end null test -- the only check here that needs no ground truth
# --------------------------------------------------------------------------
# The far end of the output bar is a free surface, so the stress there is zero
# at all times and the two travelling waves must cancel:
#
#     eps_plus + eps_minus = 0     at L_free = 0
#
# Reconstruct AT that surface -- hand `separate` the identified L_free, which
# are distances from it -- and the boundary condition becomes a residual that
# should vanish. Nothing here consults the true geometry, which makes this the
# one validation that survives contact with a real rig, where there is no truth
# to compare c0 or the positions against.
#
# ONLY the output bar's own two gauges go into this, same restriction as c0
# and alpha(f)/c_p(f) above and for the same reason: reaching the free
# surface from an in-bar gauge means crossing the threaded joint, whose
# impedance is not the bar's. It is also the only physically sound choice --
# the assembly has a reflector at each end, but only the output bar's far end
# is a genuine free surface; the anvil end behind the in bar is a lumped mass,
# not a clean reflector (see the module docstring), so there is no equivalent
# null test to run on that side at all.
#
# It also responds to a mismatched coupler, which the Q spread provably cannot:
# the coupler's extra transit time enters Q identically at every gauge and
# cancels out of the spread, but it does not cancel here. Do not oversell that:
# a 150 mm coupler at 0.9 rho moves this residual only from 1.2e-3 to 3.5e-3,
# because the null constrains L_free/c0 on baselines of METRES, where the
# coupler's bias is relatively small. The damage lands on x/c0 instead, over
# baselines of ~130 mm, where the same absolute error is 20x larger in relative
# terms -- which is why the reduction degrades more than this number suggests.
# Treat a FAIL as conclusive and a PASS as weak evidence.
#
# What it CANNOT do is break the scale degeneracy. Scaling L_free and c0
# together leaves the residual identical to seven digits (verified), because the
# test constrains the transit times L_free/c0 and nothing else -- which is
# exactly what separate() consumes, and exactly what L_free_ref cannot fix.
#
# Note the third argument: `separate` calls its positions x and phases them with
# the wavenumber xi. Here they are L_free, distances from the FREE END rather
# than from a bar face, which is what moves the reconstruction to that surface.
if EXPERIMENT:
    NULL_WINDOW = cfg.get('null', {}).get('window', 0.75)
    NULL_TOL = cfg.get('null', {}).get('tol', 5.0e-3)
else:
    NULL_WINDOW = cfg.get('null_window', 0.75)
    NULL_TOL = cfg.get('null_tol', 5.0e-3)

# `separate` is a GLOBAL fit -- one FFT of the whole record per gauge -- so a
# clipped tail anywhere in the input corrupts P(w)/M(w) everywhere, not just at
# the times it occupies. Windowing only the RESIDUAL check afterward would not
# fix that; the record fed to `separate` itself must end before the earliest
# clip onset across every gauge that contributes to it. (The edge-timing
# searches above do not have this problem -- `_xcorr` is a genuine local
# correlation, not a single whole-record transform.)
_hi_null = int(np.min(HI)) if EXPERIMENT else N
t_null = t[:_hi_null]
sig_null = [s[:_hi_null] for s in signals]

_NULL_OFF, _NULL_CNT = n_in, len(names) - n_in

print('\n--- free-end null test (no ground truth used) '
      '-----------------------------')
if _NULL_CNT < 2:
    null_rms = null_max = float('nan')
    print('cannot be evaluated: the output bar has fewer than 2 gauges -- '
          'this test only\never uses those (see the section comment above), '
          'and there is no substitute.')
else:
    sig_out_null = sig_null[_NULL_OFF:_NULL_OFF + _NULL_CNT]
    L_free_out = L_free[_NULL_OFF:_NULL_OFF + _NULL_CNT]
    p_free, m_free = separate(t_null, sig_out_null, L_free_out, c0=c0_id,
                              eta=d['eta'])
    _total = p_free + m_free
    _amp = np.abs(p_free).max()

    # The tail MUST be cut. The exponential window that regularises separate()
    # amplifies the truncation at the end of the record, and over the FULL
    # record the residual comes out ~100x larger than it really is -- 1.2e-01
    # against 1.2e-03 on a calibration that is in fact good. Start at the
    # first arrival at the free end; stop before the truncation (and, for a
    # measured shot, before whichever comes first: that truncation or the
    # clipped tail).
    _i0 = int(np.argmax(np.abs(p_free) > 0.02 * _amp))
    _i1 = min(int(NULL_WINDOW * N), len(t_null))
    _w = slice(_i0, _i1)

    if _i1 <= _i0:
        null_rms = null_max = float('nan')
        print('cannot be evaluated: once the clipped tail is excluded, the '
              'window clear of\nboth the record start and the truncation '
              'collapses to nothing at both output\ngauges at once. This '
              'record does not reach the free surface in clean form.')
    else:
        null_rms = float(np.sqrt(np.mean(_total[_w] ** 2)) / _amp)
        null_max = float(np.abs(_total[_w]).max() / _amp)
        print(f'reconstructed at the free surface from the output bar\'s '
              f'{_NULL_CNT} gauges alone, {t_null[_i0]:.2f}-'
              f'{t_null[_i1-1]:.2f} ms'
              + (f' (record truncated to {t_null[-1]:.2f} ms, clear of the '
                 'clipped tail)' if EXPERIMENT and _hi_null < N else ''))
        print(f'peak |eps+|            : {_amp*SCALE:.1f} {USYM}')
        print(f'residual |eps+ + eps-| : rms {null_rms:.2e}, max '
              f'{null_max:.2e} (relative to peak |eps+|)')
        print(f'threshold              : {NULL_TOL:.1e}   ->  '
              f'{"PASS" if null_rms <= NULL_TOL else "FAIL"}')
        if null_rms > NULL_TOL:
            print('  The free surface does not come out stress-free, so the '
                  'transit\n  times L_free/c0 are wrong. Most likely: a '
                  'coupler that is not\n  bar material, or the wrong coupler '
                  'length. Note L_free_ref is\n  NOT the suspect -- this '
                  'test is blind to it.')

# --------------------------------------------------------------------------
# density: NOT identifiable from the records; closed with the bar's mass
# --------------------------------------------------------------------------
# One uniform assembly, checked above, so either bar's density will do.
print('\n--- density and modulus '
      '---------------------------------------------------')
_rho = d.get('rho_in')
if _rho is None:
    print('NOT computed: no bar density supplied for this record (rho is '
          'optional in\n[.input_bar]/[.output_bar] and was omitted here). '
          'The reconstruction never\nasks for it anyway -- E, A and rho enter '
          'nowhere; only c0, the positions and\neta do.')
else:
    m_bar = _rho * AREA * L_ASSEMBLY
    E_id = (m_bar / (AREA * L_ASSEMBLY)) * c0_id ** 2
    print('NOT identifiable from strain records: they fix c0 = sqrt(E/rho) and '
          'no more.\nClosed here with one extra measurement, the bar mass. In '
          'the lab that is an\nindependent weighing; HERE it is computed back '
          f"from {'the simulator' if not EXPERIMENT else 'an assumed handbook'}"
          "'s own rho,\nso the rho line is circular. The E line is not: it "
          'uses the IDENTIFIED c0.\n')
    print(f'bar mass (weighed)      : {m_bar*1e3:.1f} g')
    print(f'rho = m / (A L)         : {m_bar/(AREA*L_ASSEMBLY):.4e} kg/mm^3   '
          f'({"assumed handbook value" if EXPERIMENT else "circular"})')
    if EXPERIMENT:
        print(f'E   = rho c0^2          : {E_id:.3f} GPa   (closure, not a '
              'measurement -- rho above is assumed)')
        print(f'E*A (the force scale)   : {E_id*AREA:.1f} kN')
    else:
        print(f'E   = rho c0^2          : {E_id:.3f} GPa   '
              f'(true {d["E_in"]:.3f}, rel err {E_id/d["E_in"]-1:+.1e})')
        print(f'E*A (the force scale)   : {E_id*AREA:.1f} kN   '
              f'(true {d["E_in"]*d["A_in"]:.1f})')

print('\n--- ready to use '
      '----------------------------------------------------------')
print(f'  c0 = {c0_id:.3f}')
print(f'  gauges = [{", ".join(f"{p:.2f}" for p in id_pos[:n_in])}]'
      '    # input bar, mm from its face')
print(f'  gauges = [{", ".join(f"{p:.2f}" for p in id_pos[n_in:])}]'
      '    # output bar')

# --------------------------------------------------------------------------
# hand the numbers on
# --------------------------------------------------------------------------
# The same split the simulators use: one script produces, another consumes,
# and a file in between so that iterating on a reconstruction does not mean
# re-running the identification. c_in/c_out and R_in/R_out duplicate the same
# assembly-wide value per bar -- one c0, one Q, one identification -- matching
# dump.npz's own convention for this rig's symmetric fields, which is what
# lets a bar-indexed consumer read this without special-casing a shared-
# assembly identification.
IDENT_FILE = 'bar_identified.npz'
BARS = tuple(b for b, cnt in (('in', n_in), ('out', len(names) - n_in))
            if cnt >= 1)
_out = dict(case=CASE, bars=np.array(BARS), c0_route=C0_ROUTE)
# Assembly-wide, not per bar: what the one anchor implied for the lengths it
# did not consume. Absent under the routes that never derive them.
if L_OUT_IMPLIED is not None:
    _out['L_output_implied'] = L_OUT_IMPLIED
if L_JOINT_EFF is not None:
    _out['L_joint_eff'] = L_JOINT_EFF
for b, off, cnt in (('in', 0, n_in), ('out', n_in, len(names) - n_in)):
    if cnt < 1:
        continue
    _out[f'c_{b}'] = c0_id
    _out[f'R_{b}'] = Q_MEAN
    _out[f'L_ref_{b}'] = L_FREE_REF
    _out[f'x_{b}'] = id_pos[off:off + cnt]
    _out[f'L_free_{b}'] = L_free[off:off + cnt]
    if EXPERIMENT:
        _out[f'tape_{b}'] = np.asarray(true_pos[off:off + cnt], float)
    if b in ATT:
        _out[f'alpha_f_{b}'], _out[f'alpha_{b}'] = ATT[b]['table']
        if ATT[b]['dispersion_table'] is not None:
            _out[f'dispersion_f_{b}'], _out[f'dispersion_{b}'] = \
                ATT[b]['dispersion_table']
np.savez(IDENT_FILE, **_out)
print(f'\nwrote {IDENT_FILE}: c0={c0_id:.1f} for {", ".join(BARS)}')

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'

# One column per bar -- in, out -- three rows each: what was measured, its
# derivative (the edges everything above is actually timed on), and F = P+M
# reconstructed from THAT bar's own two gauges alone, at ITS OWN interface.
# A fourth row carries a different check in each column. RIGHT: the free-end
# null test's own reconstruction (stress at the free surface, which should sit
# at zero) -- only the OUTPUT bar has one, since that test only ever uses the
# output bar's gauges (see its section comment above); the in bar's far end is
# the anvil, not a free surface. LEFT: the two bars' interface forces
# overlaid, which is the check the in bar CAN carry -- force is continuous
# across the coupler, so those two independent solves must agree.
# All rows share `t_null` (see above: the record truncated
# before the earliest clip onset, for a measured shot) so a clipped tail
# cannot leak into the reconstruction the same way it cannot leak into the
# free-end null.
COLS = [(b, off, cnt) for b, off, cnt in
        (('in', 0, n_in), ('out', n_in, len(names) - n_in)) if cnt >= 2]
fig, axes = plt.subplots(4, len(COLS), figsize=(9.5 * len(COLS), 15.5),
                         sharex=True, squeeze=False)
fig.patch.set_facecolor(SURFACE)
_blank_axes = set()
F_BARS = {}          # each bar's F = P + M at its own coupler face, for the
                     # equilibrium panel that fills the in column's last row
_ax_equil = None     # that panel's axes, once the in column has been drawn

for col, (bar, off, cnt) in enumerate(COLS):
    idx = slice(off, off + cnt)
    bsig = sig_null[idx]
    bgrad = [g[:_hi_null] for g in grads[idx]]
    bx = id_pos[idx]

    ax0 = axes[0, col]
    for j in range(cnt):
        src = 'overridden' if names[off + j] in POS_OVERRIDE else 'identified'
        ax0.plot(t_null, bsig[j] * SCALE, lw=.9, color=(BLUE, ORANGE)[j % 2],
                 label=f'{names[off + j]} at {bx[j]:.0f} mm ({src})')
    ax0.set_ylabel(f'Signal ({USYM})')
    ax0.set_title(f'What was measured — {cnt} gauges on the {bar} bar',
                  loc='left', fontsize=11)

    # Matched-filtered, not raw: the differentiated record is edge-timed by
    # cross-correlating it against TEMPLATE (the leading edge cut from the
    # reference gauge, above), and that correlation is what suppresses the
    # sample-to-sample noise a raw gradient is full of. Same filter each
    # arrival/echo search above already runs, just shown rather than only
    # used, and normalised per gauge to its own peak like the original
    # single-gauge version of this panel was.
    ax1 = axes[1, col]
    for j in range(cnt):
        xc = _xcorr(bgrad[j], TEMPLATE)
        xc = xc / np.abs(xc).max()
        ax1.plot(t_null, xc, lw=.9, color=(BLUE, ORANGE)[j % 2],
                 label=f'{names[off + j]}')
    ax1.axhline(0, color=GRID, lw=.8)
    ax1.set_ylabel('Edge filter (norm.)')
    ax1.set_title(f'Matched-filtered edge record at both {bar}-bar gauges — '
                  'what the edge timing above is actually measured on',
                  loc='left', fontsize=10)

    # Vertical markers, on BOTH the raw and differentiated rows: the end of
    # the striker pulse (arrival + P) and the free-end echo (arrival + tau),
    # per gauge -- not per bar, since the two gauges on a bar see both edges
    # at different times. Colour matches the gauge; linestyle matches the
    # edge, so one small proxy legend below explains both rows at once.
    for j in range(cnt):
        k = off + j
        gcol = (BLUE, ORANGE)[j % 2]
        for ax in (ax0, ax1):
            if not np.isnan(P):
                ax.axvline(arrival[k] + P, color=gcol, lw=1.1, ls='--',
                          alpha=.7)
            if not np.isnan(tau[k]):
                ax.axvline(arrival[k] + tau[k], color=gcol, lw=1.1, ls=':',
                          alpha=.7)
    _marks = [plt.Line2D([], [], color=MUTED, lw=1.1, ls='--',
                         label='striker pulse ends'),
              plt.Line2D([], [], color=MUTED, lw=1.1, ls=':',
                         label='free-end echo')]
    for ax, loc in ((ax0, 'lower left'), (ax1, 'upper left')):
        h, l = ax.get_legend_handles_labels()
        ax.legend(h + _marks, l + [m.get_label() for m in _marks],
                 frameon=False, fontsize=9, labelcolor=MUTED, loc=loc)

    # F = P + M at x = 0, from this bar's own two gauges only -- exactly
    # `reconstruct_interface.py`'s per-bar panel, reusing whatever alpha(f)/
    # c_p(f) this run just identified for this bar (ATT.get(bar) is None
    # whenever [.attenuation] is not configured, or the fit was not
    # measurable -- see the direction note in identify_attenuation.py).
    fit = ATT.get(bar)
    gatt = fit['table'] if fit is not None else None
    gdisp = fit['dispersion_table'] if fit is not None else None
    p_b, m_b = separate(t_null, bsig, bx, c0=c0_id, eta=d['eta'],
                        attenuation=gatt, dispersion=gdisp)
    F_b = p_b + m_b
    F_BARS[bar] = F_b

    ax2 = axes[2, col]
    pos_lbl = 'overridden position' if bar in OVERRIDDEN else 'identified positions'
    ax2.plot(t_null, F_b * SCALE, color=INK, lw=.9,
             label=f'$F = P + M$, {pos_lbl}')
    ax2.axhline(0, color=GRID, lw=1.0)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel(f'Interface force ({USYM})')
    ax2.set_title(f'F = P + M at the {bar}put-bar/specimen interface'
                  + ('' if gatt is not None else '  (LOSSLESS)'),
                  loc='left', fontsize=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

    # Fourth row: the free-end null test's own reconstruction -- stress AT
    # the free surface, which the boundary condition demands sit at zero.
    # Only the output bar has one (see the section comment above); the input
    # bar's column gets a blank row instead, styled off in the loop below.
    ax3 = axes[3, col]
    if bar == 'out':
        ax3.plot(t_null, p_free * SCALE, lw=.9, color=BLUE, label='$P$')
        ax3.plot(t_null, m_free * SCALE, lw=.9, color=ORANGE, label='$M$')
        ax3.plot(t_null, _total * SCALE, color=INK, lw=1.1,
                 label='$P + M$ (should be 0)')
        ax3.axhline(0, color=GRID, lw=1.0)
        if _i1 > _i0:
            ax3.axvspan(t_null[_i0], t_null[_i1 - 1], color=GRID, alpha=.35,
                       label='rms/max window')
        verdict = ('not evaluable' if np.isnan(null_rms) else
                  ('PASS' if null_rms <= NULL_TOL else 'FAIL'))
        _rms_txt = '--' if np.isnan(null_rms) else f'{null_rms:.2e}'
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel(f'Stress at free end ({USYM})')
        ax3.set_title(f'Free-end null: stress at the output bar\'s free '
                      f'surface -- rms {_rms_txt} vs tol {NULL_TOL:.1e} -> '
                      f'{verdict}', loc='left', fontsize=10)
        ax3.legend(frameon=False, fontsize=9, labelcolor=MUTED,
                  loc='lower left')
    else:
        _ax_equil = ax3          # filled in below, once BOTH bars are solved

# --------------------------------------------------------------------------
# the in column's last row: force equilibrium across the coupler
# --------------------------------------------------------------------------
# Row 2 already shows each bar's F = P + M at its own coupler face, one per
# column, which is the right place to judge each reconstruction on its own.
# Overlaying them is a different question: force is continuous across a rigid
# coupler, so the two curves are two INDEPENDENT measurements of one quantity
# -- separate solves, separate gauges, sharing only c0 -- and where they part
# company is the honest error bar on the whole identification. It is exactly
# bar_equilibrium.py's number, computed here from this run's own numbers
# (including the identified alpha(f)/c_p(f), which bar_equilibrium.py does not
# apply) so the figure needs no second script to be read.
#
# The two curves are NOT at the same place: they are one coupler apart, and
# nothing here shifts either of them. On this rig that gap is the identified
# L_joint_eff, ~18 us -- see the joint line in the table above, and NOTES.md
# thread 11 for why a plain time shift does not reconcile the two.
if 'in' in F_BARS and 'out' in F_BARS:
    F_in_b, F_out_b = F_BARS['in'], F_BARS['out']
    _pk = float(np.abs(F_in_b).max())
    _eq = np.abs(F_in_b - F_out_b) / (_pk if _pk > 0 else 1.0)
    # Same window convention as the free-end null and bar_equilibrium.py:
    # clear of the quiescent start, and of the tail the eta-window amplifies.
    _amp_e = max(_pk, float(np.abs(F_out_b).max()))
    _e0 = int(np.argmax((np.abs(F_in_b) + np.abs(F_out_b)) > 0.02 * _amp_e))
    _e1 = min(int(NULL_WINDOW * N), len(t_null))
    _ew = slice(_e0, _e1)
    _eq_mean = float(_eq[_ew].mean()) if _e1 > _e0 else float('nan')
    _eq_max = float(_eq[_ew].max()) if _e1 > _e0 else float('nan')

    _ax_equil.plot(t_null, F_in_b * SCALE, lw=1.0, color=BLUE,
                   label='$F_{in}$ at the input-bar / coupler face')
    _ax_equil.plot(t_null, F_out_b * SCALE, lw=1.0, color=ORANGE,
                   label='$F_{out}$ at the output-bar / coupler face')
    _ax_equil.plot(t_null, (F_in_b - F_out_b) * SCALE, color=INK, lw=1.1,
                   label='$F_{in} - F_{out}$ (should be 0)')
    _ax_equil.axhline(0, color=GRID, lw=1.0)
    if _e1 > _e0:
        _ax_equil.axvspan(t_null[_e0], t_null[_e1 - 1], color=GRID, alpha=.35,
                          label='mean/max window')
    _ax_equil.set_xlabel('Time (ms)')
    _ax_equil.set_ylabel(f'Interface force ({USYM})')
    _ax_equil.set_title(
        'Force equilibrium across the coupler — two independent solves of one '
        f'force:\nmean |$F_{{in}}-F_{{out}}$|/max|$F_{{in}}$| = {_eq_mean:.2e}, '
        f'max {_eq_max:.2e}  (the two faces are one coupler apart, unshifted)',
        loc='left', fontsize=10)
    _ax_equil.legend(frameon=False, fontsize=9, labelcolor=MUTED,
                     loc='lower left')
    print(f'\nforce equilibrium across the coupler (figure, bottom left) : '
          f'mean {_eq_mean:.4e}, max {_eq_max:.4e}')
elif _ax_equil is not None:
    # An in column exists but the out bar carries fewer than 2 gauges, so
    # there is no second force to compare it against. Blank, as before.
    _blank_axes.add(_ax_equil)

axes[0, 0].set_xlim(0, t_null[-1])

for ax in axes.flat:
    if ax in _blank_axes:
        ax.axis('off')
        continue
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.suptitle('Per-bar identification check — each column reconstructed from '
             'ONLY that bar\'s own two gauges; the bottom-left panel is the '
             'one place the two meet', x=.006, ha='left', fontsize=13,
             color=INK)
fig.tight_layout(rect=(0, 0, 1, .97))
FIG = f'bar_identification_{CASE}.png' if EXPERIMENT \
    else 'bar_identification_tension.png'
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'\nwrote {FIG}')

plotting.show_unless(HEADLESS)
