"""
Identify gauge positions, gauge spacing and wave speed for a DIRECT-IMPACT
compression bar, from a calibration shot with no specimen -- the two bar faces
struck straight against each other.

    python3 drive_calibration_compression.py
    python3 identify_bar_compression.py [--headless]
    python3 identify_bar_compression.py --l-in-ref 2000.0 --l-out-ref 1000.0

Companion to identify_bar_tension.py, which does the same for the SHTB. The
method is the same in spirit -- time EDGES on the differentiated record, never
whole pulses -- but the compression rig is a genuinely easier problem, and the
two differences below are why it gets its own script rather than a flag.

--------------------------------------------------------------------------
Both bars are free at the far end, so the round trip is in the record
--------------------------------------------------------------------------
The SHTB has exactly one clean reflector: the far free end of the output bar.
Its other end carries the anvil, which is a lumped mass rather than a
termination and reflects like a free end displaced ~257 mm outward. So that rig
can only measure gauge-to-free-end distances, and the ONE length it needs from
outside is a gauge-to-free-end distance too -- awkward, because a gauge is a
grid under a blob of adhesive somewhere along a 3 m bar.

Here BOTH far ends are free, and the bars separate the moment the contact goes
into tension. Each bar then rings on its own round trip, and a gauge x from the
contact face on a bar of length L sees, in the DERIVATIVE of its record:

    delay 0               - edge   the wave arriving from the contact
    delay 2(L - x)/c      + edge   the free-end echo, INVERTED
    delay 2L/c            - edge   that echo re-reflected at the contact end

The third one is the prize. 2L/c is the SAME at every gauge on the bar, so it
is both the measurement and its own consistency check, and the length that
closes the scale is THE BAR'S OWN LENGTH. You measure that once, on the bench,
before anything is glued to it.

--------------------------------------------------------------------------
Two bars, two scales -- and one welcome consequence
--------------------------------------------------------------------------
A strain record is invariant under (lengths, c) -> (lambda lengths, lambda c),
and here that degeneracy applies to EACH BAR SEPARATELY: once they part company
the two are acoustically independent, and no timing on one says anything about
the scale of the other. So this script wants TWO measured lengths, one per bar,
against the SHTB's one. On a rig where the bars are the same stock that feels
like a step backwards; on this one, where an aluminium input bar drives a
polycarbonate output bar 3.7x slower, the two speeds were never one number.

The welcome part: because the round trip is measured, a position comes out as

    x_k = L_ref * (f3_k - f2_k) / R,          R = 2L/c, measured

which is PROPORTIONAL to the supplied length. A tape error therefore moves the
positions by a small RELATIVE amount, not the absolute band that identify_bar_
tension.py has to warn about -- there x = L_free + a constant the tape never
touches, and +-2 mm of tape lands as +-2 mm on x. Here +-2 mm on a 2000 mm bar
is 1e-3, so +-0.13 mm on a 130 mm position.

Everything else carries over unchanged from the tension script, and is argued
there rather than repeated here:

  * DENSITY IS NOT IDENTIFIABLE. The shot fixes c = sqrt(E/rho) and never E and
    rho separately; closing it needs one absolute mass or force measurement.
  * What `separate` consumes is the TRANSIT TIMES x_k/c, not the lengths.
  * D, the gauge SPACING, is what the reduction leans on; the individual x
    matters much less.
  * The free-end null test is the only check here that consults no ground
    truth, and so the only one that survives contact with a real rig. This rig
    gets it on BOTH bars, because both far ends are free.
"""
import argparse

import numpy as np

import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--l-in-ref', type=float, metavar='MM', dest='L_in_ref',
                 help='input bar length, struck face -> free end [mm]. '
                      'Overrides L_free_in_ref in config.toml.')
_ap.add_argument('--l-out-ref', type=float, metavar='MM', dest='L_out_ref',
                 help='output bar length, struck face -> free end [mm]. '
                      'Overrides L_free_out_ref in config.toml.')
_ap.add_argument('--l-free-ref-tol', type=float, metavar='MM',
                 dest='L_free_ref_tol',
                 help='what the tape is good to [mm], applied to both bars.')
_ap.add_argument('--experiment', metavar='CASE', default=None,
                 help='identify a MEASURED shot named by a config case '
                      '(e.g. experiment_pc_bar) instead of the simulated '
                      'calibration dump. There is no ground truth then, so the '
                      'true/error columns print a dash.')
HEADLESS, ARGS = plotting.init(parser=_ap)

import config
from dump import load_dump
from wave_separation import separate

CASE = ARGS.experiment or 'calibration_compression'
EXPERIMENT = CASE in config.EXPERIMENT_CASES
cfg = config.load(CASE)

# The two measured lengths -- one per bar, because the two bars are
# acoustically independent once they separate. config.toml is their durable
# home; the flags exist so their influence can be swept without editing it.
# Absent from both, the script falls back to the model's own geometry, which a
# simulation can supply and a rig cannot: that is the self-check mode.
L_REF_CFG = {'in': ARGS.L_in_ref if ARGS.L_in_ref is not None
                   else cfg.get('L_free_in_ref'),
             'out': ARGS.L_out_ref if ARGS.L_out_ref is not None
                    else cfg.get('L_free_out_ref')}
L_REF_TOL = (ARGS.L_free_ref_tol if ARGS.L_free_ref_tol is not None
             else cfg.get('L_free_ref_tol', 2.0))


# --------------------------------------------------------------------------
# signal processing -- same three primitives as identify_bar_tension.py
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
    den = y[i - 1] - 2.0 * y[i] + y[i + 1]
    return 0.0 if den == 0 else 0.5 * (y[i - 1] - y[i + 1]) / den


def _peak(c, sign, lo=0, hi=None):
    """Sub-sample index of the strongest peak of the given sign in [lo, hi)."""
    hi = len(c) if hi is None else hi
    lo, hi = max(0, int(lo)), min(len(c), int(hi))
    if hi <= lo:
        return np.nan
    i = lo + int(np.argmax(c[lo:hi] * sign))
    return i + _refine(c * sign, i)


def _rise_time(g, dt):
    """
    10-90 % rise time of the leading edge, in ms.

    identify_bar_tension.py hardcodes its template width, which NOTES.md flags
    as the constant most likely to bite on a real bar: it is tuned to that
    model's 59 us edge, and Pochhammer-Chree dispersion widens edges. Here it
    cannot be hardcoded at all -- the polycarbonate bar's edges are several
    times broader than the aluminium's -- so it is measured, per bar, from the
    record itself.
    """
    a = np.abs(g)
    pk = a.max()
    i_pk = int(np.argmax(a))
    i_lo = int(np.argmax(a[:i_pk + 1] > 0.1 * pk)) if i_pk else 0
    return max((i_pk - i_lo), 1) * dt


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
if EXPERIMENT:
    from experiment import load_experiment
    d = load_experiment(CASE)
else:
    d = load_dump()
t, dt, N = d['t'], d['dt'], d['N']

if d['loading'] != 'compression':
    raise SystemExit(
        f"this dump is a {d['loading']} shot; identify_bar_compression.py reads "
        "a direct-impact\nrecord. Use identify_bar_tension.py for the SHTB.")
if d['L_specimen'] != 0.0:
    raise SystemExit(
        f"this dump has a {d['L_specimen']:.0f} mm specimen between the bars; "
        "the identification\nassumes they are struck face to face. Run "
        "drive_calibration_compression.py.")

# WHICH BARS ARE IDENTIFIABLE is a property of the record, not of the rig. A bar
# needs two gauges before `separate` has two equations for two unknowns, and a
# real rig routinely instruments one bar and not the other -- the PC shot has no
# gauge on its aluminium input bar at all. Asking the data rather than assuming
# ('in', 'out') is what lets one script serve both.
ALL_BARS = ('in', 'out')
BARS = tuple(b for b in ALL_BARS if d[f'eps_{b}'].shape[0] >= 2)
SKIPPED = {b: int(d[f'eps_{b}'].shape[0]) for b in ALL_BARS if b not in BARS}
if not BARS:
    raise SystemExit('no bar in this record carries two gauges; nothing to '
                     'identify. Separation needs two equations per bar.')

NAMES = {b: [f'{b}-{k}' for k in range(d[f'eps_{b}'].shape[0])] for b in BARS}
SIG = {b: [d[f'eps_{b}'][k] for k in range(d[f'eps_{b}'].shape[0])] for b in BARS}
GRAD = {b: [np.gradient(s, dt) for s in SIG[b]] for b in BARS}

# Ground truth is a property of a SIMULATED record and of nothing else. On a
# measured shot these stay empty and every `true` / `error` column prints a
# dash, rather than being filled with the tape values -- which would quietly
# turn an input into a validation. UNITS likewise: a dump carries strain, a
# scope carries whatever the conditioner was calibrated in.
TRUE_POS = {b: list(d[f'pos_{b}']) for b in BARS} if not EXPERIMENT else {}
TRUE_C = {'in': d['c0_in'], 'out': d['c0_out']} if not EXPERIMENT else {}
TRUE_L = {'in': d['L_free_in'], 'out': d['L_free_out']} if not EXPERIMENT else {}
TAPE_POS = {b: list(d[f'pos_{b}']) for b in BARS} if EXPERIMENT else {}
UNITS = d.get('units', 'strain')
SCALE, USYM = (1.0, UNITS) if EXPERIMENT else (1e6, 'ustrain')
PKFMT = '10.4g' if EXPERIMENT else '10.1f'

print(__doc__.split('---')[0].strip())
if EXPERIMENT:
    print(f'\nMEASURED shot     : {d["source"]}')
    print('no ground truth   : the true/error columns print a dash. The only '
          'checks are\n                    internal -- the 2L/c spread across '
          'gauges, and the free-end null.')
print(f'\nrecord            : {N} samples at {dt*1e3:.4f} us  ({t[-1]:.3f} ms)')
print(f'gauges            : {sum(len(v) for v in NAMES.values())} on '
      f'{len(BARS)} bar(s), positions NOT read from config')
for b, n_g in SKIPPED.items():
    print(f'{"":18}  {b} bar SKIPPED: {n_g} gauge(s), and separation needs 2')
if len(BARS) > 1:
    print('the two bars are identified INDEPENDENTLY -- separate templates, '
          'separate\nround trips, separate wave speeds. Nothing crosses '
          'between them.')


# --------------------------------------------------------------------------
# per bar: the three edges
# --------------------------------------------------------------------------
def read_bar(bar):
    """f1, f2, f3 at every gauge of one bar, plus what was rejected on the way."""
    grads, names = GRAD[bar], NAMES[bar]

    # One template per BAR, from whichever of its gauges the wave reaches first,
    # so that every delay on this bar is measured against the same feature. It
    # spans a little more than the measured rise: long enough for a sharp
    # correlation peak, short enough that f2 and f3 -- which close to 2(L-x)/c
    # apart on the NEAREST gauge, 51 us on the shipped aluminium layout -- still
    # resolve into two peaks.
    rise = min(_rise_time(g, dt) for g in grads)
    n_t = max(4, int(round(3.0 * rise / dt)))
    lead = [int(np.argmax(np.abs(g) > 0.3 * np.abs(g).max())) for g in grads]
    i_first = int(np.argmin(lead))
    ir = lead[i_first]
    template = grads[i_first][max(0, ir - n_t // 4): ir + 3 * n_t // 4]

    corr = [_xcorr(g, template) for g in grads]

    # f1: the arrival. Same sense as the template, so a POSITIVE correlation
    # peak; it is also much the largest thing in the record.
    f1 = np.array([_peak(c, +1) * dt for c in corr])

    # Candidates for the two INVERTED edges: the free-end echo f2, and -- on the
    # output bar only -- the moment the bars part, which stops the drive. Both
    # are positive-going strain steps and so negative correlation peaks. They
    # are told apart the way identify_bar_tension.py tells its striker pulse
    # from its echo: the separation delay is IDENTICAL at every gauge on the bar
    # and the echo delay is not. Nothing about the impact has to be known.
    def cands(c, t0, sign, n=5):
        """(delay after t0, |correlation|) of the n strongest edges of a sign."""
        work = c.copy()
        work[:int(t0 / dt + 1.5 * n_t)] = 0.0
        out = []
        for _ in range(n):
            i = _peak(work, sign)
            if not np.isfinite(i):
                break
            out.append((i * dt - t0, abs(work[int(i)])))
            work[max(0, int(i) - n_t // 2): int(i) + n_t // 2] = 0.0
        return out

    neg = [cands(c, a, -1) for c, a in zip(corr, f1)]
    tol = 0.6 * (n_t * dt)

    # EVERY shared delay comes out, not just one. A delay that is identical at
    # every gauge cannot be an echo -- an echo's delay is 2(L-x)/c and depends
    # on x -- so it is something that happened once, at the contact, and
    # propagated to all of them alike. There is usually one such event (the bars
    # parting) and the original code found exactly one.
    #
    # A LONG striker breaks that. The PC shot is driven by 2415 mm of aluminium
    # against a 1027 mm bar, so the striker's own round trip is 944 us against
    # the bar's 1460: it is still in contact when the free-end echo comes back,
    # and its unloading staircase puts a shared edge at 944 us AND another near
    # 1424 us. Finding one and stopping leaves the other in the pool, where it
    # competes with the real echo on amplitude. On this record the real echoes
    # happen to be stronger (0.30 / 0.54 against 0.14 / 0.18) so the answer
    # survives -- but that is luck, and a shot with a softer edge would not get
    # it. Removing them all costs one loop.
    shared = []
    pool = [list(cs) for cs in neg]
    while True:
        found, best = None, 1
        for probe, _ in [x for cs in pool for x in cs]:
            hits = [min((x for x, _ in cs), key=lambda x: abs(x - probe))
                    for cs in pool if any(abs(x - probe) < tol for x, _ in cs)]
            if len(hits) > best:
                best, found = len(hits), float(np.median(hits))
        if found is None or best < len(names):
            break                # shared by some but not all: not a common event
        # strength of the shared edge, for reporting: the mean over gauges
        amp = float(np.mean([max(v for x, v in cs if abs(x - found) < tol)
                             for cs in pool]))
        shared.append((found, amp))
        pool = [[(x, v) for x, v in cs if abs(x - found) > tol] for cs in pool]
    shared.sort()

    # The STRONGEST surviving edge, not the earliest. A wavefront in a lumped
    # chain rings, so the correlation carries sidelobes a few microseconds
    # ahead of the true peak; taking the first candidate locks onto one of
    # those and lands the position several mm out. Amplitude picks the edge.
    f2 = np.array([f1[k] + max(cs, key=lambda p: p[1])[0] if cs else np.nan
                   for k, cs in enumerate(pool)])

    # f3: the same echo re-reflected at the contact end, so it is a negative-
    # going step again -- a POSITIVE correlation peak, after f2.
    f3 = []
    for c, x2, a in zip(corr, f2, f1):
        if not np.isfinite(x2):
            f3.append(np.nan)
            continue
        keep = cands(c, x2 + 0.75 * n_t * dt, +1)
        f3.append(x2 + 0.75 * n_t * dt + max(keep, key=lambda p: p[1])[0]
                  if keep else np.nan)
    f3 = np.array(f3)
    return dict(f1=f1, f2=f2, f3=f3, corr=corr, template=template,
                rise=rise, n_t=n_t, shared=shared, tol=tol)


READ = {b: read_bar(b) for b in BARS}

print('\n--- edges, per gauge '
      '------------------------------------------------------')
print(f'{"gauge":>7} {"peak":>10} {"f1 arrive":>10} {"f2 free end":>12} '
      f'{"f3 round trip":>14} {"2L/c":>10}')
print(f'{"":>7} {"["+USYM+"]":>10} {"[ms]":>10} {"[ms]":>12} {"[ms]":>14} '
      f'{"[us]":>10}')
for b in BARS:
    r = READ[b]
    for k, nm in enumerate(NAMES[b]):
        print(f'{nm:>7} {np.abs(SIG[b][k]).max()*SCALE:{PKFMT}} {r["f1"][k]:10.4f} '
              f'{r["f2"][k]:12.4f} {r["f3"][k]:14.4f} '
              f'{(r["f3"][k]-r["f1"][k])*1e3:10.3f}')
    for k, (x, _) in enumerate(r['shared']):
        what = ('the striker unloading' if k == 0 and len(r['shared']) > 1
                else 'the bars parting')
        print(f'{"":>7} edge shared by every gauge at +{x*1e3:.1f} us -- '
              f'{what}, not an echo; excluded')
    print(f'{"":>7} rise {r["rise"]*1e3:.1f} us (measured) -> template '
          f'{r["n_t"]*dt*1e3:.1f} us')


# --------------------------------------------------------------------------
# per bar: the round trip, then c, then the positions
# --------------------------------------------------------------------------
# R = f3 - f1 = 2L/c is the same at every gauge on a bar, so its spread across
# gauges is a genuine consistency check and a gross outlier -- a gauge whose f2
# and f3 merged, or that locked onto the wrong echo -- throws itself out. This
# is the compression rig's analogue of the tension script's Q, and it is the
# better of the two: Q needs a tape measurement to a GAUGE, R needs one to the
# end of the bar.
ID = {}
for b in BARS:
    r = READ[b]
    R = r['f3'] - r['f1']
    ok = np.isfinite(R)
    if ok.sum():
        ok &= np.abs(R - np.median(R[ok])) < 0.01 * np.median(R[ok])
    if ok.sum() < 1:
        raise SystemExit(f'{b} bar: no gauge gave a usable round trip')
    R_MEAN = float(np.mean(R[ok]))

    # THE measured length for this bar. The model's own geometry is the
    # FALLBACK, which makes the run a self-check rather than an instrument.
    if L_REF_CFG[b] is None:
        if b not in TRUE_L:
            raise SystemExit(
                f'{b} bar: no reference length. A measured shot has no model '
                f'geometry to fall\nback on, so L_free_{b}_ref must be in '
                f'config.toml or given as --l-{b}-ref. One\nlength per bar is '
                'the irreducible minimum -- see "Two bars, two scales" above.')
        L_ref, src = TRUE_L[b], 'model geometry -- nothing configured, SELF-CHECK'
    else:
        L_ref, src = float(L_REF_CFG[b]), 'tape, bench measurement of the bar'
        # Only a simulated record has a geometry to be checked against. On a
        # measured one the tape IS the geometry and there is nothing to compare.
        if b in TRUE_L and abs(L_ref - TRUE_L[b]) > L_REF_TOL:
            print(f'\n!! WARNING: {b} bar reference length is {L_ref:.1f} mm but '
                  f'the model\n!! geometry says {TRUE_L[b]:.1f} mm, a slip of '
                  f'{L_ref - TRUE_L[b]:+.1f} mm, outside +-{L_REF_TOL:.1f} mm. '
                  'Has a bar length changed?')

    c_id = 2.0 * L_ref / R_MEAN
    # x = L_ref (f3 - f2) / R: PROPORTIONAL to the supplied length, so the tape
    # lands on it as a relative band. identify_bar_tension.py cannot do this --
    # its x is L_free plus a constant the tape never scales, so there the same
    # +-2 mm arrives as +-2 mm on the position instead of +-0.13 mm.
    x_id = L_ref * (r['f3'] - r['f2']) / R_MEAN
    ID[b] = dict(R=R, ok=ok, R_MEAN=R_MEAN, L_ref=L_ref, src=src,
                 c=c_id, x=x_id, L_free=L_ref - x_id)

print('\n--- wave speed, one per bar '
      '-----------------------------------------------')
print('the two bars are acoustically independent once they part, so each needs '
      'its\nown measured length and neither constrains the other.\n')
print(f'{"bar":>5} {"L_ref [mm]":>11} {"R = 2L/c [ms]":>15} {"spread":>10} '
      f'{"c [mm/ms]":>11} {"true":>10} {"rel err":>10}')
for b in BARS:
    i, r = ID[b], READ[b]
    spread = np.ptp(i['R'][i['ok']]) * 1e3 if i['ok'].sum() > 1 else 0.0
    _tc = TRUE_C.get(b)
    print(f'{b:>5} {i["L_ref"]:11.1f} {i["R_MEAN"]:15.5f} {spread:9.3f}u '
          f'{i["c"]:11.3f} {_tc:10.3f} {i["c"]/_tc-1:+10.2e}' if _tc else
          f'{b:>5} {i["L_ref"]:11.1f} {i["R_MEAN"]:15.5f} {spread:9.3f}u '
          f'{i["c"]:11.3f} {'-':>10} {'-':>10}')
    print(f'{"":>5} source: {i["src"]}')

print('\n--- gauge positions '
      '-------------------------------------------------------')
_ref_lbl = 'tape' if EXPERIMENT else 'true'
print(f'{"gauge":>7} {"x (from face)":>15} {"+-tape":>8} {_ref_lbl:>9} '
      f'{"error":>9} {"L_free":>10}')
for b in BARS:
    i = ID[b]
    band = L_REF_TOL / i['L_ref'] * i['x']       # relative, not absolute
    # On a measured shot the tape positions are shown for COMPARISON only. They
    # are an independent measurement of the same quantity, not ground truth, and
    # the identification was not given them.
    ref = TRUE_POS.get(b) or TAPE_POS.get(b)
    for k, nm in enumerate(NAMES[b]):
        if ref is None:
            print(f'{nm:>7} {i["x"][k]:15.2f} {band[k]:8.2f} {"-":>9} '
                  f'{"-":>9} {i["L_free"][k]:10.2f}')
        else:
            print(f'{nm:>7} {i["x"][k]:15.2f} {band[k]:8.2f} {ref[k]:9.2f} '
                  f'{i["x"][k]-ref[k]:+9.3f} {i["L_free"][k]:10.2f}')
if EXPERIMENT and TAPE_POS:
    print('the "tape" column is an INDEPENDENT MEASUREMENT, not truth -- the '
          'identification\nwas never shown it. A common offset shared by every '
          'gauge is benign (it moves\nwhere the wave is reconstructed, not its '
          'shape); a spread between them is not.')

print('\n--- gauge spacing D, which is what the reduction leans on '
      '-----------')
# TWO ROUTES TO D, and they are not equally good.
#
#   x from (f3 - f2):  both features have travelled the SAME path to the same
#                      gauge, so whatever an attenuating bar does to an edge it
#                      does to both, and the difference is clean.
#   c * lag:           the arrival lag compares gauge 1 with gauge 2, and the
#                      far gauge's edge has propagated D further. In a lossy bar
#                      that edge is broader and its correlation peak lands
#                      LATER, so this route reads long.
#
# On the elastic simulated bars the two agree and this is a formality. On the
# real PC bar they differ by 3 mm against 0.9 mm of error, and the difference is
# a MEASUREMENT of the edge-broadening bias rather than a fault. Print both.
_ref_lbl2 = 'tape' if EXPERIMENT else 'true'
print(f'{"bar":>5} {"D (f3-f2)":>11} {"D = c*lag":>11} {"broadening":>11} '
      f'{_ref_lbl2:>9} {"error":>9} {"+-tape":>9}')
for b in BARS:
    i, r = ID[b], READ[b]
    if len(NAMES[b]) < 2:
        continue
    D_id = abs(i['x'][1] - i['x'][0])
    D_lag = i['c'] * abs(r['f1'][1] - r['f1'][0])
    ref = TRUE_POS.get(b) or TAPE_POS.get(b)
    D_ref = abs(ref[1] - ref[0]) if ref else None
    cols = (f'{b:>5} {D_id:11.3f} {D_lag:11.3f} {D_lag-D_id:+11.3f} ')
    cols += (f'{D_ref:9.2f} {D_id-D_ref:+9.3f} ' if D_ref is not None
             else f'{"-":>9} {"-":>9} ')
    print(cols + f'{L_REF_TOL/i["L_ref"]*D_id:9.3f}')
print('the "broadening" column is c*lag minus D: how much longer the '
      'arrival-lag route\nreads because the far gauge sees a BROADER edge. It '
      'is ~1 us here from the lumped\nchain\'s own numerical dispersion and '
      '3 us on the real PC bar, where the loss is\nphysical. Either way D from '
      '(f3 - f2) is the route to use: both of its features\ntravelled the same '
      'path, so the bias cancels.')
_lev = [ID[b]['L_ref'] / abs(ID[b]['x'][1] - ID[b]['x'][0])
        for b in BARS if len(NAMES[b]) > 1]
print(f'leverage L_ref/D = {", ".join(f"{v:.1f}" for v in _lev)}: a tape good to '
      f'+-{L_REF_TOL:.1f} mm on the bar\nbuys D to the +-tape column above. '
      'Unlike the tension rig, the positions get\nthe same relative treatment '
      '-- they scale with L_ref rather than sitting on\ntop of a constant.')

print('\n--- transit times, which is what separate() really needs '
      '-------------')
print(f'{"gauge":>7} {"x/c [us]":>12} {"true [us]":>12} {"rel err":>10}')
for b in BARS:
    i = ID[b]
    for k, nm in enumerate(NAMES[b]):
        a = i['x'][k] / i['c']
        if b in TRUE_POS and b in TRUE_C:
            c = TRUE_POS[b][k] / TRUE_C[b]
            print(f'{nm:>7} {a*1e3:12.4f} {c*1e3:12.4f} {a/c-1:+10.2e}')
        else:
            print(f'{nm:>7} {a*1e3:12.4f} {"-":>12} {"-":>10}')

# --------------------------------------------------------------------------
# attenuation -- only where the record shows it
# --------------------------------------------------------------------------
# A metal bar is lossless enough to ignore and this section then does nothing.
# A POLYMER bar is not, and `separate` fitting one lossless pair of waves to two
# gauges that disagree about the wave's SHAPE is what puts a floor under
# everything below. alpha(f) is measured from the two gauges themselves --
# magnitudes only, no boundary condition -- so the free-end null underneath
# stays an independent check of it rather than its objective.
ATT = {b: None for b in BARS}
if EXPERIMENT and 'attenuation' in cfg:
    from identify_attenuation import fit_attenuation
    ac = cfg['attenuation']
    print('\n--- attenuation, from the two gauges alone '
          '--------------------------')
    for b in BARS:
        i, r = ID[b], READ[b]
        try:
            a = fit_attenuation(t, SIG[b], i['x'], r['f1'], r['f2'],
                                f_lo=float(ac.get('f_lo', 2.0)),
                                f_hi=float(ac.get('f_hi', 50.0)),
                                snr=float(ac.get('snr', 0.005)))
        except ValueError as exc:
            print(f'{b:>5} not measurable: {exc}')
            continue
        ATT[b] = a['table']
        fb, ab = a['table']
        print(f'{b:>5} single-wave window {a["span"]*1e3:.0f} us, '
              f'{a["pairs"]} gauge pair(s), band {a["f_lo"]:.0f}-{a["f_hi"]:.0f} kHz')
        print(f'{"":>5} alpha at 10 / 25 / {a["f_hi"]:.0f} kHz: '
              f'{np.interp(10, fb, ab):.2e} / {np.interp(25, fb, ab):.2e} / '
              f'{ab[-1]:.2e} /mm   (linear-law summary k = {a["k"]:.2e})')
        print(f'{"":>5} far gauge predicted from the near one: '
              f'{a["misfit"]:.3f} relative L2, against {a["misfit_lossless"]:.3f} '
              f'lossless')
        print(f'{"":>5} c_p from the transfer-function PHASE: {a["c_p"]:.1f} '
              f'mm/ms, an independent look at c ({i["c"]:.1f} from 2L/c)')
    print(f'{"":>5} the table ENDS at f_hi and np.interp holds it there, which '
          'is the band\n      limit: exp(+alpha x) on the minus branch would '
          'otherwise grow without bound.')

# --------------------------------------------------------------------------
# free-end null test -- on BOTH bars, because both far ends are free
# --------------------------------------------------------------------------
# Reconstruct at the free surface by handing `separate` the identified L_free,
# which are distances FROM that surface. The stress there is zero at all times,
# so eps_plus + eps_minus must vanish. Nothing here consults the true geometry,
# which makes it the one validation that survives contact with a real rig.
#
# The SHTB gets this on one bar only -- its input bar ends on the anvil. Here
# both bars are free at the far end and both are tested, which is two
# independent screens instead of one.
#
# The tail MUST be cut: the exponential window that regularises separate()
# amplifies the record-end truncation and inflates the residual ~100x. See
# "The free-end null test" in README.md.
# One threshold PER BAR: the two are different materials and their floors
# differ by 3x, so a single number would either pass everything on one bar or
# fail everything on the other. config.toml carries both, with the measured
# floors that set them.
# An experiment case carries one [.null] table -- one instrumented bar, one
# threshold -- where a simulated two-bar case carries null_tol_in / null_tol_out.
if EXPERIMENT:
    _nc = cfg.get('null', {})
    NULL_WINDOW = float(_nc.get('window', 0.75))
    NULL_TOL = {b: float(_nc.get('tol', 1.0e-1)) for b in BARS}
else:
    NULL_WINDOW = cfg.get('null_window', 0.75)
    NULL_TOL = {b: cfg.get(f'null_tol_{b}', 1.0e-1) for b in BARS}

print('\n--- free-end null test (no ground truth used) '
      '-----------------------------')
if len(BARS) > 1:
    print('both far ends are free, so both bars are screened -- the SHTB gets '
          'this on one\nbar only.')
print('The floor is high on a direct impact: it starts from a velocity STEP, '
      'and the\ndispersed wake behind that wavefront is not something one '
      'uniform bar can fit at\ntwo gauges at once. Where the bar is also lossy '
      'the attenuation above is what\nbrings it down -- and that is a fair '
      'test, because alpha was fitted to gauge\nmagnitudes and never to this '
      'boundary condition.\n')
print(f'{"bar":>5} {"peak |wave+|":>13} {"rms":>11} {"max":>11} '
      f'{"threshold":>11} {"verdict":>9}')
NULL = {}
def _null(b, attenuation):
    i = ID[b]
    p, m = separate(t, SIG[b], i['L_free'], c0=i['c'], eta=d['eta'],
                    attenuation=attenuation)
    tot, amp = p + m, np.abs(p).max()
    i0 = int(np.argmax(np.abs(p) > 0.02 * amp))
    i1 = int(NULL_WINDOW * N)
    w = slice(i0, i1)
    return dict(p=p, m=m, tot=tot, amp=amp, w=w, i0=i0, i1=i1,
                rms=float(np.sqrt(np.mean(tot[w] ** 2)) / amp),
                mx=float(np.abs(tot[w]).max() / amp), tol=NULL_TOL[b])

for b in BARS:
    NULL[b] = _null(b, ATT[b])
    r = NULL[b]
    _pk = (f'{r["amp"]*SCALE:12.1f}u' if not EXPERIMENT
           else f'{r["amp"]*SCALE:12.4g} ')
    print(f'{b:>5} {_pk}{r["rms"]:11.2e} {r["mx"]:11.2e} '
          f'{NULL_TOL[b]:11.1e} '
          f'{"PASS" if r["rms"] <= NULL_TOL[b] else "FAIL":>9}')
    # What the attenuation was worth, on a test it was not fitted to.
    if ATT[b] is not None:
        print(f'{"":>5} lossless, same positions and c: '
              f'{_null(b, None)["rms"]:.2e} -- the attenuation is worth '
              f'{_null(b, None)["rms"]/r["rms"]:.1f}x here, on a boundary '
              'condition it never saw')
print('\na FAIL is conclusive; a PASS is weak evidence. The test constrains '
      'L_free/c\nand nothing else, so it cannot see a tape error at all -- '
      'scaling L_free and c\ntogether leaves it unchanged.')

# --------------------------------------------------------------------------
# density and modulus -- not identifiable, closed with a weighing
# --------------------------------------------------------------------------
print('\n--- density and modulus '
      '---------------------------------------------------')
print('NOT identifiable from strain records: they fix c = sqrt(E/rho) and no '
      'more.\nClosed here with one extra measurement per bar, its mass. HERE '
      "that is computed\nback from the simulator's own rho, so the rho line is "
      'circular; the E line is\nnot, because it uses the IDENTIFIED c.\n')
print(f'{"bar":>5} {"rho [kg/mm3]":>14} {"E = rho c^2":>12} {"true":>9} '
      f'{"rel err":>10} {"E*A [kN]":>11}')
for b in BARS:
    i = ID[b]
    rho, A = d.get(f'rho_{b}'), d.get(f'A_{b}')
    if rho is None:
        print(f'{b:>5} {"-":>14}   no density supplied; E cannot be closed')
        continue
    E_id = rho * i['c'] ** 2
    E_true = d.get(f'E_{b}')
    tail = (f'{E_true:9.3f} {E_id/E_true-1:+10.2e} ' if E_true is not None
            else f'{"-":>9} {"-":>10} ')
    print(f'{b:>5} {rho:14.4e} {E_id:12.3f} ' + tail +
          (f'{E_id*A:11.1f}' if A is not None else f'{"-":>11}'))
if EXPERIMENT:
    print('rho above is an ASSUMED handbook value, so E inherits its error '
          'directly and\nis a closure rather than a measurement. Note the '
          'reconstruction never asks for\neither: force in gives force out, '
          'and only c, the positions and eta are used.')

print('\n--- ready to use '
      '----------------------------------------------------------')
for b in BARS:
    i = ID[b]
    print(f'  {b:>3} bar:  c0 = {i["c"]:.3f}   gauges = '
          f'[{", ".join(f"{p:.2f}" for p in i["x"])}]    # mm from its face')

# The uninstrumented bar is not identified, but a shot with a long striker
# hands over its wave speed anyway: the shared edge every gauge sees is the
# striker's own release returning, so its delay is 2 L_striker / c_striker. It
# costs nothing and it is a check on the whole time base.
_P = min((r['shared'][0][0] for r in READ.values() if r['shared']), default=None)
_L_in = L_REF_CFG.get('in')
if _P and _L_in and 'in' in SKIPPED:
    print(f'  {"in":>3} bar:  c0 = {2.0*float(_L_in)/_P:.3f}   '
          f'# from the striker pulse 2L/c = {_P*1e3:.1f} us; no gauge on it, '
          'so nothing else')

# --------------------------------------------------------------------------
# hand the numbers on
# --------------------------------------------------------------------------
# The same split the simulators use: one script produces, another consumes, and
# a file in between so that iterating on the reconstruction does not mean
# re-running the identification. reconstruct_interface.py reads this.
#
# The TAPE positions travel with it and are kept SEPARATE from the identified
# ones. They are two independent measurements of the same thing and the point
# downstream is to run both and compare, not to have quietly chosen one here.
IDENT_FILE = 'bar_identified.npz'
_out = dict(case=CASE, bars=np.array(BARS))
for b in BARS:
    i = ID[b]
    _out[f'c_{b}'] = i['c']
    _out[f'x_{b}'] = i['x']
    _out[f'L_free_{b}'] = i['L_free']
    _out[f'L_ref_{b}'] = i['L_ref']
    _out[f'R_{b}'] = i['R_MEAN']
    _out[f'f1_{b}'] = READ[b]['f1']
    _out[f'f2_{b}'] = READ[b]['f2']
    _out[f'f3_{b}'] = READ[b]['f3']
    if b in TAPE_POS:
        _out[f'tape_{b}'] = np.asarray(TAPE_POS[b], float)
    if ATT[b] is not None:
        _out[f'alpha_f_{b}'], _out[f'alpha_{b}'] = ATT[b]
np.savez(IDENT_FILE, **_out)
print(f'\nwrote {IDENT_FILE}: '
      + ', '.join(f'{b} c0={ID[b]["c"]:.1f}' for b in BARS)
      + (', with alpha(f)' if any(ATT[b] is not None for b in BARS) else ''))

# --------------------------------------------------------------------------
# figure -- one column per bar, because the two are separate identifications
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt   # backend already chosen by plotting.init

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'

fig, axes = plt.subplots(4, len(BARS), figsize=(7.5 * len(BARS), 13),
                         squeeze=False)
fig.patch.set_facecolor(SURFACE)
for j in range(len(BARS)):
    axes[1, j].sharex(axes[0, j])
    axes[3, j].sharex(axes[2, j])

for j, b in enumerate(BARS):
    r, i, nul = READ[b], ID[b], NULL[b]
    k = int(np.nanargmin(r['f1']))            # the gauge the wave reaches first
    title = f'{"Input" if b == "in" else "Output"} bar'
    if EXPERIMENT:
        title += ' (measured)'

    # --- the record, with the three edges the method uses ------------------
    axes[0, j].plot(t, SIG[b][k] * SCALE, color=BLUE, lw=.9)
    axes[0, j].set_ylabel(f'Signal ({USYM})')
    axes[0, j].set_title(f'{title} — calibration shot at {NAMES[b][k]}, '
                         f'c = {i["c"]:.0f} mm/ms', loc='left', fontsize=11)

    # --- the edge filter ---------------------------------------------------
    cc = r['corr'][k]
    axes[1, j].plot(t, cc / np.abs(cc).max(), color=INK, lw=.9)
    marks = [('f1  arrives', r['f1'][k], ORANGE),
             ('f2  free end', r['f2'][k], ORANGE),
             ('f3  round trip', r['f3'][k], BLUE)]
    for x, _ in r['shared']:
        marks.append(('shared edge', r['f1'][k] + x, MUTED))
    for lab, tt, col in marks:
        if np.isfinite(tt):
            axes[1, j].axvline(tt, color=col, lw=1.1, ls='--')
            axes[1, j].annotate(lab, (tt, 1.0), rotation=90, fontsize=8,
                                color=MUTED, va='top', ha='right')
    axes[1, j].axhline(0, color=GRID, lw=.8)
    axes[1, j].set_xlabel('Time (ms)')
    axes[1, j].set_ylabel('Edge filter (norm.)')
    axes[1, j].set_title('Differentiated record, matched against the leading '
                         f'edge — f3 − f1 = 2L/c = {i["R_MEAN"]*1e3:.1f} us',
                         loc='left', fontsize=11)
    axes[1, j].set_xlim(0, min(t[-1], r['f3'][k] * 1.25))

    # --- the free-end null, made visible -----------------------------------
    axes[2, j].plot(t, nul['p'] * SCALE, color=BLUE, lw=.9,
                    label=r'$\varepsilon_+$ (toward the free end)')
    axes[2, j].plot(t, nul['m'] * SCALE, color=ORANGE, lw=.9,
                    label=r'$\varepsilon_-$ (reflected)')
    axes[2, j].set_ylabel(f'Signal ({USYM})')
    axes[2, j].set_title('Reconstructed AT the free surface — a free end '
                         'inverts, so these mirror', loc='left', fontsize=11)
    axes[2, j].legend(frameon=False, fontsize=9, labelcolor=MUTED,
                      loc='upper left')
    # headroom, so the legend sits above the traces rather than across them
    _pk = max(np.abs(nul['p']).max(), np.abs(nul['m']).max()) * SCALE
    axes[2, j].set_ylim(-1.15 * _pk, 1.45 * _pk)

    band = nul['tol'] * nul['amp'] * SCALE
    axes[3, j].axhspan(-band, band, color=BLUE, alpha=.18,
                       label=f'threshold ±{nul["tol"]:.1e} × peak |ε₊|')
    axes[3, j].plot(t, nul['tot'] * SCALE, color=INK, lw=.9,
                    label=r'$\varepsilon_+ + \varepsilon_-$  (= stress / E)')
    axes[3, j].axhline(0, color=GRID, lw=.8)
    for _b in (t[nul['i0']], t[nul['i1']]):
        axes[3, j].axvline(_b, color=ORANGE, lw=1.1, ls='--')
    axes[3, j].set_xlabel('Time (ms)')
    axes[3, j].set_ylabel(f'Signal ({USYM})')
    axes[3, j].set_title(f'Free-surface stress — rms {nul["rms"]:.2e} of peak '
                         f'|ε₊|, '
                         f'{"PASS" if nul["rms"] <= nul["tol"] else "FAIL"}',
                         loc='left', fontsize=11)
    axes[3, j].legend(frameon=False, fontsize=9, labelcolor=MUTED,
                      loc='upper left')
    # Scale to the residual inside the window, not the truncation outside it.
    _r = np.abs(nul['tot'][nul['w']]).max() * SCALE
    axes[3, j].set_ylim(-3 * _r, 3 * _r)
    axes[2, j].set_xlim(0, t[-1]); axes[3, j].set_xlim(0, t[-1])

for ax in axes.ravel():
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.suptitle('Direct-impact bar identified from a no-specimen calibration shot'
             + (' — the two bars are independent problems' if len(BARS) > 1
                else f' — {d["source"].rsplit("/", 1)[-1]}' if EXPERIMENT
                else ''),
             x=.006, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .975))
FIG = (f'bar_identification_{CASE}.png' if EXPERIMENT
       else 'bar_identification_compression.png')
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'\nwrote {FIG}')

plotting.show_unless(HEADLESS)
