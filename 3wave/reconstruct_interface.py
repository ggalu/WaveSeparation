"""
Reconstruct the FORCE at the impact interface, and check it against the physics
the rig itself guarantees.

    python3 identify_bar_compression.py cases/identifications/pc_bar
    python3 reconstruct_interface.py cases/identifications/pc_bar  # that shot
    python3 reconstruct_interface.py cases/analyses/pc_specimen    # a specimen
                                                                   # shot on it

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
What the figure shows
--------------------------------------------------------------------------
The same rows as identify_bar_tension.py's figure, minus its matched-filter
edge row -- there is nothing to time here, since a specimen is exactly what
destroys the edges, and nothing on this record is being identified:

    row 0, per bar   what was measured
    row 1, per bar   F = P + M at THAT bar's own face
    row 2, left      force equilibrium across the interface, both bars overlaid
    row 2, right     the free-end null, at BAR's own far free surface

The bottom row is the pair of checks that need no ground truth. Neither is
possible on a single bar's record alone: the equilibrium needs the OTHER bar's
independent solve, and the null needs the identified distance to a free
surface. Both come out of bar_identified.npz, which is the point -- on a real
shot the specimen has destroyed every feature the identification reads, so
c0, the gauge positions, alpha(f), c_p(f) and the free-end distances all come
from a calibration shot fired on the same bars, and NOTHING here is identified
from the record being reduced.

Which makes the free-end null the check that matters most on this script. A
calibration carried over from another shot can be stale -- a gauge re-bonded, a
bar swapped, the wrong case named -- and the null is where that shows up,
because it is the one panel whose answer is known in advance.

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

from wave_separation_code import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('case',
                 help='an identification folder (reconstruct its own shot) or '
                      'an analysis folder (a specimen shot, reconstructed '
                      'with the bars its `bars` folder identified)')
_ap.add_argument('--bar', default=None,
                 help='which bar, when the identification covered more than '
                      'one. Default: the only one, or "out".')
_ap.add_argument('--no-attenuation', action='store_true',
                 help='ignore the identified alpha(f) and reconstruct with a '
                      'lossless bar, for comparison.')
_ap.add_argument('--no-dispersion', action='store_true',
                 help='ignore the identified c_p(f)/c0 and reconstruct with '
                      'c_p = c0 at every frequency, for comparison. '
                      'Independent of --no-attenuation: dispersion is an '
                      'elastic effect (Pochhammer-Chree), not a lossy one.')
HEADLESS, ARGS = plotting.init(parser=_ap)

from wave_separation_code import cases
from wave_separation_code import config
from wave_separation_code.wave_separation import separate, separate_field, wavefront_time

cfg = config.load(ARGS.case)
if cfg['kind'] == 'simulation':
    raise SystemExit(f'{ARGS.case} is a simulation; reconstruct an '
                     'identification or an analysis folder')
CASE = cfg['case']
# SELF: reconstructing the identification's own shot, rather than a specimen
# shot that borrows its bars.
SELF = cfg['kind'] == 'identification'
BARS_DIR = cases.rel(cases.bars_dir(cfg))
IDENT_FILE = f'{BARS_DIR}/{cases.IDENT_FILE}'
ID = cases.identification(cfg)
BARS = [str(b) for b in ID['bars']]
BAR = ARGS.bar or ('out' if 'out' in BARS else BARS[0])
if BAR not in BARS:
    raise SystemExit(f'{IDENT_FILE} covers {BARS}, not {BAR!r}')

d = cases.record(cfg)
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

# Each gauge's own distance to its bar's far FREE surface, as identified. This
# is a property of the BAR -- swapping the coupler for a specimen does not move
# the output bar's free end -- so it travels with the calibration to any later
# shot, exactly like x and c0 do.
#
# It has to be read, not reconstructed as L - x. L_ref_{b} is the assembly-wide
# L_free_ref under identify_bar_tension.py's default route (3730 mm from an
# in-bar gauge, across the joint), not either bar's own length to its own free
# end, so L - x there is off by the whole input bar -- 3611 mm against the true
# 2658 on experiment_tension_bar_2, and the free-end null built on it is noise.
# The compression identifier's L_ref IS that bar's own length, which is why the
# subtraction looked right for as long as only that one was in use.
L_FREE = {}
for _b in BARS:
    if f'L_free_{_b}' in ID.files:
        L_FREE[_b] = np.asarray(ID[f'L_free_{_b}'], float)
    else:
        L_FREE[_b] = (float(ID[f'L_ref_{_b}'])
                      - np.asarray(ID[f'x_{_b}'], float))
        print(f'!! {IDENT_FILE} carries no L_free_{_b}; falling back to '
              f'L_ref - x, which\n!! is only that bar\'s free-end distance if '
              'L_ref is its OWN length. Re-run the\n!! identification to get '
              'the identified value.')

# On the SHTB the input bar's own far end is the anvil, not a free surface, so
# its L_free is measured the long way round -- through whatever sat at the
# interface during the calibration, and then down the whole output bar. That
# path is intact only while the interface is the one the calibration saw. Swap
# the coupler for a specimen and the distance is still the same mm of bar but
# no longer the same acoustic path, so the null on this bar stops meaning
# anything. Say so rather than print a number that looks like a check.
NULL_VALID = not (BAR == 'in' and not SELF)

# What sits at x = 0. The calibration shot has the two bars touching directly;
# a specimen shot has something in between. Only the REPORTING depends on it --
# the solve is identical, because `separate` was never told about boundaries.
IMPACT = str(cfg.get('interface', 'impact')) == 'impact'
IFACE = ('impact interface' if IMPACT else
         f'{BAR}put-bar / specimen interface')
# The UNILATERAL check (F >= 0) assumes a dry contact that cannot pull -- true
# of a direct-impact bar or a compression specimen pressed between two bars,
# false of a bonded tension joint, which legitimately carries either sign.
# This is a property of the loading convention, not of IMPACT/interface: even
# a bonded COMPRESSION specimen still cannot pull (see the comment in
# checks()), so TENSION is what gates it, not whether a specimen is present.
TENSION = str(d.get('loading', cfg.get('loading'))) == 'tension'

# The identified numbers belong to a BAR. Reconstructing a different shot with
# them is the whole point of calibrating -- but it is only valid on the SAME
# bar, and a length mismatch is the cheap way to catch a case pointed at the
# wrong one.
_L_cfg = float(cfg.get('bar', {}).get('length', L))
if abs(_L_cfg - L) > 1.0:
    raise SystemExit(
        f'{IDENT_FILE} identified a {L:.1f} mm bar but {CASE} describes one '
        f'{_L_cfg:.1f} mm long.\nThose are not the same bar. Re-run the '
        'identification for this rig, or fix the case.')

ATT = None
if f'alpha_{BAR}' in ID.files and not ARGS.no_attenuation:
    ATT = (np.asarray(ID[f'alpha_f_{BAR}'], float),
           np.asarray(ID[f'alpha_{BAR}'], float))

DISP = None
if f'dispersion_{BAR}' in ID.files and not ARGS.no_dispersion:
    DISP = (np.asarray(ID[f'dispersion_f_{BAR}'], float),
            np.asarray(ID[f'dispersion_{BAR}'], float))

# --------------------------------------------------------------------------
# the OTHER bar, when the identification covered both -- its own waves and
# the force equilibrium across the shared interface, nothing more. It is
# deliberately NOT run through checks()/free_end(): those assume THIS bar's
# own far end is free, which holds for 'out' on this rig (an SHTB calibration
# shot) but not for 'in', whose far end is the anvil. L_ref_{b} in
# bar_identified.npz is also the ASSEMBLY-wide L_free_ref, duplicated per bar
# for the c0 lookup -- not either bar's own length to its own free end -- so
# free_end() would silently reconstruct nonsense if pointed at it here.
# --------------------------------------------------------------------------
OTHER = next((b for b in BARS if b != BAR), None)
if OTHER is not None:
    sig_o = list(d[f'eps_{OTHER}'])
    c0_o = float(ID[f'c_{OTHER}'])
    x_o = np.asarray(ID[f'x_{OTHER}'], float)
    ATT_o = None
    if f'alpha_{OTHER}' in ID.files and not ARGS.no_attenuation:
        ATT_o = (np.asarray(ID[f'alpha_f_{OTHER}'], float),
                np.asarray(ID[f'alpha_{OTHER}'], float))
    DISP_o = None
    if f'dispersion_{OTHER}' in ID.files and not ARGS.no_dispersion:
        DISP_o = (np.asarray(ID[f'dispersion_f_{OTHER}'], float),
                 np.asarray(ID[f'dispersion_{OTHER}'], float))

print(__doc__.split('---')[0].strip())
print(f'\nrecord     : {d.get("source", "dump.npz")}')
print(f'bar        : {BAR}, {L:.1f} mm, c0 = {c0:.2f} mm/ms, '
      f'2L/c = {R*1e3:.1f} us (measured)')
print(f'signals    : {len(sig)} gauges in {UNITS}, eta = {eta:g} /ms')
print(f'x = 0 is   : the {IFACE}')
_h = float(cfg.get('holder_length', 0.0))
print(f'holder     : {_h:g} mm (holder_length) -- forces reported at '
      + (f'x = -{_h:g} mm, the holder/specimen interface' if _h else
         'the bar faces'))
if not SELF:
    print(f'reusing    : c0, positions and alpha(f) identified in '
          f'{BARS_DIR}.\n             Nothing is identified from THIS '
          'record -- they are properties of the bar.')
print(f'attenuation: ' + ('lossless (--no-attenuation)' if ARGS.no_attenuation
                          else 'none identified' if ATT is None else
                          f'alpha(f) up to {ATT[0][-1]:.0f} kHz, '
                          f'{ATT[1][-1]:.2e} /mm there'))
print(f'dispersion : ' + ('c_p = c0 (--no-dispersion)' if ARGS.no_dispersion
                          else 'none identified' if DISP is None else
                          f'c_p/c0(f) up to {DISP[0][-1]:.0f} kHz, '
                          f'{DISP[1][-1]:.4f} there'))


# --------------------------------------------------------------------------
# the reconstruction, and the checks on it
# --------------------------------------------------------------------------
def reconstruct(x, gsig=sig, gc0=c0, gatt=ATT, gdisp=DISP):
    """P, M and F = P + M at the contact plane, for one set of gauge positions.

    gsig/gc0/gatt/gdisp default to the primary BAR's own signals/c0/attenuation/
    dispersion, so every existing call site is unaffected; the OTHER bar's
    reconstruction below passes its own instead, reusing this rather than
    repeating the separate() call.
    """
    p, m = separate(t, gsig, x, c0=gc0, eta=eta, dispersion=gdisp,
                    attenuation=gatt)
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

    P is reconstructed AT x = 0, so P's own onset is the moment the wave left --
    taken with `wavefront_time`, which anchors on the steepest rise before the
    peak rather than on a first-crossing rule. On a record with a low-level
    precursor a first-crossing rule finds the precursor, and every window keyed
    to this instant moves with it. See that function.
    """
    return wavefront_time(t, p) + R


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
        # The echo lands within a few samples of the record end, so there is no
        # plateau to walk back from. Same (rise, measured?) shape as the normal
        # return -- a bare float here used to crash the unpack in checks().
        return 3.0 * float(np.mean(np.diff(t))), False
    top = float(np.max(a[lo:hi]))
    j90 = lo + int(np.argmax(a[lo:hi] > 0.9 * top))
    # Do not walk back past the moment the wave LEFT x = 0. Nothing before that
    # can be part of this echo's edge, and on a record whose M never returns to
    # a clean zero -- 2026-08-20_PC_AFC.txt, where a precursor contaminates the
    # low-frequency split -- an unbounded walk-back runs to the start of the
    # record and swallows the whole window it was supposed to clear.
    floor = int(np.searchsorted(t, t_echo - R))
    j10 = j90
    while j10 > floor and a[j10] > 0.1 * top:
        j10 -= 1
    return (max(float(t[j90] - t[j10]), 3.0 * float(np.mean(np.diff(t)))),
            j10 > floor)


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
    t_echo = _echo_time(p)
    # wavefront_time() anchors on the steepest rise before the GLOBAL peak of
    # |P|, which assumes that peak belongs to the initial loading. On a record
    # where P keeps building past the initial edge -- multiple reflections in
    # a short assembly loading a real specimen, as opposed to the calibration
    # shot's one clean pulse -- the global peak can sit late in the record and
    # the search walks back from THAT instead, putting t_echo past the record's
    # own end. The reconstruction itself (F = P + M, the free-end null) never
    # depends on this; only the four boundary-condition checks below do, and
    # they simply do not resolve on a record like that. Say so rather than
    # report a number computed from a clamped, meaningless window.
    echo_in_record = t_echo <= float(t[-1])
    # windows start at the wavefront, not at the record's first stirring
    i_on = int(np.searchsorted(t, t_echo - R))
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
    rise, rise_ok = _echo_rise(m, t_echo)
    i_pre = int(np.searchsorted(t, t_echo - rise))    # M must be quiet before
    i_sep = int(np.searchsorted(t, t_echo + rise))    # contact open after
    i_end = int(0.95 * N)
    # With a specimen at x = 0 the bars never part, so there is no "after
    # separation" to score -- but F >= 0 still holds, for the WHOLE record,
    # since a dry compression interface cannot pull either way.
    i_ten = i_sep if IMPACT else i_end
    # A causality window can come out empty when the echo edge is so broad that
    # it reaches back to the wavefront -- there is then no interval in which M
    # is BOTH after the loading and before the echo, and the check simply does
    # not apply to this record. Say so rather than clamp it into a number.
    causality = (float(np.abs(m[i_on:i_pre]).max() / amp)
                 if echo_in_record and i_pre > i_on + 2 else float('nan'))
    return dict(
        amp=amp, i_on=i_on, i_pre=i_pre, i_sep=i_sep, i_end=i_end, rise=rise,
        causality_ok=(echo_in_record and i_pre > i_on + 2),
        echo_in_record=echo_in_record, rise_ok=rise_ok,
        rise_p=rise_p,
        causality=causality,
        tensile=(float('nan') if TENSION else
                 float(max(0.0, -F[i_on:i_ten].min()) / amp)),
        after=(float(np.sqrt(np.mean(F[i_sep:i_end] ** 2)) / amp)
               if IMPACT and i_end > i_sep else float('nan')),
        peak=float(F.max()),
        t_echo=float(t_echo * 1e3),
        t_open=float(t[min(i_sep, N - 1)] * 1e3),
    )


def free_end(x):
    """
    The free-end null, from one position set: eps_+ + eps_- = 0 at the surface.

    The identified L_FREE belongs to the identified x. A different position set
    is the same gauges measured from a different origin ON THE SAME BAR, so it
    shifts L_free by exactly what it shifts x, the other way -- which is what
    makes running the tape set through this test meaningful rather than
    circular.
    """
    lf = L_FREE[BAR] + (x_id - np.asarray(x, float))
    p, m = separate(t, sig, lf, c0=c0, eta=eta,
                    dispersion=DISP, attenuation=ATT)
    tot, amp = p + m, float(np.abs(p).max())
    w = slice(int(np.argmax(np.abs(p) > 0.02 * amp)),
              int(float(cfg.get('null', {}).get('window', 0.75)) * N))
    return dict(p=p, m=m, tot=tot, amp=amp, w=w, L_free=lf,
                rms=float(np.sqrt(np.mean(tot[w] ** 2)) / amp),
                max=float(np.abs(tot[w]).max() / amp))


SETS = [('identified', x_id)]
if x_tape is not None:
    SETS.append(('tape', x_tape))
RES = {}
NULL = {}
for name, x in SETS:
    p, m, F = reconstruct(x)
    RES[name] = dict(x=x, p=p, m=m, F=F, **checks(p, m, F))
    NULL[name] = free_end(x)
    RES[name]['null'] = NULL[name]['rms']
NULL_TOL = float(cfg.get('null', {}).get('tol', cfg.get('null_tol', 5.0e-3)))

# The other bar's own waves at ITS face, and the force each side of the
# interface implies -- should agree, since force is continuous across it.
# |F_BAR - F_OTHER| / max|F_BAR| is wave_separation.specimen_response's own
# `equilibrium` field, reused directly rather than through the rest of that
# function's velocity/strain machinery, which assumes a deforming specimen.
if OTHER is not None:
    p_o, m_o, F_o = reconstruct(x_o, gsig=sig_o, gc0=c0_o, gatt=ATT_o,
                                gdisp=DISP_o)
    _r0 = RES[SETS[0][0]]
    _peak_eq = float(np.abs(_r0['F']).max())
    equilibrium = np.abs(_r0['F'] - F_o) / (_peak_eq if _peak_eq > 0 else 1.0)
    # Force continuity across the interface is a property of the two
    # RECONSTRUCTIONS alone -- it needs no wavefront search, and i_on (the
    # window's usual start) is only used here to skip the quiescent lead-in.
    # When checks()'s echo-timing search resolved, keep that start exactly as
    # before; when it did not (echo_in_record False -- a record whose P has no
    # single clean peak, see above), i_on is not trustworthy and this check
    # should not inherit that failure, so fall back to the whole record.
    _win = slice(_r0['i_on'] if _r0['echo_in_record'] else 0, _r0['i_end'])

# The interface force by BAR NAME rather than by which one --bar made primary.
# The figure and the .dat both want it that way round: F_in is the input bar's
# face whichever bar the checks happened to run on.
F_BY = {BAR: RES[SETS[0][0]]['F']}
if OTHER is not None:
    F_BY[OTHER] = F_o
DAT_BARS = [b for b in ('in', 'out') if b in F_BY]

# --------------------------------------------------------------------------
# Where the specimen actually is: holder_length past each bar face.
#
# With the specimen held in a holder screwed onto the face, F at the face also
# accelerates the holder, and is not the force on the specimen. The reported
# forces -- the F panels, the equilibrium panel, the .dat -- are therefore
# evaluated at x = -holder_length, the holder/specimen interface, on BOTH bars.
# That extrapolates this bar's own model across the holder, i.e. treats the
# holder as more of the same bar: `separate_field`, exactly as the sliders do.
# holder_length = 0 (specimen straight on the bars) is the face itself, and
# then nothing is recomputed. The checks above -- causality, echo, tensile, the
# free-end null -- belong to the bar's own boundaries and stay at the face.
# --------------------------------------------------------------------------
HOLDER = float(cfg.get('holder_length', 0.0))
_GEOM = {BAR: (sig, c0, ATT, DISP)}
if OTHER is not None:
    _GEOM[OTHER] = (sig_o, c0_o, ATT_o, DISP_o)


def at_plane(b, x, delta=0.0):
    """P and M on bar b, gauges at x, `delta` mm off the specimen plane
    (positive INTO the bar, as in separate_field)."""
    gs, gc, ga, gd = _GEOM[b]
    p_f, m_f, _ = separate_field(t, gs, x, gc, eta, [delta - HOLDER],
                                 dispersion=gd, attenuation=ga)
    return p_f[0], m_f[0]


if HOLDER:
    PLANE = {name: at_plane(BAR, x) for name, x in SETS}
    PLANE_O = at_plane(OTHER, x_o) if OTHER is not None else None
else:
    PLANE = {name: (RES[name]['p'], RES[name]['m']) for name, _ in SETS}
    PLANE_O = (p_o, m_o) if OTHER is not None else None
F_BY[BAR] = sum(PLANE[SETS[0][0]])
if OTHER is not None:
    F_BY[OTHER] = sum(PLANE_O)
    _peak_eq = float(np.abs(F_BY[BAR]).max())
    equilibrium = (np.abs(F_BY[BAR] - F_BY[OTHER])
                   / (_peak_eq if _peak_eq > 0 else 1.0))


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
    _cau = f'{r["causality"]:11.3f}' if r['causality_ok'] else f'{"n/a":>11}'
    _ten = f'{"n/a":>9}' if TENSION else f'{r["tensile"]:9.3f}'
    print(f'{name:>11} {Dg:8.2f} {r["peak"]:9.3f} {r["null"]:10.2e} '
          + _cau + ' ' + _ten
          + (f'{r["after"]:11.3f}' if IMPACT else ''))
print('the right-hand columns are fractions of peak |P|. Zero is the '
      'ideal;\nwhat is left is model error, and its SIGN is known -- a contact '
      'that pulls or a\nwave that arrives early is not a measurement, it is the '
      'residual.')
if not NULL_VALID:
    print(f'the free-end column is NOT a check on this run: the {BAR} bar '
          'reaches the free\nend through the interface, and this is not the '
          'shot the calibration measured\nthat path on. Reported, not to be '
          'read as PASS or FAIL.')
else:
    print(f'the free-end null uses the IDENTIFIED distances to the {BAR} bar\'s '
          f'own free\nsurface, {L_FREE[BAR].min():.0f}-{L_FREE[BAR].max():.0f} '
          f'mm; threshold for this case is {NULL_TOL:.1e}.')
_r0 = RES[SETS[0][0]]
_T0 = float(d.get('t0_file', 0.0))
if _r0['echo_in_record']:
    print(f'the echo reaches x = 0 at {_r0["t_echo"] + _T0:.0f} us (source-file '
          'base) -- one round trip\nafter the wave LEFT it, not after the '
          'record started.')
else:
    print(f'the echo-timing search did NOT resolve on this record: P keeps '
          'rising past its\ninitial edge -- likely multiple reflections '
          'loading a real specimen, unlike the single\nclean pulse a '
          'calibration shot gives -- so the search for "where the wave left '
          'x = 0"\nwalked back from that later rise instead. The '
          'reconstruction itself (F = P + M,\nthe free-end null) does not '
          'depend on this; only causality below does, and it is\nreported '
          'n/a rather than computed from a meaningless window.')
if _r0['rise_ok']:
    print(f'its own 10-90 rise there measures {_r0["rise"]*1e3:.0f} us after '
          f'crossing 2L = {2*L:.0f} mm of\nlossy bar, and that is the clearance '
          'held either side.')
else:
    print(f'its edge could NOT be measured: |M| never falls back to 10 % of the '
          'echo peak\nbetween the wavefront and the echo, so the walk-back hit '
          'its bound. The\nclearance is that bound and the causality check is '
          'reported n/a above.')

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

if OTHER is not None:
    print(f'\n--- force equilibrium across the interface, {BAR} vs {OTHER} bar '
          '-------')
    print(f'{OTHER} bar    : c0 = {c0_o:.2f} mm/ms, x = '
          f'[{", ".join(f"{v:.1f}" for v in x_o)}] mm ({len(sig_o)} gauges), '
          'attenuation ' + ('lossless' if ATT_o is None else
                            f'alpha(f) up to {ATT_o[0][-1]:.0f} kHz'))
    print(f'|F_{BAR}-F_{OTHER}|/max|F_{BAR}| : mean {equilibrium[_win].mean():.4e}, '
          f'max {equilibrium[_win].max():.4e}, over {t[_win.start]:.3f}-'
          f'{t[min(_win.stop, N-1)]:.3f} ms')
    print('a genuine joint carries the same force on both sides; what is left '
          'is model\nerror -- most likely a small extra transit time through '
          'the coupler that this\nreconstruction, treating each bar as if it '
          'ran straight to the interface, does\nnot account for.')

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
    if r['causality_ok']:
        print(f'  nothing returns to x = 0 until the free-end echo at '
              f'{r["t_echo"]:.0f} us, and M holds\n    to {r["causality"]:.3f} '
              'of peak before it')
    elif not r['echo_in_record']:
        print('  the causality check DOES NOT APPLY to this record: the '
              'echo-timing search did not\n    resolve (see above) -- '
              'reported n/a, not computed from a meaningless window.')
    else:
        print(f'  the causality check DOES NOT APPLY to this record: the echo '
              f'edge at x = 0 is\n    so broad that it reaches back to the '
              'wavefront, leaving no interval that is\n    both after the '
              'loading and before the echo. Reported n/a, not clamped.')
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

# Same rows as identify_bar_tension.py's figure, minus its matched-filter row
# (there is nothing to time here -- the edges are what a specimen destroys, and
# nothing on this record is being identified from them):
#
#   row 0, per bar   what was measured
#   row 1, per bar   F = P + M at THAT bar's own face
#   row 2, left      force equilibrium across the interface, both bars overlaid
#   row 2, right     the free-end null, at BAR's own far free surface
#
# The bottom row is the pair of checks that need no ground truth, one per
# column, exactly as there. With a single bar identified there is no
# equilibrium to draw and only the null is shown.
#
# Only BAR (the one checks()/free_end() ran on) has the echo/causality/tensile/
# bars-part diagnostics computed at all, so those annotations sit on BAR's
# column, whichever physical bar that happens to be.
PANEL = {BAR: dict(sig=sig, x=x_id, F=None, att=ATT, primary=True)}
if OTHER is not None:
    PANEL[OTHER] = dict(sig=sig_o, x=x_o, F=F_o, att=ATT_o, primary=False)
COLS = [b for b in ('in', 'out') if b in PANEL]
NULL_COL = COLS.index(BAR)
EQ_COLI = next((i for i, b in enumerate(COLS) if b != BAR), None)

# 11 in is the narrowest the suptitle fits in; two columns get 9.5 each, which
# is the width the two-bar layout has always had.
fig, axes = plt.subplots(3, len(COLS), figsize=(max(11.0, 9.5 * len(COLS)), 12),
                         sharex=True, squeeze=False)
fig.patch.set_facecolor(SURFACE)

CTRL = {}                                       # per-bar slider state
for col, bname in enumerate(COLS):
    pnl = PANEL[bname]

    # --- what went in ------------------------------------------------------
    ax0 = axes[0, col]
    for k, s in enumerate(pnl['sig']):
        ax0.plot(tt_f, s * SCALE, lw=.9, color=(BLUE, ORANGE, INK)[k % 3],
                 label=f'gauge {k} at {pnl["x"][k]:.0f} mm (identified)')
    ax0.set_ylabel(f'Gauge signal ({USYM})')
    ax0.set_title(f'What was measured — {len(pnl["sig"])} gauges on the '
                  f'{bname} bar', loc='left', fontsize=11)
    _g = max(np.abs(s_).max() for s_ in pnl['sig']) * SCALE
    ax0.set_ylim(-1.3 * _g, 1.35 * _g)
    ax0.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

    # --- F = P+M at the specimen plane (the face when holder_length = 0) ----
    ax1 = axes[1, col]
    line_F = None
    # The travelling waves themselves, identified positions only -- the tape
    # set is not worth a second pair of traces here, and the point of this
    # panel is F; P and M are context for how F was built.
    _p0, _m0 = PLANE[SETS[0][0]] if pnl['primary'] else PLANE_O
    line_p, = ax1.plot(tt_f, _p0 * SCALE, lw=.8, color=BLUE, alpha=.6,
                       label='$P$')
    line_m, = ax1.plot(tt_f, _m0 * SCALE, lw=.8, color=ORANGE, alpha=.6,
                       label='$M$')
    if pnl['primary']:
        for (name, _), col_c, ls in zip(SETS, (INK, BLUE), ('-', '--')):
            (_line,) = ax1.plot(tt_f, sum(PLANE[name]) * SCALE, color=col_c,
                                lw=1.0, ls=ls,
                                label=f'$F = P + M$, {name} positions')
            if name == SETS[0][0]:
                line_F = _line
    else:
        (line_F,) = ax1.plot(tt_f, F_BY[bname] * SCALE, color=INK, lw=1.0,
                             label='$F = P + M$, identified positions')
    ax1.axhline(0, color=GRID, lw=1.0)
    band = 0.03 * r['amp'] * SCALE
    ax1.axhspan(-band, band, color=BLUE, alpha=.15,
                label='±3 % of peak $|P|$')
    title = (f'F = P + M at the {bname} holder/specimen interface, '
             f'{HOLDER:g} mm past the face' if HOLDER else
             f'F = P + M at the {bname}put-bar/specimen interface')
    if pnl['primary']:
        # The echo instant and the clearance held either side of it: the
        # windows every check above is scored over. They belong on the panel
        # the checks are about, which is BAR's own force.
        ax1.axvline(r['t_echo'] + T0_FIG, color=MUTED, lw=1.1, ls='--')
        # Labelled to the LEFT of its own line: the echo lands near the end of
        # the record by construction, so a label growing rightward runs off.
        ax1.annotate('free-end echo arrives, $2L/c$ after the wave left  ',
                     (r['t_echo'] + T0_FIG, 0), fontsize=9, color=MUTED,
                     va='bottom', ha='right')
        for _b in (r['t_echo'] + T0_FIG - r['rise'] * 1e3,
                   r['t_echo'] + T0_FIG + r['rise'] * 1e3):
            ax1.axvline(_b, color=GRID, lw=1.0, ls=':')
        if IMPACT:
            ax1.axvline(r['t_open'] + T0_FIG, color=ORANGE, lw=1.1, ls='--')
            ax1.annotate('  bars part', (r['t_open'] + T0_FIG,
                                         r['peak'] * SCALE * .6),
                         fontsize=9, color=MUTED)
        title += ('' if TENSION else
                  f' — cannot go negative: residual {r["tensile"]:.3f} '
                  'of peak')
        if r['causality_ok']:
            title += f', causality {r["causality"]:.3f}'
    title += '' if pnl['att'] is not None else '  (LOSSLESS)'
    ax1.set_ylabel(f'Interface force ({UNITS})')
    ax1.set_title(title, loc='left', fontsize=10)
    ax1.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')
    CTRL[bname] = dict(ax=ax1, line_p=line_p, line_m=line_m, line_F=line_F,
                       title_base=title, x=pnl['x'])

# --- bottom left: force equilibrium across the interface -------------------
# Two INDEPENDENT solves of one quantity -- separate gauges, separate solves,
# sharing only the bar's identified constants -- so where they part company is
# the honest error bar on the whole reconstruction. Identical panel to
# identify_bar_tension.py's, and neither curve is shifted: the two faces are a
# specimen apart.
#
# EQ holds the handles the sliders below need to keep this panel in sync: it
# always carries BOTH bars, and each bar's slider moves that bar's own
# reconstruction, so this panel follows whichever one last moved.
EQ = None
if EQ_COLI is not None:
    axeq = axes[2, EQ_COLI]
    # Coloured and differenced by BAR NAME, not by which one --bar happened to
    # make primary, so the panel reads the same way whichever was picked.
    _eq_lines = {}
    for _b, _c in (('in', BLUE), ('out', ORANGE)):
        (_eq_lines[_b],) = axeq.plot(
            tt_f, F_BY[_b] * SCALE, lw=1.0, color=_c,
            label=(f'$F_{{{_b}}}$ at the {_b} holder / specimen interface'
                   if HOLDER else
                   f'$F_{{{_b}}}$ at the {_b}put-bar / specimen face'))
    (_eq_diff,) = axeq.plot(tt_f, (F_BY['in'] - F_BY['out']) * SCALE,
                            color=INK, lw=1.1,
                            label='$F_{in} - F_{out}$ (should be 0)')
    axeq.axhline(0, color=GRID, lw=1.0)
    axeq.axvspan(tt_f[_win.start], tt_f[min(_win.stop, N - 1)], color=GRID,
                 alpha=.35, label='mean/max window')
    axeq.set_xlabel('Time (us)')
    axeq.set_ylabel(f'Interface force ({UNITS})')

    def _eq_title(mean, max_, sfx='', shifted=', unshifted'):
        return ('Force equilibrium across the specimen — two independent '
               f'solves of one force:\nmean {mean:.2e}, max {max_:.2e} of '
               f'peak $|F_{{{BAR}}}|${sfx}  (the two faces are a specimen '
               f'apart{shifted})')

    axeq.set_title(_eq_title(equilibrium[_win].mean(), equilibrium[_win].max()),
                   loc='left', fontsize=10)
    axeq.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')
    EQ = dict(ax=axeq, line_in=_eq_lines['in'], line_out=_eq_lines['out'],
              line_diff=_eq_diff, title_fn=_eq_title)

# --- reconstruction-location sliders, one per bar ----------------------------
# Same solve as the P/M/F traces above, just evaluated off the specimen plane
# instead of AT it. The slider's 0 IS that plane -- the holder/specimen
# interface, holder_length past the bar face -- so it reads "how far from
# where the specimen is", whatever holds it. `separate_field` is the field version of `separate`'s
# normal-equations solve -- exact for a lossless bar, and consistent with
# `separate` when attenuation/dispersion are given -- so x = 0 here
# reproduces the static P/M/F lines exactly, and the slider is a continuous
# extension of them rather than a different calculation.
#
# -200 to holder_length + 50 mm: positive is INTO the bar (away from the
# specimen, per separate_field's own convention), and +holder_length is the bar
# face itself; negative walks on past the specimen plane, into and through
# where the specimen sits. Everything short of the face is EXTRAPOLATION of
# this bar's model -- fair across a holder of bar material, not across the
# specimen, where the bar's c0/attenuation/dispersion no longer describe the
# material, so a large negative reading is a "what would this bar's model
# predict there", not a measurement.
#
# Each bar has its own slider, and they are independent: F_CUR holds each
# bar's force at its CURRENT offset, so the equilibrium panel compares the two
# planes the sliders actually point at, not one moved plane against a fixed one.
from matplotlib.widgets import Slider

F_CUR = dict(F_BY)
DELTA = {b: 0.0 for b in CTRL}


def _update_eq():
    if EQ is None:
        return
    diff = F_CUR['in'] - F_CUR['out']
    EQ['line_in'].set_ydata(F_CUR['in'] * SCALE)
    EQ['line_out'].set_ydata(F_CUR['out'] * SCALE)
    EQ['line_diff'].set_ydata(diff * SCALE)
    eq_frac = np.abs(diff) / (_peak_eq if _peak_eq > 0 else 1.0)
    moved = [f'{b} {DELTA[b]:+.0f} mm' for b in ('in', 'out') if DELTA[b]]
    eq_sfx = ((', ' + ', '.join(moved) + ' off the specimen interface')
              if moved else '')
    EQ['ax'].set_title(EQ['title_fn'](eq_frac[_win].mean(), eq_frac[_win].max(),
                                      eq_sfx, '' if moved else ', unshifted'),
                       loc='left', fontsize=10)


def _make_offset_handler(bname):
    c = CTRL[bname]

    def _on_offset(val):
        delta = float(val)
        p1, m1 = at_plane(bname, c['x'], delta)
        F1 = p1 + m1
        c['line_p'].set_ydata(p1 * SCALE)
        c['line_m'].set_ydata(m1 * SCALE)
        c['line_F'].set_ydata(F1 * SCALE)
        where = 'into the bar' if delta > 0 else 'towards/through the specimen'
        suffix = (f'  [{delta:+.0f} mm off the specimen interface, {where}]'
                  if delta else '')
        c['ax'].set_title(c['title_base'] + suffix, loc='left', fontsize=10)
        F_CUR[bname], DELTA[bname] = F1, delta
        _update_eq()
        fig.canvas.draw_idle()

    return _on_offset


_SLIDERS = {}                                   # keep references alive
for _b, _c in CTRL.items():
    _sax = _c['ax'].inset_axes((0.60, 0.88, 0.38, 0.07))
    _sax.set_facecolor(SURFACE)
    _sl = Slider(_sax, 'x off specimen (mm)', -200.0, HOLDER + 50.0,
                 valinit=0.0,
                 valstep=1.0, color=BLUE)
    _sl.label.set_fontsize(8); _sl.label.set_color(MUTED)
    _sl.valtext.set_fontsize(8); _sl.valtext.set_color(MUTED)
    _sl.on_changed(_make_offset_handler(_b))
    _SLIDERS[_b] = _sl

# --- bottom right: the free-end null ---------------------------------------
# The same record reconstructed at BAR's own far FREE surface instead of at the
# interface, where the boundary condition demands zero stress. It consumes
# nothing but this record and the identified numbers, which is what makes it
# the check that survives contact with a rig -- and it is the one place a
# calibration carried over from another shot can be caught being wrong.
axnull = axes[2, NULL_COL]
_nl = NULL[SETS[0][0]]
axnull.plot(tt_f, _nl['p'] * SCALE, lw=.9, color=BLUE, label='$P$')
axnull.plot(tt_f, _nl['m'] * SCALE, lw=.9, color=ORANGE, label='$M$')
axnull.plot(tt_f, _nl['tot'] * SCALE, color=INK, lw=1.1,
            label='$P + M$ (should be 0)')
axnull.axhline(0, color=GRID, lw=1.0)
axnull.axvspan(tt_f[_nl['w'].start], tt_f[min(_nl['w'].stop, N - 1)],
               color=GRID, alpha=.35, label='rms/max window')
axnull.set_xlabel('Time (us)')
axnull.set_ylabel(f'Stress at free end ({USYM})')
_verdict = (f'rms {_nl["rms"]:.2e} vs tol {NULL_TOL:.1e} -> '
            f'{"PASS" if _nl["rms"] <= NULL_TOL else "FAIL"}' if NULL_VALID else
            f'rms {_nl["rms"]:.2e} — NOT A CHECK: the {BAR} bar reaches the '
            'free end through the\ninterface, and this is not the shot the '
            'calibration measured that path on')
axnull.set_title(
    f'Free-end null: stress at the {BAR} bar\'s free surface, '
    f'{_nl["L_free"].min():.0f}-{_nl["L_free"].max():.0f} mm away —\n'
    + _verdict, loc='left', fontsize=10)
axnull.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

axes[0, 0].set_xlim(tt_f[0], tt_f[int(0.95 * N)])

if OTHER is not None:
    _Din = abs(PANEL['in']['x'][1] - PANEL['in']['x'][0])
    _Dout = abs(PANEL['out']['x'][1] - PANEL['out']['x'][0])
    _title = (f'Force at each bar\'s own interface — in: {_Din:.0f} mm gauge '
              f'spacing, out: {_Dout:.0f} mm'
              + ('' if SELF
                 else f'\ncalibrated on {BARS_DIR} — nothing is '
                      'identified from this record'))
else:
    _title = (f'Force at the {IFACE}, reconstructed from two gauges '
              f'{abs(x_id[1]-x_id[0]):.0f} mm apart on the {BAR} bar'
              + ('' if SELF
                 else f'\ncalibrated on {BARS_DIR} — nothing is '
                      'identified from this record'))

for ax in axes.flat:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)

fig.suptitle(_title, x=.006, ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, .975))
_stem = cases.output(cfg, 'interface_force')
FIG = (f'{_stem}.png' if ATT is not None else f'{_stem}_lossless.png')
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'\nwrote {cases.rel(FIG)}')

# Three columns and no more: time, and the force at each bar's own face. P, M
# and the equilibrium residual used to ride along here; they are derived from
# these (F = P + M, and the residual is their difference over a peak) or shown
# in the figure, and a reduction file that has to be explained is worse than
# one that does not. Only the bars actually identified get a column.
DAT = (f'{_stem}.dat' if ATT is not None else f'{_stem}_lossless.dat')
T0 = float(d.get('t0_file', 0.0))
_cols = [tt + T0] + [F_BY[b] for b in DAT_BARS]
_header = 'time[us]  ' + '  '.join(
    f'F_{b}[{UNITS}]' for b in DAT_BARS)
_geom = '\n'.join(
    f'F_{b}: {b} bar, c0={float(ID[f"c_{b}"]):.3f} mm/ms, '
    f'x={[round(float(v), 2) for v in np.asarray(ID[f"x_{b}"], float)]} mm, '
    + (f'reconstructed at the holder/specimen interface, x=-{HOLDER:g} mm'
       if HOLDER else 'reconstructed at ITS OWN face (x=0)')
    for b in DAT_BARS)
np.savetxt(DAT, np.column_stack(_cols),
           header=_header + '\n'
                  f'time is the SOURCE FILE\'s own base (analysis t=0 sits at '
                  f'{T0:.1f} us there)\n'
                  + _geom + f'\neta={eta:g}, x=0 is the {IFACE}')
print(f'wrote {cases.rel(DAT)}: {", ".join(["time"] + [f"F_{b}" for b in DAT_BARS])}')

plotting.show_unless(HEADLESS)
