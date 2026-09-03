"""
Interactive, config-only per-bar wave separation -- the simple counterpart to
identify_bar_tension.py.

    python3 identify_bar_tension.py --experiment experiment_tension_bar_2
    python3 identify_bar_tension_manual.py --experiment experiment_tension_bar_2

identify_bar_tension.py IDENTIFIES c0 and the gauge positions from the echo
train; this script does neither. It takes both as given -- c0 from
bar_identified.npz (the file the other script just wrote) and the gauge
positions from [<case>.gauges] in config.toml, i.e. the tape measurements --
and, for each bar separately, feeds its own two gauges straight into
`separate` to reconstruct F = P + M at that bar's own interface. No edge
timing, no striker-pulse or free-end-echo search, no attenuation: just the
two-gauge solve, exactly as the tape says. Pochhammer-Chree dispersion --
the c_p(f) table identify_bar_tension.py fit from the SAME two gauges, also
in bar_identified.npz -- is applied by default and can be switched off with
the "dispersion" checkbox to see what it is actually doing to F.

A "method" radio button switches the reconstruction itself between the
frequency-domain solve above (`wave_separation.separate`) and a time-domain
one (`wave_separation.separate_time_domain`) that shifts the two gauge
records directly instead of solving in Fourier space -- exact for a lossless,
non-dispersive bar, and nothing else: it has no eta and no dispersion
correction, so the dispersion checkbox is inert while it is selected. It also
needs at least 2*tau of quiescent record before the first arrival (tau being
the transit time between that bar's two gauges) to seed its recursion; where
[<case>.trim].lead is too short for that it raises rather than guessing.

The point is to let a person SEE what the tape positions do to the
reconstruction and nudge them by hand. Every gauge gets a slider, in 1 mm
steps; dragging one instantly re-solves and redraws that gauge's bar. Closing
the plot window writes whatever positions the sliders are left at back into
[<case>] in config.toml, in place, so the next run of either script starts
from them. --headless renders once from the tape values and exits without
touching config.toml, since there is then nothing to hand-adjust.
"""
import argparse
import re

import numpy as np

import plotting

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('--experiment', metavar='CASE',
                 default='experiment_tension_bar_2',
                 help='config case to read (default: experiment_tension_bar_2). '
                      'Must have [<case>.input_bar] and [<case>.output_bar] -- '
                      'two instrumented bars.')
_ap.add_argument('--window', type=float, default=50.0, metavar='MM',
                 help='slider half-range around each tape position [mm] '
                      '(default 50)')
HEADLESS, ARGS = plotting.init(parser=_ap)

import config
from experiment import load_experiment
from wave_separation import separate, separate_time_domain

CASE = ARGS.experiment
if CASE not in config.EXPERIMENT_CASES:
    raise SystemExit(f'{CASE!r} is not a measured case; expected one of '
                     f'{config.EXPERIMENT_CASES}')

cfg = config.load(CASE)
if 'input_bar' not in cfg or 'output_bar' not in cfg:
    raise SystemExit(f'[{CASE}] has no [.input_bar]/[.output_bar] pair -- this '
                     'script needs two instrumented bars, one gauge pair each.')

d = load_experiment(CASE)
t, dt, N, eta = d['t'], d['dt'], d['N'], d['eta']
UNITS = d.get('units', 'strain')
SCALE, USYM = (1.0, UNITS) if UNITS != 'strain' else (1e6, 'ustrain')

if d['eps_in'].shape[0] < 2 or d['eps_out'].shape[0] < 2:
    raise SystemExit(f'[{CASE}]: needs >= 2 gauges on EACH bar; got '
                     f'{d["eps_in"].shape[0]} in, {d["eps_out"].shape[0]} out')

# --------------------------------------------------------------------------
# gauge names, in the order config.toml's flat "gauges" list uses -- the same
# split experiment.py already applied to build eps_in/eps_out/pos_in/pos_out,
# recovered here only so positions can be written BACK to that same list.
# --------------------------------------------------------------------------
_cols = dict(cfg['columns'])
_cols.pop('time')
NAMES = list(_cols)
NAMES_IN = [nm for nm in NAMES if nm.startswith('in-')]
NAMES_OUT = [nm for nm in NAMES if nm.startswith('out-')]

L_IN = float(cfg['input_bar']['L_input'])
L_OUT = float(cfg['output_bar']['L_output'])

# An amplifier rail is not signal; separate() is a global transform, so the
# record fed to it must end before the earliest clip onset across the gauges
# it uses. Same idea as identify_bar_tension.py, without the edge search.
_CLIP_MARGIN = 0.05   # ms
_clip_in = d['clip_onset'][:d['eps_in'].shape[0]]
_clip_out = d['clip_onset'][d['eps_in'].shape[0]:]


def _hi(clips):
    hi = N
    for co in clips:
        if not np.isnan(co):
            hi = min(hi, max(1, int((co - _CLIP_MARGIN) / dt)))
    return hi


HI_IN, HI_OUT = _hi(_clip_in), _hi(_clip_out)
t_in, t_out = t[:HI_IN], t[:HI_OUT]
sig_in = d['eps_in'][:, :HI_IN]
sig_out = d['eps_out'][:, :HI_OUT]

# c0 is a property of the bar, not of one shot -- borrowed from whatever
# identify_bar_tension.py last identified, since this script identifies
# nothing of its own.
IDENT_FILE = 'bar_identified.npz'
try:
    ident = np.load(IDENT_FILE)
except FileNotFoundError:
    raise SystemExit(f'{IDENT_FILE} not found. Run\n\n'
                     f'    python3 identify_bar_tension.py --experiment {CASE}\n\n'
                     'first, to identify c0 -- this script only handles gauge '
                     'positions, it does not identify a wave speed.')
if 'c_in' not in ident.files or 'c_out' not in ident.files:
    raise SystemExit(f'{IDENT_FILE} has no c_in/c_out -- re-run '
                     'identify_bar_tension.py on a two-bar case first.')
if str(ident['case']) != CASE:
    print(f'note: {IDENT_FILE} was identified from case '
          f'{str(ident["case"])!r}, using its c0 anyway -- c0 is a property '
          'of the bar, not of one shot.')
C_IN, C_OUT = float(ident['c_in']), float(ident['c_out'])

# Dispersion table per bar: (freq [kHz], c_p/c0), or None where
# identify_bar_tension.py could not measure one (e.g. a bar whose only gauge
# pair runs opposite to propagation, and that borrowed nothing either).
# `separate`'s `dispersion` argument takes exactly this (freq, ratio) shape.
DISP = {}
for _bar in ('in', 'out'):
    _fk, _ck = f'dispersion_f_{_bar}', f'dispersion_{_bar}'
    DISP[_bar] = ((ident[_fk], ident[_ck])
                 if _fk in ident.files and _ck in ident.files else None)
    if DISP[_bar] is None:
        print(f'note: no dispersion table for the {_bar} bar in {IDENT_FILE} -- '
              f'that bar stays lossless/non-dispersive regardless of the toggle')
DISPERSION_ON = True   # start with the identified correction applied
METHOD = 'freq'         # 'freq' (separate, FFT) or 'time' (separate_time_domain)

pos0 = np.asarray(cfg['gauges'], float)
is_in = np.array([nm.startswith('in-') for nm in NAMES])
is_out = np.array([nm.startswith('out-') for nm in NAMES])
pos = pos0.copy()   # mutable, in NAMES order -- what the sliders drive

print(f'{CASE}: c0 in={C_IN:.3f}, out={C_OUT:.3f} mm/ms (from {IDENT_FILE})')
print(f'tape positions: in={pos[is_in]}, out={pos[is_out]} mm')


def solve(bar):
    """F = P + M at the given bar's own interface, from its own two gauges."""
    if bar == 'in':
        x, sig, t_b, c0 = pos[is_in], sig_in, t_in, C_IN
    else:
        x, sig, t_b, c0 = pos[is_out], sig_out, t_out, C_OUT
    if METHOD == 'time':
        p, m = separate_time_domain(t_b, list(sig), x, c0=c0)
    else:
        disp = DISP[bar] if DISPERSION_ON else None
        p, m = separate(t_b, list(sig), x, c0=c0, eta=eta, dispersion=disp)
    return p + m


# --------------------------------------------------------------------------
# figure -- two columns (in, out), two rows (raw signal, F = P + M), and one
# slider per gauge below. No edge timing here, so no matched-filter row and
# no striker-pulse / free-end-echo markers -- the two things this script
# deliberately does not identify.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d8d7d3'
SURFACE = '#fcfcfb'

COLS = (('in', NAMES_IN, t_in, sig_in), ('out', NAMES_OUT, t_out, sig_out))

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex='col', squeeze=False)
fig.patch.set_facecolor(SURFACE)

sig_lines = {}      # (bar, j) -> Line2D, row 0
f_lines = {}         # bar -> Line2D, row 1
f_titles = {}        # bar -> Axes, for the position readout in the title

for col, (bar, names_b, t_b, sig_b) in enumerate(COLS):
    ax0, ax1 = axes[0, col], axes[1, col]
    for j, nm in enumerate(names_b):
        ln, = ax0.plot(t_b, sig_b[j] * SCALE, lw=.9, color=(BLUE, ORANGE)[j % 2],
                       label=nm)
        sig_lines[(bar, j)] = ln
    ax0.set_ylabel(f'Signal ({USYM})')
    ax0.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')

    F = solve(bar)
    ln, = ax1.plot(t_b, F * SCALE, color=INK, lw=.9, label='F = P + M')
    f_lines[bar] = ln
    ax1.axhline(0, color=GRID, lw=1.0)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel(f'Interface force ({USYM})')
    ax1.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower left')
    f_titles[bar] = ax1

axes[0, 0].set_title('input bar', loc='left', fontsize=11, color=INK)
axes[0, 1].set_title('output bar', loc='left', fontsize=11, color=INK)


def _bar_title(bar):
    names_b = NAMES_IN if bar == 'in' else NAMES_OUT
    vals = pos[is_in] if bar == 'in' else pos[is_out]
    tag = ', '.join(f'{nm}={v:.0f}' for nm, v in zip(names_b, vals))
    if METHOD == 'time':
        method_tag = 'time domain'
    elif not DISPERSION_ON:
        method_tag = 'freq, disp OFF'
    elif DISP[bar] is not None:
        method_tag = 'freq, disp ON'
    else:
        method_tag = 'freq, no disp'
    f_titles[bar].set_title(f'{tag} mm  --  {method_tag}',
                            loc='left', fontsize=9, color=INK)


for bar in ('in', 'out'):
    _bar_title(bar)

for ax in axes.flat:
    ax.set_facecolor(SURFACE); ax.grid(True, color=GRID, lw=.7, alpha=.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

fig.suptitle('Manual per-bar wave separation -- tape positions only, no '
            'identification', x=.006, ha='left', fontsize=13, color=INK)

# --------------------------------------------------------------------------
# sliders -- one per gauge, in NAMES order, 1 mm steps around the tape value
# --------------------------------------------------------------------------
n_sliders = len(NAMES)
fig.subplots_adjust(bottom=0.10 + 0.045 * n_sliders, top=0.90, hspace=0.28)

sliders = {}
for i, nm in enumerate(NAMES):
    L = L_IN if nm.startswith('in-') else L_OUT
    v0 = pos0[i]
    vlo = max(1.0, v0 - ARGS.window)
    vhi = min(L - 1.0, v0 + ARGS.window)
    sax = fig.add_axes((0.30, 0.03 + 0.045 * (n_sliders - 1 - i), 0.45, 0.025))
    sax.set_facecolor(SURFACE)
    sl = Slider(sax, f'{nm}  [mm]', vlo, vhi, valinit=v0, valstep=1.0,
               color=(BLUE if nm.startswith('in-') else ORANGE))
    sl.label.set_color(MUTED); sl.label.set_fontsize(9)
    sl.valtext.set_color(MUTED); sl.valtext.set_fontsize(9)
    sliders[nm] = sl


def _redraw(bar):
    F = solve(bar)
    f_lines[bar].set_ydata(F * SCALE)
    ax = f_lines[bar].axes
    ax.relim(); ax.autoscale_view(scalex=False)
    _bar_title(bar)


def _on_change(_val, name=None):
    k = NAMES.index(name)
    pos[k] = sliders[name].val
    _redraw('in' if name.startswith('in-') else 'out')
    fig.canvas.draw_idle()


for nm, sl in sliders.items():
    sl.on_changed(lambda v, name=nm: _on_change(v, name))

# --------------------------------------------------------------------------
# method selector -- frequency domain (separate, FFT) vs time domain
# (separate_time_domain, pure shift) -- and, only meaningful in frequency
# mode, the dispersion toggle from before. Both act on both bars at once, so
# the difference is visible on the same figure without re-running.
# --------------------------------------------------------------------------
rax = fig.add_axes((0.62, 0.895, 0.18, 0.075))
rax.set_facecolor(SURFACE)
for sp in rax.spines.values(): sp.set_color(GRID)
radio = RadioButtons(rax, ['frequency domain', 'time domain'], active=0)
for txt in radio.labels:
    txt.set_color(MUTED); txt.set_fontsize(9)

cax = fig.add_axes((0.82, 0.905, 0.16, 0.055))
cax.set_facecolor(SURFACE)
for sp in cax.spines.values(): sp.set_color(GRID)
check = CheckButtons(cax, ['dispersion c_p(f)'], [DISPERSION_ON])
for txt in check.labels:
    txt.set_color(MUTED); txt.set_fontsize(9)


def _sync_dispersion_label():
    # Inert while time-domain is selected -- separate_time_domain() has no
    # dispersion correction at all, so grey the label rather than let it sit
    # there implying it still does something.
    check.labels[0].set_color(GRID if METHOD == 'time' else MUTED)


def _on_toggle(_label):
    global DISPERSION_ON
    DISPERSION_ON = not DISPERSION_ON
    _redraw('in')
    _redraw('out')
    fig.canvas.draw_idle()


def _on_method(label):
    global METHOD
    METHOD = 'time' if label == 'time domain' else 'freq'
    _sync_dispersion_label()
    _redraw('in')
    _redraw('out')
    fig.canvas.draw_idle()


check.on_clicked(_on_toggle)
radio.on_clicked(_on_method)
_sync_dispersion_label()

FIG = f'bar_manual_{CASE}.png'
fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
print(f'wrote {FIG}')


# --------------------------------------------------------------------------
# save adjusted positions back to config.toml on close
# --------------------------------------------------------------------------
def _save_positions(path, case, names, values):
    """
    Rewrite `gauges = [...]` inside [<case>] in place, leaving every other
    line -- including its own comments -- untouched. Targeted text surgery
    rather than a round-trip through tomllib, which carries no comments and
    would silently drop them.
    """
    with open(path) as fh:
        lines = fh.read().split('\n')

    sect_re = re.compile(rf'^\[{re.escape(case)}\]\s*$')
    start = next((i for i, ln in enumerate(lines) if sect_re.match(ln)), None)
    if start is None:
        raise SystemExit(f'could not find [{case}] in {path}; not saving')
    end = next((i for i in range(start + 1, len(lines))
               if lines[i].startswith('[')), len(lines))

    gauges_re = re.compile(r'^(gauges\s*=\s*)\[[^\]]*\](.*)$')
    for i in range(start, end):
        m = gauges_re.match(lines[i])
        if m:
            new_list = ', '.join(f'{v:.2f}' for v in values)
            lines[i] = f'{m.group(1)}[{new_list}]{m.group(2)}'
            break
    else:
        raise SystemExit(f'could not find "gauges = [...]" inside [{case}]; '
                         'not saving')

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'wrote {len(values)} gauge position(s) for [{case}] to {path}: '
          + ', '.join(f'{nm}={v:.2f}' for nm, v in zip(names, values)))


def _on_close(_event):
    if np.allclose(pos, pos0):
        print('gauge positions unchanged; config.toml left as is')
        return
    fig.savefig(FIG, dpi=140, facecolor=fig.get_facecolor())
    print(f'wrote {FIG}')
    _save_positions(config.DEFAULT_PATH, CASE, NAMES, pos)


if not HEADLESS:
    fig.canvas.mpl_connect('close_event', _on_close)

plotting.show_unless(HEADLESS)
