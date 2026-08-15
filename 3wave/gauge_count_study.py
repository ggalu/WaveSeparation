"""
How many gauges per bar are actually needed?

Compares gauge layouts against the simulator's own specimen measurement. The
layouts are given as INDEX subsets into the gauge list configured in
config.toml, so this adapts to whatever gauges the dump happens to carry.

    python3 drive.py
    python3 gauge_count_study.py
"""
import numpy as np

from dump import load_dump
from wave_separation import (separate, backpropagate, bar_interface,
                             specimen_response, single_wave_window)

d = load_dump()
E, A, c0, dt, t = d['E'], d['A'], d['c0'], d['dt'], d['t']
L_SPEC, A_SPEC = d['L_specimen'], d['A_specimen']
LOADING, ETA = d['loading'], d['eta']
L_IN, L_OUT = d['L_free_in'], d['L_free_out']
N_G = d['eps_in'].shape[0]

# The single-gauge rows use backpropagate, which assumes only ONE wave is
# present at the gauge. That holds on a direct-impact bar before the far-end
# reflection returns; it does NOT hold on the SHTB, where the specimen is
# bonded and a reflected wave exists from the start. Running them on a tension
# dump used to print zeros with no warning.
SINGLE_GAUGE_OK = LOADING == 'compression'


def waves(bar, idx):
    """Separate (or backpropagate) using the gauges at positions `idx`."""
    sig = d[f'eps_{bar}'][list(idx)]
    pos = d[f'pos_{bar}'][list(idx)]
    if len(idx) == 1:
        # direct impact: the only wave present travels AWAY from the specimen
        return backpropagate(t, sig[0], pos[0], c0, eta=ETA, direction='plus')
    return separate(t, list(sig), list(pos), c0=c0, eta=ETA)


_sgn = -1.0 if LOADING == 'compression' else 1.0
sig_true, eps_true = _sgn * d['spec_stress'], _sgn * d['spec_strain']
_live = np.abs(sig_true) > 0.02 * np.abs(sig_true).max()
_i0 = int(np.argmax(_live))
_i1 = int(np.argmax(np.abs(eps_true))) + int(0.05 / dt)
win = (t >= t[_i0]) & (t <= t[min(_i1, len(t) - 1)])
rel = lambda a, b: np.linalg.norm(a[win] - b[win]) / np.linalg.norm(b[win])

# Layouts as (input gauge indices, output gauge indices), nearest gauge first.
# Slices collapse onto each other when the dump carries fewer gauges than the
# layouts assume, so de-duplicate rather than print the same row twice.
ALL = tuple(range(N_G))
LAYOUTS = [(ALL, ALL), (ALL[:2], ALL), (ALL[:1], ALL),
           (ALL[:1], ALL[:2]), (ALL[:1], ALL[:1]), (ALL[-1:], ALL[:2])]
LAYOUTS = list(dict.fromkeys(LAYOUTS))

print(f'loading = {LOADING}, eta = {ETA} /ms, '
      f'gauges at {[f"{p:.1f}" for p in d["pos_in"]]} mm')
print(f'single-wave window ends at: input bar '
      f'{single_wave_window(L_IN, d["pos_in"][0], c0):.3f} ms, '
      f'output bar {single_wave_window(L_OUT, d["pos_out"][0], c0):.3f} ms')
print(f'loading event ends at ~{2*L_IN/c0:.3f} ms  '
      f'(peak strain {eps_true.max():.3f})')
if not SINGLE_GAUGE_OK:
    print('\nNOTE: this is a TENSION dump. Single-gauge rows are skipped -- '
          '\n      backpropagate assumes one wave, which is false here.')
print()
print(f'{"input":>10s} {"output":>10s}   {"stress err":>11s} '
      f'{"strain err":>11s}  {"peak strain":>11s}')

for din, dout in LAYOUTS:
    note = ''
    if (len(din) == 1 or len(dout) == 1) and not SINGLE_GAUGE_OK:
        print(f'{len(din):>7d} ga {len(dout):>7d} ga   '
              f'{"skipped":>11s} {"skipped":>11s}  {"":>11s}'
              '   <- needs a direct-impact bar')
        continue
    if din == ALL[-1:]:
        note = f'   <- gauge at {d["pos_in"][-1]:.0f} mm'
    p_in, m_in = waves('in', din)
    p_out, m_out = waves('out', dout)
    F_in, v_in = bar_interface(p_in, m_in, E, A, c0, outward=-1, v0=d['v0_in'])
    F_out, v_out = bar_interface(p_out, m_out, E, A, c0, outward=+1, v0=d['v0_out'])
    r = specimen_response(t, F_in, v_in, F_out, v_out, L_SPEC, A_SPEC,
                          loading=LOADING)
    print(f'{len(din):>7d} ga {len(dout):>7d} ga   {rel(r["stress"], sig_true):11.3e} '
          f'{rel(r["strain"], eps_true):11.3e}  {r["strain"].max():11.4f}' + note)
