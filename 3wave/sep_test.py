"""
Independent cross-check of the separation, and an eta sweep.

This runs wsep.py -- the frozen literal transcription of MATLAB's
wave_separation3 -- rather than wave_separation.py, deliberately: it keeps the
check on the library independent instead of circular. See the header of wsep.py.

The measure is the reconstructed interface force against the force the simulator
actually carried in the bar element at each specimen face, which the dump stores
as ground truth.

    python3 drive_tension.py    (or drive.py)
    python3 sep_test.py
"""
import numpy as np

from dump import load_dump
from wsep import wave_separation

d = load_dump()
E, A, c0, dt, t = d['E'], d['A'], d['c0'], d['dt'], d['t']
NFFT = 1 << 18
fmax = 1.0 / dt

# The dump already resolved the gauges; wsep takes them positionally, three at
# a time, in MATLAB's argument order.
if d['eps_in'].shape[0] != 3:
    raise SystemExit(f"wsep.py is hardwired to three gauges per bar; the dump "
                     f"has {d['eps_in'].shape[0]}. Adjust config.toml, or use "
                     f"wave_separation.separate, which takes any number.")

# skip the very first instants, and stop before the record ends
win = (t > 0.02) & (t < 0.98 * t[-1])

# The window is undone with exp(+eta*t) on the way out, so eta * record length
# much above ~30 is numerically dead. wsep.py has no guard on purpose -- MATLAB
# has none either -- and returns a huge number rather than complaining;
# wave_separation.separate raises instead. Flag those rows so the blow-up is not
# mistaken for a measurement.
T = t[-1] - t[0]
ETA_SWEEP = (0.5, 2.0, 5.0, 20.0, 50.0)

print(f"loading = {d['loading']}, record length {T:.3f} ms "
      f"(eta above {30/T:.1f} /ms overflows)")
print(f"{'bar':5s} {'eta':>7s}  {'rel L2 err vs true interface force':>34s}"
      "   peak-normalised")
for bar, sig, pos, truth in (
        ('in', d['eps_in'], d['pos_in'], d['force_iface_in']),
        ('out', d['eps_out'], d['pos_out'], d['force_iface_out'])):
    for eta in ETA_SWEEP:
        a0, b0 = wave_separation(t, *sig, *pos, c0, eta, NFFT, fmax)
        rec = E * A * (a0 + b0)
        err = np.linalg.norm(rec[win] - truth[win]) / np.linalg.norm(truth[win])
        peak = np.abs(rec[win] - truth[win]).max() / np.abs(truth[win]).max()
        flag = '   <- exp(+eta t) overflowed' if eta * T > 30 else ''
        print(f'{bar:5s} {eta:7.1f}  {err:34.4e}   {peak:.4e}{flag}')
    print(f"      exact distances used: {[f'{x:.1f}' for x in pos]}")
