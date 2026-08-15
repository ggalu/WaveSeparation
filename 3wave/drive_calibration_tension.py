"""
Run the CALIBRATION case -- the two bars bolted straight together, no specimen --
and write dump.npz.

All parameters live in config.toml under [calibration_tension]. It is the SHTB model
with the specimen replaced by the rig's 150 mm threaded coupler, in bar stock at
bar diameter, so the assembly is one uniform bar and the joint reflects nothing.
The coupler's LENGTH must match the real one; its material must match the bars,
or the identification is biased silently -- see README.md.

    python3 drive_calibration_tension.py
    python3 identify_bar_tension.py

identify_bar_tension.py then recovers the gauge positions, their spacing and c0 from the
echo train alone, and never reads the gauge list this run was configured with.

Writes the SAME dump.npz as drive_compression.py and drive_tension.py, so the last driver
run is what the analysis scripts see.
"""
import config
import simulate_tension
from dump import write_dump

cfg = config.load('calibration_tension')
sim = simulate_tension.SimulateSHTB(cfg)
write_dump(sim, cfg)
