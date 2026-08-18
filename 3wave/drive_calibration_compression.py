"""
Run the COMPRESSION calibration shot and write dump.npz.

The two bars are struck face to face with NO specimen between them, which is
what [calibration_compression] in config.toml says -- it is the same
simulate_compression.py, given a different case. Analysed by
identify_bar_compression.py, which recovers each bar's gauge positions, spacing
and wave speed from the echo train alone.

    python3 drive_calibration_compression.py
    python3 identify_bar_compression.py

Its sibling drive_calibration_tension.py does the same for the SHTB, where the
bars must be bolted together through a coupler. Here they simply touch.
"""
import config
import simulate_compression
from dump import write_dump

cfg = config.load('calibration_compression')
sim = simulate_compression.SimulateDirectImpact(cfg)
write_dump(sim, cfg)
