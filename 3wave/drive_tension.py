"""
Run the TENSION case (SHTB with a POM striker) and write dump.npz.

All parameters live in config.toml under [tension] -- materials, geometry,
gauge locations, numerics. Nothing is set here.

    python3 drive_tension.py

Its sibling drive_compression.py runs the compression bar and writes the SAME filename, so
the last one run is what the analysis scripts see. The one thing downstream must
know is the sign convention, and that travels in the dump: this data is TENSION
POSITIVE.
"""
import config
import simulate_tension
from dump import write_dump

cfg = config.load('tension')
sim = simulate_tension.SimulateSHTB(cfg)
write_dump(sim, cfg)
