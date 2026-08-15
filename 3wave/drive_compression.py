"""
Run the COMPRESSION case (direct-impact bar) and write dump.npz.

All parameters live in config.toml under [compression] -- materials, geometry,
gauge locations, numerics. Nothing is set here.

    python3 drive_compression.py

Its sibling drive_tension.py runs the SHTB and writes the SAME filename, so the
last one run is what the analysis scripts see.
"""
import config
import simulate_compression
from dump import write_dump

cfg = config.load('compression')
sim = simulate_compression.SimulateDirectImpact(cfg)
write_dump(sim, cfg)
