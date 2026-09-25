"""
Run a simulation case and write its dump.npz, into the case's own folder.

    python3 simulate.py cases/simulations/tension
    python3 simulate.py cases/simulations/compression
    python3 simulate.py cases/simulations/calibration_tension
    python3 simulate.py cases/simulations/calibration_compression

All parameters live in the case's case.toml (plus defaults.toml); nothing is
set here. `model` picks the simulator: "compression" is the direct-impact bar
(simulate_compression.py), "tension" the SHTB with striker and anvil
(simulate_tension.py). The dump records the sign convention -- TENSION POSITIVE
for the SHTB -- so nothing downstream has to be told.

Never run a simulate_*.py directly: it produces no dump.
"""
import argparse

from wave_separation_code import config
from simulation_code import simulate_compression
from simulation_code import simulate_tension
from wave_separation_code.dump import write_dump

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('case', help='a folder under cases/simulations/')
ARGS = _ap.parse_args()

cfg = config.load(ARGS.case)
if cfg['kind'] != 'simulation':
    raise SystemExit(f'{ARGS.case} has kind = "{cfg["kind"]}"; simulate.py runs '
                     'folders under cases/simulations/ only')

SIMULATORS = {'compression': simulate_compression.SimulateDirectImpact,
              'tension': simulate_tension.SimulateSHTB}
sim = SIMULATORS[cfg['model']](cfg)
write_dump(sim, cfg)
