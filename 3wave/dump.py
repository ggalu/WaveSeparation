"""
dump.npz -- the single file the simulators write and the analysis scripts read.

It replaces the old eps.npy / force.npy / meta.npz trio. Two things changed
besides the filename:

  * it holds only what is consumed -- gauge signals, the interface-force ground
    truth and the specimen truth -- rather than every element at every timestep
    (see recording.py). The tension case went from 1.8 GB to a few MB;
  * the gauge signals arrive with their exact positions already resolved, so the
    analysis scripts no longer each re-derive element indices from the geometry.
    That lookup lived in four copies and is now in one.

Contents
--------
Gauge data, ready to hand straight to wave_separation.separate:
    eps_in, eps_out       (n_gauge, N)  strain history at each gauge
    pos_in, pos_out       (n_gauge,)    exact distance from the interface [mm]

Ground truth:
    force_iface_in/out    (N,)   force in the bar element at each specimen face
    spec_strain           (N,)   mean specimen strain, loading sense of the case
    spec_stress           (N,)   mean specimen stress [GPa]

Geometry and units (mm, ms, kg => kN, GPa; mm/ms == m/s):
    E, A                  bar Young's modulus [GPa] and cross-section [mm^2]
    c0                    elastic bar wave speed [mm/ms]
    dt, dx, N             timestep [ms], element length [mm], number of steps
    iface_in, iface_out   element indices bounding the specimen
    X_IN, X_OUT           the two interface planes [mm] -- what the separation
                          reconstructs at, and what gauge distances measure from
    L_free_in/out         interface to that bar's far end [mm]
    L_specimen, A_specimen    original specimen length [mm] and area [mm^2]
    v0_in, v0_out         rigid-body bar velocity before any wave arrives
                          [mm/ms]. Separation cannot see rigid-body motion, so
                          these must be handed to bar_interface(v0=...)
    loading               'compression' or 'tension' -- the sign convention of
                          the stored data, for specimen_response(loading=...)
    eta                   the analysis exponential window [1/ms], from config

Present only when record_full_field is set in config.toml:
    eps_full, force_full  (N_x, N) float32
"""

import numpy as np

__all__ = ['write_dump', 'load_dump', 'DUMP_FILE']

DUMP_FILE = 'dump.npz'


def write_dump(sim, cfg, path=DUMP_FILE):
    """Collect a finished simulator's recorded output into one .npz."""
    fields = dict(sim.rec.as_dump())
    fields.update(
        E=sim.E_bar, A=sim.A_bar, c0=sim.c0, dt=sim.dt, dx=sim.dx0,
        N=sim.num_timesteps,
        L_specimen=sim.L_specimen, A_specimen=sim.specimen_cross_section_area,
        v0_in=sim.v0_in, v0_out=sim.v0_out,
        loading=sim.loading, eta=cfg['analysis']['eta'],
        spec_strain=sim.epsS, spec_stress=sim.sigS,
    )
    np.savez(path, **fields)

    n_g = fields['eps_in'].shape[0]
    print(f"\nwrote {path}: {n_g} gauges/bar at "
          f"{[f'{p:.1f}' for p in fields['pos_in']]} mm (input) and "
          f"{[f'{p:.1f}' for p in fields['pos_out']]} mm (output)")
    print(f"  interface elements {fields['iface_in']} / {fields['iface_out']}, "
          f"planes at x = {fields['X_IN']} and {fields['X_OUT']} mm")
    if 'eps_full' in fields:
        print(f"  full field INCLUDED: {fields['eps_full'].shape} "
              "(record_full_field is on in config.toml)")


def load_dump(path=DUMP_FILE):
    """
    Read dump.npz into a plain dict, with the scalars already unwrapped.

    np.load returns every scalar as a 0-d array, which then has to be cast at
    every use site. Doing it once here is what lets the analysis scripts open
    with a single line instead of a block of float()/int() calls.
    """
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    for k in ('E', 'A', 'c0', 'dt', 'dx', 'X_IN', 'X_OUT', 'L_free_in',
              'L_free_out', 'L_specimen', 'A_specimen', 'v0_in', 'v0_out',
              'eta'):
        d[k] = float(d[k])
    for k in ('N', 'iface_in', 'iface_out'):
        d[k] = int(d[k])
    d['loading'] = str(d['loading'])
    d['t'] = np.arange(d['N']) * d['dt']
    return d
