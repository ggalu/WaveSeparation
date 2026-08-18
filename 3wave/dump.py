"""
dump.npz -- the single file the simulators write and the analysis scripts read.

It replaces the old eps.npy / force.npy / meta.npz trio. Two things changed
besides the filename (and see the per-bar note under "Geometry and units"):

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
    E_in,   E_out         each bar's Young's modulus [GPa]
    A_in,   A_out         each bar's cross-section [mm^2]
    rho_in, rho_out       each bar's density [kg/mm^3]. Not used by the
                          reduction -- c0 and E*A are what that needs -- but
                          identify_bar_tension.py checks its density closure
                          against it.
    c0_in,  c0_out        each bar's elastic wave speed [mm/ms]

                          THESE ARE PER BAR, and the two are not interchangeable:
                          the compression case runs an aluminium input bar
                          against a polycarbonate output bar, whose c0 is a
                          quarter of the other's and whose impedance is 0.12x.
                          The SHTB cases are symmetric and write the same number
                          twice. There is deliberately no bare E / A / rho / c0
                          any more: it used to mean "the input bar" and every
                          reader silently applied it to both. A dump written
                          before this change still loads -- load_dump copies its
                          single value into both slots, which is what it meant.
    dt, dx, N             timestep [ms], element length [mm], number of steps
    iface_in, iface_out   element indices bounding the specimen
    X_IN, X_OUT           the two interface planes [mm] -- what the separation
                          reconstructs at, and what gauge distances measure from
    L_free_in/out         interface to that bar's far end [mm]
    L_bar_in/out          interface to the end of the UNIFORM bar [mm]. Equal
                          to L_free_* unless another material sits between the
                          bar and its far end -- the SHTB's 20 mm steel anvil
                          does, so L_bar_in is 3000 against L_free_in's 3020.
                          This, not L_free_*, is how far a reconstruction from
                          separate() / separate_field() is valid: past it the
                          wave speed is no longer c0 and the result is
                          extrapolation through the wrong material.
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
        E_in=sim.E_bar, E_out=sim.E_outbar,
        A_in=sim.A_bar, A_out=sim.A_outbar,
        rho_in=sim.rho_bar, rho_out=sim.rho_outbar,
        c0_in=sim.c0, c0_out=sim.c_outbar,
        dt=sim.dt, dx=sim.dx0,
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
    # Bar properties became per-bar when the compression case grew a
    # polycarbonate output bar. A dump written before that carries one value for
    # both, which is exactly what it meant back when both bars were one alloy.
    for stem in ('E', 'A', 'rho', 'c0'):
        for side in ('in', 'out'):
            key = f'{stem}_{side}'
            if key not in d:
                d[key] = d[stem]
        d.pop(stem, None)

    for k in ('E_in', 'E_out', 'A_in', 'A_out', 'rho_in', 'rho_out',
              'c0_in', 'c0_out', 'dt', 'dx', 'X_IN', 'X_OUT', 'L_free_in',
              'L_free_out', 'L_specimen', 'A_specimen', 'v0_in', 'v0_out',
              'eta'):
        d[k] = float(d[k])
    # Added later than the rest; a dump written before it falls back to the
    # far-end distance, which is right for every bar that has nothing beyond it.
    for k, fallback in (('L_bar_in', 'L_free_in'), ('L_bar_out', 'L_free_out')):
        d[k] = float(d[k]) if k in d else d[fallback]
    for k in ('N', 'iface_in', 'iface_out'):
        d[k] = int(d[k])
    d['loading'] = str(d['loading'])
    d['t'] = np.arange(d['N']) * d['dt']
    return d
