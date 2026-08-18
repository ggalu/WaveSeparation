"""
Loader for config.toml, the single source of truth for both simulators and for
the analysis scripts.

    from config import load
    cfg = load('tension')

    cfg['input_bar']['E']    # material and geometry, as written in the file
    cfg['gauges']            # gauge distances from the interface [mm]
    cfg['numerics']['dx']    # shared numerics, with any per-case override applied
    cfg['analysis']['eta']

Every case carries TWO bar tables, [<case>.input_bar] and [<case>.output_bar],
whether or not the two bars are the same. The compression case genuinely differs
(aluminium 2000 mm against polycarbonate 1000 mm); the SHTB cases repeat
themselves. That uniformity is deliberate -- dump.npz records E / A / rho / c0
per bar, so nothing downstream has to ask whether a rig happens to be symmetric.
bar_lengths() returns the pair without the caller reaching into either table.

Nothing here computes derived quantities -- areas, wave speeds and element
indices are the simulators' business, because that is where the geometry lives.
This module only reads, merges and validates.

tomllib is in the standard library from Python 3.11, so this adds no dependency.
"""

import os
import tomllib

__all__ = ['load', 'bar_lengths', 'CASES', 'BAR_TABLES', 'DEFAULT_PATH']

CASES = ('compression', 'calibration_compression',
         'tension', 'calibration_tension')

# Cases that run through simulate_tension.py and therefore need a striker and
# an anvil table as well as a bar and a specimen.
_SHTB_CASES = ('tension', 'calibration_tension')

# The bar tables every case carries, and the key in each that holds its length.
# Everything needing a bar length goes through bar_lengths() rather than
# reaching into a table by name.
BAR_TABLES = (('input_bar', 'L_input'), ('output_bar', 'L_output'))

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'config.toml')


def load(case, path=DEFAULT_PATH):
    """
    Return the merged configuration for one case.

    Parameters
    ----------
    case : one of CASES
        Which setup to read: 'compression', 'tension' or
        'calibration_tension'.
    path : str
        Location of the TOML file. Defaults to config.toml beside this module,
        so the scripts work regardless of the current working directory.

    Returns
    -------
    dict
        The case's own tables, plus 'numerics' (the shared [numerics] table
        updated with any [<case>.numerics] override) and 'analysis'.
    """
    if case not in CASES:
        raise ValueError(f'unknown case {case!r}; expected one of {CASES}')

    with open(path, 'rb') as fh:
        raw = tomllib.load(fh)

    for section in ('numerics', 'analysis', case):
        if section not in raw:
            raise KeyError(f'{path}: missing [{section}] section')

    cfg = dict(raw[case])
    # shared numerics, overridden per case where the case says so
    numerics = dict(raw['numerics'])
    numerics.update(cfg.get('numerics', {}))
    cfg['numerics'] = numerics
    cfg['analysis'] = dict(raw['analysis'])
    cfg['case'] = case

    _validate(cfg, case, path)
    return cfg


def bar_lengths(cfg):
    """
    (L_input, L_output) for a loaded case, whichever bar layout it uses.

    Saves every caller from reaching into [<case>.input_bar]['L_input'] and its
    output-bar twin by hand, and from caring that the two lengths live in
    different tables.
    """
    return tuple(cfg[table][key] for table, key in BAR_TABLES)


def _validate(cfg, case, path):
    """Catch the mistakes that would otherwise fail silently or far downstream."""
    where = f'{path} [{case}]'

    if cfg.get('loading') not in ('compression', 'tension'):
        raise ValueError(f"{where}: loading must be 'compression' or 'tension'")

    for key in [table for table, _ in BAR_TABLES] + ['specimen']:
        if key not in cfg:
            raise KeyError(f'{where}: missing [{case}.{key}] table')
    if case in _SHTB_CASES:
        for key in ('striker', 'anvil'):
            if key not in cfg:
                raise KeyError(f'{where}: missing [{case}.{key}] table')

    num = cfg['numerics']
    if not 0 < num['courant'] <= 1.0:
        raise ValueError(f"{where}: courant must be in (0, 1]; "
                         f"got {num['courant']}")
    if num['damping'] < 0:
        raise ValueError(f"{where}: damping must be >= 0")
    if num['ncyc'] < 0:
        raise ValueError(f"{where}: ncyc must be >= 0")
    if num['dx'] <= 0:
        raise ValueError(f"{where}: dx must be > 0")
    if cfg['analysis']['eta'] <= 0:
        raise ValueError(f"{where}: eta must be > 0 (separate() is singular "
                         "at DC for eta = 0)")

    gauges = cfg.get('gauges')
    if not gauges:
        raise KeyError(f'{where}: missing or empty "gauges"')
    if any(g <= 0 for g in gauges):
        raise ValueError(f'{where}: gauge distances must be > 0 (distance from '
                         f'the interface); got {gauges}')
    if len(set(gauges)) != len(gauges):
        raise ValueError(f'{where}: gauge distances must be distinct; got {gauges}')

    # A gauge further from the interface than the bar is long would silently be
    # clamped to some element in the wrong region, or in the striker's range.
    L_input, L_output = bar_lengths(cfg)
    for side, length in (('input', L_input), ('output', L_output)):
        if max(gauges) >= length:
            raise ValueError(
                f'{where}: gauge at {max(gauges)} mm does not fit on the '
                f'{side} bar ({length} mm)')
