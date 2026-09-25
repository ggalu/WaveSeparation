"""
Loader for the per-case configuration: one folder per case under cases/, each
with its own case.toml, on top of the shared defaults.toml.

    from wave_separation_code import config
    cfg = config.load('cases/simulations/tension')

    cfg['input_bar']['E']    # material and geometry, as written in the file
    cfg['gauges']            # gauge distances from the interface [mm]
    cfg['numerics']['dx']    # defaults.toml's numerics, with the case's override
    cfg['analysis']['eta']
    cfg['case_dir']          # absolute path of the folder: inputs and outputs
    config.resolve(cfg, 'data')    # a path in case.toml, made absolute

A case is a FOLDER, and a folder is one of three kinds, stated in its
case.toml as `kind` and mirrored by the directory it sits in:

    cases/simulations/<name>      kind = "simulation"      a model to integrate;
                                  `model` = "compression" | "tension" picks the
                                  simulator. simulate.py writes dump.npz here.
    cases/identifications/<name>  kind = "identification"  a no-specimen shot the
                                  bars are identified from; `method` =
                                  "compression" | "tension" picks the script.
                                  The record is EITHER `data` (a measured file in
                                  this folder) OR `simulation` (a simulation
                                  folder whose dump.npz is read -- and whose own
                                  configuration is inherited underneath this
                                  one's keys). Writes bar_identified.npz here.
    cases/analyses/<name>         kind = "analysis"        a measured shot with a
                                  specimen. `data` is the record; `bars` names
                                  the identification folder its c0, positions,
                                  alpha(f) and c_p(f) come from.

Paths inside case.toml (`data`, `simulation`, `bars`) are relative to the
case's own folder, never to the working directory.

Every case carries TWO bar tables, [input_bar] and [output_bar], whether or not
the two bars are the same (a single-bar measured case has [bar] instead). The
compression model genuinely differs (aluminium 2000 mm against polycarbonate
1000 mm); the SHTB cases repeat themselves. That uniformity is deliberate --
dump.npz records E / A / rho / c0 per bar, so nothing downstream has to ask
whether a rig happens to be symmetric. bar_lengths() returns the pair without
the caller reaching into either table.

Nothing here computes derived quantities -- areas, wave speeds and element
indices are the simulators' business, because that is where the geometry lives.
This module only reads, merges and validates.

tomllib is in the standard library from Python 3.11, so this adds no dependency.
"""

import os
import tomllib

__all__ = ['load', 'resolve', 'measured', 'bar_lengths', 'KINDS', 'BAR_TABLES',
           'ROOT', 'DEFAULTS_PATH', 'CASE_FILE']

# 3wave/, the folder above this package: defaults.toml and cases/ live there.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS_PATH = os.path.join(ROOT, 'defaults.toml')
CASE_FILE = 'case.toml'

# kind -> the directory under cases/ that holds that kind. The folder a case
# sits in must agree with what its case.toml says it is.
KINDS = {'simulation': 'simulations',
         'identification': 'identifications',
         'analysis': 'analyses'}

# What each simulation `model` / identification `method` may be.
MODELS = ('compression', 'tension')

# The bar tables every case carries, and the key in each that holds its length.
# Everything needing a bar length goes through bar_lengths() rather than
# reaching into a table by name.
BAR_TABLES = (('input_bar', 'L_input'), ('output_bar', 'L_output'))

# Keys load() adds; not inherited from a referenced simulation.
_META = ('kind', 'case', 'case_dir', 'model', 'numerics', 'analysis')


def _case_dir(case):
    """Absolute path of a case folder, given the folder or its case.toml."""
    d = os.path.abspath(os.fspath(case))
    if os.path.basename(d) == CASE_FILE:
        d = os.path.dirname(d)
    if not os.path.isfile(os.path.join(d, CASE_FILE)):
        raise FileNotFoundError(
            f'{case}: not a case folder -- no {CASE_FILE} in {d}. Cases live '
            'under cases/simulations/, cases/identifications/ and '
            'cases/analyses/, one folder each.')
    return d


def _read(path):
    with open(path, 'rb') as fh:
        return tomllib.load(fh)


def load(case):
    """
    Return the merged configuration for one case folder.

    Parameters
    ----------
    case : path
        The case folder (or its case.toml), absolute or relative to the working
        directory.

    Returns
    -------
    dict
        The case.toml's own keys and tables, plus 'numerics' (defaults.toml's
        [numerics] updated with the case's [numerics]), 'analysis' (likewise),
        'case' (the folder name), 'case_dir' (its absolute path) and 'kind'.
        An identification of a simulated shot also carries every key of that
        simulation's configuration its own case.toml does not set, and
        'simulation_dir'.
    """
    d = _case_dir(case)
    where = os.path.join(d, CASE_FILE)
    own = _read(where)
    defaults = _read(DEFAULTS_PATH)

    kind = own.get('kind')
    if kind not in KINDS:
        raise ValueError(f'{where}: kind must be one of {tuple(KINDS)}; '
                         f'got {kind!r}')
    parent = os.path.basename(os.path.dirname(d))
    if parent != KINDS[kind]:
        raise ValueError(f'{where}: kind = "{kind}" belongs under '
                         f'cases/{KINDS[kind]}/, not {parent}/')

    base = {}
    numerics = dict(defaults['numerics'])
    analysis = dict(defaults['analysis'])
    if kind == 'identification' and 'simulation' in own:
        sim = load(_resolve_in(d, own['simulation'], where, 'simulation'))
        if sim['kind'] != 'simulation':
            raise ValueError(f'{where}: `simulation` must name a simulation '
                             f'folder; {sim["case_dir"]} is a {sim["kind"]}')
        base = {k: v for k, v in sim.items() if k not in _META}
        base['simulation_dir'] = sim['case_dir']
        numerics, analysis = dict(sim['numerics']), dict(sim['analysis'])

    cfg = dict(base)
    cfg.update(own)
    numerics.update(own.get('numerics', {}))
    analysis.update(own.get('analysis', {}))
    cfg['numerics'] = numerics
    cfg['analysis'] = analysis
    cfg['case'] = os.path.basename(d)
    cfg['case_dir'] = d

    _validate(cfg, where)
    return cfg


def _resolve_in(d, rel, where, key):
    p = os.path.normpath(os.path.join(d, os.fspath(rel)))
    if not os.path.exists(p):
        raise FileNotFoundError(f'{where}: {key} = "{rel}" -> {p}, which does '
                                'not exist (paths are relative to the case '
                                'folder)')
    return p


def resolve(cfg, key):
    """Absolute path of a path-valued key (`data`, `bars`, `simulation`)."""
    return _resolve_in(cfg['case_dir'], cfg[key],
                       os.path.join(cfg['case_dir'], CASE_FILE), key)


def measured(cfg):
    """True for a case whose record is a measured file rather than a dump."""
    return 'data' in cfg


def bar_lengths(cfg):
    """
    (L_input, L_output) for a loaded case, whichever bar layout it uses.

    Saves every caller from reaching into [input_bar]['L_input'] and its
    output-bar twin by hand, and from caring that the two lengths live in
    different tables.
    """
    return tuple(cfg[table][key] for table, key in BAR_TABLES)


def _validate(cfg, where):
    """Catch the mistakes that would otherwise fail silently or far downstream."""
    kind = cfg['kind']
    if cfg.get('loading') not in ('compression', 'tension'):
        raise ValueError(f"{where}: loading must be 'compression' or 'tension'")

    # Bar face -> specimen, through whatever holds the specimen; the analysis
    # reconstructs there instead of at the face. Optional, default 0.
    h = cfg.get('holder_length', 0.0)
    if not isinstance(h, (int, float)) or isinstance(h, bool) or h < 0:
        raise ValueError(f'{where}: holder_length must be a number >= 0 [mm]; '
                         f'got {h!r}')

    if kind == 'identification':
        if cfg.get('method') not in MODELS:
            raise ValueError(f'{where}: method must be one of {MODELS}; '
                             f'got {cfg.get("method")!r}')
        if ('data' in cfg) == ('simulation' in cfg):
            raise KeyError(f'{where}: an identification needs exactly one of '
                           '`data` (a measured record) or `simulation` (a '
                           'simulation folder)')
    if kind == 'analysis':
        if 'bars' not in cfg:
            raise KeyError(f'{where}: an analysis needs `bars`, the '
                           'identification folder its bar properties come from')
        bars = resolve(cfg, 'bars')
        own = _read(os.path.join(_case_dir(bars), CASE_FILE))
        if own.get('kind') != 'identification':
            raise ValueError(f'{where}: `bars` must name an identification '
                             f'folder; {bars} is a {own.get("kind")!r}')
        if 'data' not in cfg:
            raise KeyError(f'{where}: an analysis needs `data`, its record')

    if measured(cfg):
        _validate_experiment(cfg, where)
        return

    if kind == 'simulation' and cfg.get('model') not in MODELS:
        raise ValueError(f'{where}: model must be one of {MODELS}; '
                         f'got {cfg.get("model")!r}')
    model = cfg.get('model') or cfg.get('method')

    for key in [table for table, _ in BAR_TABLES] + ['specimen']:
        if key not in cfg:
            raise KeyError(f'{where}: missing [{key}] table')
    if model == 'tension':
        for key in ('striker', 'anvil'):
            if key not in cfg:
                raise KeyError(f'{where}: missing [{key}] table')

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


def _validate_experiment(cfg, where):
    """
    The subset of _validate that still means something for a measured shot.

    Gone: the mesh and the timestep -- there is nothing to integrate. Kept: the
    sign convention, eta, and the gauge list, because those reach `separate`
    exactly as they do for a simulated case.

    Two bar-table shapes are accepted, and exactly one must be present:

    - a single [.bar] table -- one instrumented bar (identifications/pc_bar has no
      gauge on its aluminium input bar at all, so this is the common case);
    - [.input_bar]/[.output_bar] -- both bars instrumented, joined through
      [.specimen].length (0 for bars butted directly together). Gauge columns
      must then be named "in-*"/"out-*" so they can be split by bar.

    NOTE the gauge list here is TAPE, not truth. identify_bar_compression.py is
    never told it; reconstruct_interface.py uses it as one of the two position
    sets it compares. It is validated so that a typo fails here rather than
    inside an FFT.
    """
    if cfg['analysis']['eta'] <= 0:
        raise ValueError(f'{where}: eta must be > 0 (separate() is singular '
                         'at DC for eta = 0)')

    for key in ('data', 'columns'):
        if key not in cfg:
            raise KeyError(f'{where}: missing "{key}"')

    cols = cfg['columns']
    if 'time' not in cols:
        raise KeyError(f'{where}: [.columns] must name a "time" column index')
    gauge_cols = [k for k in cols if k != 'time']
    if not gauge_cols:
        raise KeyError(f'{where}: [.columns] names no gauge channels')
    if len(set(cols.values())) != len(cols):
        raise ValueError(f'{where}: two channels share a column index: {cols}')

    gauges = cfg.get('gauges')
    if not gauges:
        raise KeyError(f'{where}: missing or empty "gauges" (tape positions)')
    if len(gauges) != len(gauge_cols):
        raise ValueError(f'{where}: {len(gauges)} tape positions but '
                         f'{len(gauge_cols)} gauge channels in [.columns]')
    if any(g <= 0 for g in gauges):
        raise ValueError(f'{where}: gauge distances must be > 0; got {gauges}')

    two_bar = all(table in cfg for table, _ in BAR_TABLES)
    if two_bar and 'bar' in cfg:
        raise KeyError(f'{where}: has both "[.bar]" and [.input_bar]/'
                       '[.output_bar] -- use one shape or the other')
    if not two_bar and 'bar' not in cfg:
        raise KeyError(f'{where}: missing "[.bar]" (one instrumented bar) or '
                       '[.input_bar]/[.output_bar] (both bars instrumented)')

    if not two_bar:
        bar = cfg['bar']
        for key in ('length', 'diameter'):
            if key not in bar:
                raise KeyError(f'{where}: missing [.bar].{key}')
            if bar[key] <= 0:
                raise ValueError(f'{where}: [.bar].{key} must be > 0')
        if len(set(gauges)) != len(gauges):
            raise ValueError(f'{where}: gauge distances must be distinct; '
                             f'got {gauges}')
        if max(gauges) >= bar['length']:
            raise ValueError(f'{where}: gauge at {max(gauges)} mm does not '
                             f'fit on a {bar["length"]} mm bar')
        return

    # Two instrumented bars: gauge columns say which bar they're on, and
    # distinctness / fits-on-the-bar are scoped per bar -- the two bars
    # legitimately share a tape distance (e.g. in-1 = out-0 = 120 mm from the
    # interface), which a global uniqueness check would wrongly reject.
    for name in gauge_cols:
        if not (name.startswith('in-') or name.startswith('out-')):
            raise ValueError(f'{where}: [.columns] gauge {name!r} must be '
                             'named "in-*" or "out-*" to say which bar it is on')

    if 'specimen' not in cfg or 'length' not in cfg['specimen']:
        raise KeyError(f'{where}: missing [.specimen].length (the joint/'
                       'coupler between the two bars, 0 if butted directly '
                       'together)')
    if cfg['specimen']['length'] < 0:
        raise ValueError(f'{where}: [.specimen].length must be >= 0')

    for table, lkey in BAR_TABLES:
        bar = cfg[table]
        for key in (lkey, 'diameter'):
            if key not in bar:
                raise KeyError(f'{where}: missing [.{table}].{key}')
            if bar[key] <= 0:
                raise ValueError(f'{where}: [.{table}].{key} must be > 0')

    for prefix, (table, lkey) in zip(('in-', 'out-'), BAR_TABLES):
        idx = [i for i, n in enumerate(gauge_cols) if n.startswith(prefix)]
        sub = [gauges[i] for i in idx]
        if not sub:
            continue
        if len(set(sub)) != len(sub):
            raise ValueError(f'{where}: {prefix}* gauge distances must be '
                             f'distinct; got {sub}')
        length = cfg[table][lkey]
        if max(sub) >= length:
            raise ValueError(f'{where}: a {prefix}* gauge at {max(sub)} mm '
                             f'does not fit on the {length} mm bar')
