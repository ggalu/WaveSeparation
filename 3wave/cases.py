"""
The files a case folder holds, and where to find the ones it depends on.

Every script takes a case FOLDER on its command line. What it reads and writes
follows from that folder alone -- never from whatever happens to be lying in
the working directory:

    record(cfg)          the shot: a measured `data` file, or the dump.npz of
                         the simulation folder (the case itself, or the one an
                         identification's `simulation` names)
    identification(cfg)  bar_identified.npz: an identification folder's own,
                         or, for an analysis, the one in its `bars` folder
    output(cfg, name)    `name` inside the case's own folder

A missing dump or identification stops with the command that makes it, rather
than falling back to a file some other case left behind.
"""
import os

import numpy as np

import config
from dump import DUMP_FILE, load_dump
from experiment import load_experiment

__all__ = ['record', 'identification', 'bars_dir', 'output', 'rel',
           'IDENT_FILE']

IDENT_FILE = 'bar_identified.npz'


def rel(path):
    """`path` relative to the working directory, for messages."""
    return os.path.relpath(path)


def output(cfg, name):
    """Where a script writes `name` for this case: inside its own folder."""
    return os.path.join(cfg['case_dir'], name)


def record(cfg):
    """The case's shot, as the dump-shaped dict every analysis consumes."""
    if config.measured(cfg):
        return load_experiment(cfg)
    if cfg['kind'] == 'simulation':
        sim_dir = cfg['case_dir']
    elif 'simulation_dir' in cfg:
        sim_dir = cfg['simulation_dir']
    else:
        raise SystemExit(f'{rel(cfg["case_dir"])}: no record -- neither `data` '
                         'nor a `simulation` to read a dump from')
    path = os.path.join(sim_dir, DUMP_FILE)
    if not os.path.exists(path):
        raise SystemExit(f'{rel(path)} not found. Run the simulation first:\n'
                         f'    python3 simulate.py {rel(sim_dir)}')
    return load_dump(path)


def bars_dir(cfg):
    """The identification folder this case's bar properties come from."""
    if cfg['kind'] == 'identification':
        return cfg['case_dir']
    if cfg['kind'] == 'analysis':
        return config.resolve(cfg, 'bars')
    raise SystemExit(f'{rel(cfg["case_dir"])} is a simulation; bar properties '
                     'come from an identification or analysis folder')


def identification(cfg):
    """
    bar_identified.npz for this case, loaded (allow_pickle, as written).

    Refuses to run when it is missing, naming the identify command -- there is
    no fallback to another case's file.
    """
    d = bars_dir(cfg)
    path = os.path.join(d, IDENT_FILE)
    if not os.path.exists(path):
        method = config.load(d)['method']
        raise SystemExit(f'{rel(path)} not found. Run the identification '
                         f'first:\n    python3 identify_bar_{method}.py {rel(d)}')
    return np.load(path, allow_pickle=True)
