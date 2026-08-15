"""
Shared --headless handling for the plotting scripts.

The scripts should open a window when a person runs them and write only the
.png when something automated does. Both need the same rule, and matplotlib
forces the ordering: the backend has to be chosen BEFORE pyplot is imported, so
the argument parsing has to happen at the top of the file even though the
plotting is at the bottom. That is the whole reason this lives in a helper.

    import plotting
    HEADLESS = plotting.init(__doc__)     # parse --headless, pick the backend
    import matplotlib.pyplot as plt       # only now
    ...
    fig.savefig(...)                      # always
    plotting.show_unless(HEADLESS)        # window only when interactive

Headless is chosen if --headless is passed, or MPL_HEADLESS=1 is set, or there
is no display at all -- so the scripts are safe over ssh and in batch jobs.
"""

import argparse
import os

__all__ = ['init', 'show_unless']


def init(doc=None):
    """Parse --headless, select the matplotlib backend, return the decision."""
    ap = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--headless', action='store_true',
                    help='write the figure without opening a window')
    headless = (ap.parse_args().headless
                or os.environ.get('MPL_HEADLESS') == '1'
                or not (os.environ.get('DISPLAY')
                        or os.environ.get('WAYLAND_DISPLAY')))
    import matplotlib
    if headless:
        matplotlib.use('Agg')   # must precede the pyplot import
    return headless


def show_unless(headless):
    """Open the figure window unless running headless."""
    if not headless:
        import matplotlib.pyplot as plt
        plt.show()
