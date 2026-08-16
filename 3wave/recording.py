"""
Selective recording of simulator output.

The simulators used to keep strain and force for EVERY element at EVERY
timestep. For the tension case that is a 6030 x 23038 pair of float64 arrays --
2.2 GB in memory, 1.1 GB on disk -- to produce six gauge signals. This module
records only what the analysis scripts actually consume:

  * strain at each gauge element, one row per gauge per bar;
  * force in the two bar elements bounding the specimen, which sep_test.py uses
    as the ground truth for the separation;
  * the mean specimen strain and force, accumulated per step rather than stored
    per element, which is all extract_discrete_results and plot_forces need.

That is ~10 numbers per timestep instead of ~12000.

Both simulators share this class so that the gauge-to-element lookup is defined
exactly once. The convention matches what the analysis scripts previously
duplicated: element i spans [i, i+1]*dx and is centred at (i+0.5)*dx, so the
gauge nearest a requested distance d from the interface is round((X - d)/dx - .5)
and its exact distance is recovered rather than assumed.
"""

import numpy as np

__all__ = ['GaugeRecorder']


class GaugeRecorder:
    def __init__(self, specimen_indices, dx, total_length, gauge_distances,
                 n_steps, n_elements, record_full_field=False,
                 bar_indices=None):
        """
        Parameters
        ----------
        specimen_indices : array of int
            ELEMENT indices occupied by the specimen.
        dx : float
            Element length.
        total_length : float
            Overall model length, used for the output bar's free-end distance.
        gauge_distances : sequence of float
            Requested gauge distances from the interface plane [mm]. The same
            list is used on both bars.
        n_steps, n_elements : int
            Sizes of the run.
        record_full_field : bool
            Also keep the every-element/every-timestep arrays (the old
            behaviour). Expensive; see config.toml.
        bar_indices : (input_elements, output_elements), optional
            ELEMENT indices of the two bars, used to record how far the UNIFORM
            bar material actually extends from each interface. That is not the
            same as the distance to the far end: the SHTB carries a 20 mm steel
            anvil beyond the end of the input bar, and a reconstruction that
            runs into it is extrapolating through the wrong wave speed. Omit
            on a model whose bars run all the way to their far ends.
        """
        spec = np.asarray(specimen_indices)

        # Interface elements and the planes they bound. Derived from the
        # specimen indices rather than assumed, because simulate_compression.py's regions
        # are built as node indices and used on element arrays.
        self.iface_in = int(spec.min()) - 1      # last input-bar element
        self.iface_out = int(spec.max()) + 1     # first output-bar element
        self.X_IN = (self.iface_in + 1) * dx     # input bar face
        self.X_OUT = self.iface_out * dx         # output bar face

        # Distance from each interface to that bar's far end. Kills the
        # hardcoded total length the analysis scripts used to carry.
        self.L_free_in = self.X_IN
        self.L_free_out = total_length - self.X_OUT

        # How far the UNIFORM bar reaches from each interface, which is what
        # bounds a wave reconstruction. Equal to L_free_* unless something of
        # another material sits between the bar and its far end -- the SHTB's
        # steel anvil does, so L_bar_in is 3000 mm against L_free_in's 3020.
        # separate() assumes one wave speed all the way from the gauge to the
        # plane it reconstructs at, so past L_bar_* the result is extrapolation
        # through the wrong material and must be masked, not plotted.
        # Clamped to the far end: simulate_compression.py builds these as NODE
        # indices and uses them on element arrays (see its own comment), so its
        # out_e.max() runs one past the last element and would put the bar
        # beyond the end of the model. A bar can never be longer than the
        # distance to its own far end, so clamping is right regardless.
        if bar_indices is None:
            self.L_bar_in, self.L_bar_out = self.L_free_in, self.L_free_out
        else:
            in_e, out_e = (np.asarray(a) for a in bar_indices)
            self.L_bar_in = min(self.X_IN - int(in_e.min()) * dx, self.L_free_in)
            self.L_bar_out = min((int(out_e.max()) + 1) * dx - self.X_OUT,
                                 self.L_free_out)

        self.dx = dx
        self.gauge_distances = list(gauge_distances)
        self.elem_in, self.pos_in = self._resolve('in', gauge_distances)
        self.elem_out, self.pos_out = self._resolve('out', gauge_distances)

        for name, elems, lo, hi in (('input', self.elem_in, 0, self.iface_in),
                                    ('output', self.elem_out, self.iface_out,
                                     n_elements - 1)):
            if elems.min() < lo or elems.max() > hi:
                raise ValueError(
                    f'a gauge falls outside the {name} bar (elements {lo}..{hi}); '
                    f'requested distances {list(gauge_distances)} mm')

        n_g = len(self.gauge_distances)
        self.eps_in = np.zeros((n_g, n_steps))
        self.eps_out = np.zeros((n_g, n_steps))
        self.force_iface = np.zeros((2, n_steps))
        self.spec_strain = np.zeros(n_steps)
        self.spec_force = np.zeros(n_steps)
        self._spec = spec
        self._iface = np.array([self.iface_in, self.iface_out])

        self.record_full_field = bool(record_full_field)
        if self.record_full_field:
            self.eps_full = np.zeros((n_elements, n_steps))
            self.force_full = np.zeros((n_elements, n_steps))
        else:
            self.eps_full = self.force_full = None

    def _resolve(self, bar, distances):
        """
        Requested distances -> (element indices, exact distances achieved).

        A gauge sits on whichever element centre is nearest the requested
        distance, so the distance actually realised differs from the request by
        up to dx/2. That exact value is what goes into the dump and into
        separate(): the reconstruction is sensitive to gauge position, and
        rounding it back to the nominal figure would introduce a real error.
        """
        dx, plane = self.dx, (self.X_IN if bar == 'in' else self.X_OUT)
        sign = -1.0 if bar == 'in' else +1.0
        elems, exact = [], []
        for d in distances:
            i = int(round((plane + sign * d) / dx - 0.5))
            elems.append(i)
            exact.append(sign * ((i + 0.5) * dx - plane))
        return np.asarray(elems, int), np.asarray(exact, float)

    def record(self, step, eps, element_forces):
        """Store one timestep. Called once per step from the integration loop."""
        self.eps_in[:, step] = eps[self.elem_in]
        self.eps_out[:, step] = eps[self.elem_out]
        self.force_iface[:, step] = element_forces[self._iface]
        self.spec_strain[step] = eps[self._spec].mean()
        self.spec_force[step] = element_forces[self._spec].mean()
        if self.record_full_field:
            self.eps_full[:, step] = eps
            self.force_full[:, step] = element_forces

    def as_dump(self):
        """The arrays that go into dump.npz, as a dict."""
        out = dict(eps_in=self.eps_in, eps_out=self.eps_out,
                   pos_in=self.pos_in, pos_out=self.pos_out,
                   force_iface_in=self.force_iface[0],
                   force_iface_out=self.force_iface[1],
                   iface_in=self.iface_in, iface_out=self.iface_out,
                   X_IN=self.X_IN, X_OUT=self.X_OUT,
                   L_free_in=self.L_free_in, L_free_out=self.L_free_out,
                   L_bar_in=self.L_bar_in, L_bar_out=self.L_bar_out)
        if self.record_full_field:
            out['eps_full'] = self.eps_full.astype(np.float32)
            out['force_full'] = self.force_full.astype(np.float32)
        return out
