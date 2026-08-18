# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-24 16:35:10
#
# 1D direct-impact Hopkinson bar: a moving input bar strikes a specimen resting
# against a stationary output bar. Explicit leapfrog integration of a lumped
# mass-spring chain.
#
# The two bars are DIFFERENT: an aluminium input bar and a short polycarbonate
# output bar, from [compression.input_bar] and [compression.output_bar]. The
# chain is therefore not uniform -- element stiffness, area and density are
# per-element arrays and nodal masses are assembled from the two adjacent
# elements, exactly as simulate_tension.py already had to do for its steel
# anvil. Getting that wrong at the aluminium/polycarbonate junction would give
# the wrong reflection there, which is the whole point of the output bar.
#
# NOTE the dump still carries ONE set of bar properties (E, A, rho, c0), taken
# from the INPUT bar. The reduction downstream therefore treats the output bar
# as aluminium too, which it no longer is.
#
# Reduced to what the wave-separation pipeline consumes:
#   - self.history_elems_strain, self.history_elems_force  (read by drive_compression.py)
#   - specimen.dat                                          (ground truth for
#     reduce_specimen.py and gauge_count_study.py)
# All plotting, animation and the other four output files have been removed --
# visualisation is handled downstream by plot_forces.py and reduce_specimen.py.
#
# Units: mm, ms, kg  =>  kN, GPa, and mm/ms (numerically equal to m/s).

import numpy as np

import config as _config
from recording import GaugeRecorder


class SimulateDirectImpact:
    """
    Parameters come from the [compression] case of config.toml -- see that file
    for what each one means. Pass a config dict to override for a sweep.
    """

    def __init__(self, cfg=None):
        cfg = _config.load('compression') if cfg is None else cfg
        self.cfg = cfg
        in_bar, out_bar = cfg['input_bar'], cfg['output_bar']
        spec, num = cfg['specimen'], cfg['numerics']

        # Specimen properties. A calibration shot has NO specimen -- the two bar
        # faces touch directly -- and says so with length = 0. Every material key
        # is meaningless then, so none of them is required; the defaults below
        # are never used because no element carries them.
        self.L_specimen = spec['length']
        self.no_specimen = self.L_specimen == 0.0
        _need = (lambda k: spec.get(k, 0.0)) if self.no_specimen else (lambda k: spec[k])
        self.specimen_dia = _need('diameter') or in_bar['diameter']
        self.specimen_cross_section_area = 0.25 * np.pi * self.specimen_dia**2
        self.specimen_E = _need('E') or in_bar['E']
        self.JC_A, self.JC_B, self.JC_n = _need('JC_A'), _need('JC_B'), _need('JC_n')
        self.elastic_only = bool(spec.get('elastic_only', self.no_specimen))
        # Only the specimen's inertia depends on this, and it is 10 mm of a
        # 3010 mm chain: absent from the TOML, assume the input bar's alloy.
        self.rho_specimen = spec.get('rho', in_bar['rho'])

        # --- bar materials and geometry ------------------------------------
        # E_bar / rho_bar / A_bar / diameter_bar are the INPUT bar. They keep
        # those names because dump.py writes them as "the bar", and because the
        # reduction has only one set of bar properties to be told about.
        self.E_bar, self.rho_bar = in_bar['E'], in_bar['rho']
        self.E_outbar, self.rho_outbar = out_bar['E'], out_bar['rho']
        self.diameter_bar = in_bar['diameter']
        self.diameter_outbar = out_bar['diameter']
        self.A_bar = 0.25 * np.pi * self.diameter_bar**2
        self.A_outbar = 0.25 * np.pi * self.diameter_outbar**2

        self.L_inputbar = in_bar['L_input']
        self.L_outputbar = out_bar['L_output']
        self.initial_velocity = in_bar['initial_velocity']

        # numerics
        self.courant = num['courant']
        self.damping = num['damping']
        self.N_cycles = num['ncyc']
        self.dx_target = num['dx']
        self.record_full_field = num['record_full_field']
        self.gauge_distances = list(cfg['gauges'])

        # the input bar arrives at its impact velocity; the output bar is at rest
        self.v0_in, self.v0_out = self.initial_velocity, 0.0
        self.loading = cfg['loading']

        self.initialize_spatial_mesh()
        self.initialize_time_discretization()
        self.initialize_history_arrays()
        self.apply_initial_conditions()

        self.integrate_time()

        self.extract_discrete_results()

    def integrate_time(self):
        """
        Perform explicit time integration.
        """
        print(f"duration {self.T[-1]:.4f} ms in {self.num_timesteps} steps "
              f"of {self.dt*1e3:.4f} us")
        for i in range(self.num_timesteps):

            # 1st part of leapfrog: update velocities half-step using old accelerations
            self.v += 0.5 * self.f * self.dt / self.m

            # second part of leapfrog: update positions
            self.x += self.dt * self.v
            # compute force between springs
            # we have N_x + 1 nodes and N_x elements
            dx = self.x[1:] - self.x[0:-1]
            eps = (dx - self.dx0) / self.dx0

            # purely elastic stress update, per element: the chain carries
            # aluminium, specimen and polycarbonate and E is no longer one number
            stress = self.E_elem * eps

            # now, overwrite stress for the specimen elements only
            if not self.elastic_only and not self.no_specimen:
                stress[self.specimenIndices] = self.computeStressStrainSpecimen(dx[self.specimenIndices])

            # unilateral contact: a bar face can push, never pull. Two faces
            # with a loose specimen between them, one when the bars touch.
            for _f in self.contact_faces:
                stress[_f] = min(stress[_f], 0.0)

            # add artificial viscosity. It acts through the local impedance
            # rho*c == sqrt(E*rho), which is per element for the same reason.
            dv = self.v[1:] - self.v[0:-1]
            stress += self.damping * self.Zc_elem * dv

            # create element force array
            element_forces = stress * self.A_elem

            self.f[:] = 0.0
            self.f[:-1] = element_forces  # ... distribute stress as nodal forces
            self.f[1:] -= element_forces

            # third part of leapfrog: update velocities with new accelerations
            self.v += 0.5 * self.f * self.dt / self.m

            self.rec.record(i, eps, element_forces)

    def computeStressStrainSpecimen(self, l):
        """
        Simple linear elastic - perfect plastic material behaviour.
        l : array of current element lengths
        """
        eps = (l - self.dx0) / self.dx0
        eps -= self.eps_plastic

        # compute state-dependent yield stress
        yield_stress = self.JC_A + self.JC_B * abs(self.eps_plastic)**self.JC_n
        stress = self.specimen_E * eps  # predict stress

        for i in range(len(eps)):  # compression
            if stress[i] < -yield_stress[i]:
                dsig = stress[i] + yield_stress[i]  # amount outside the yield surface
                stress[i] = -yield_stress[i]
                # NOTE: this uses the INPUT BAR modulus, not specimen_E.
                # Preserved as-is so the specimen response is unchanged; see README.
                deps = dsig / self.E_bar
                self.eps_plastic[i] += deps

        return stress

    def extract_discrete_results(self):
        """
        Write the specimen ground truth consumed by the reduction scripts.
        """
        # Specimen stress is recovered from the element force rather than stored
        # separately: for specimen elements force = stress * specimen area, so
        # this is exact. The means were accumulated during the run.
        self.epsS = self.rec.spec_strain
        self.forceS = self.rec.spec_force
        self.sigS = self.forceS / self.specimen_cross_section_area

        np.savetxt("specimen.dat", np.column_stack((self.T, self.sigS, self.epsS)),
                   header="time[ms]  mean specimen stress[GPa]  mean specimen strain[-]"
                          "  (compression negative)")
        print("... wrote specimen.dat")

    def apply_initial_conditions(self):
        # Apply the impact velocity to the input bar's NODES. inputBarIndices is
        # an element list now, so it cannot be reused here -- and the node on the
        # interface plane belongs to the impacting face and must move too.
        self.v[self.x <= self.L_inputbar] = self.initial_velocity

    def initialize_history_arrays(self):
        """
        Initialize the history variables used to save simulation results.

        Only the gauge rows, the two interface elements and the specimen means
        are kept; see recording.py. Nodal velocity/displacement and element
        stress histories were dropped earlier, and the full element field is now
        opt-in via config.toml.
        """
        self.rec = GaugeRecorder(
            self.specimenIndices, self.dx0, self.L, self.gauge_distances,
            self.num_timesteps, self.N_x,
            record_full_field=self.record_full_field,
            bar_indices=(self.inputBarIndices, self.outputBarIndices))

    def initialize_time_discretization(self):
        # c0 is the INPUT bar's wave speed -- it is what dump.py writes and what
        # the reduction uses. The output bar and the specimen have their own,
        # and the fastest material present sets the stable timestep.
        self.c0 = np.sqrt(self.E_bar / self.rho_bar)
        self.c_outbar = np.sqrt(self.E_outbar / self.rho_outbar)
        # Round trips of the SHORTER-IN-TIME bar, each at its own wave speed.
        # With two identical bars this is the old min(L_in, L_out)*2/c0.
        endTime = self.N_cycles * 2 * min(self.L_inputbar / self.c0,
                                          self.L_outputbar / self.c_outbar)
        self.dt = self.courant * self.dx0 / self.c_elem.max()
        self.num_timesteps = int(endTime / self.dt)
        # arange, not linspace: this must agree sample-for-sample with the
        # t = np.arange(N)*dt used by drive_compression.py and the reduction scripts.
        self.T = np.arange(self.num_timesteps) * self.dt

    def initialize_spatial_mesh(self):
        self.L = self.L_inputbar + self.L_specimen + self.L_outputbar
        dx = self.dx_target  # discretisation (element) length, from config.toml
        self.N_x = int(self.L / dx)  # number of elements
        self.x = np.linspace(0, self.L, self.N_x + 1)  # nodal position
        self.v = np.zeros_like(self.x)  # nodal velocity
        self.f = np.zeros_like(self.x)  # nodal forces
        self.m = np.zeros_like(self.x)  # nodal masses
        self.dx0 = self.x[1] - self.x[0]  # initial length of each spring

        # Regions are assigned by ELEMENT CENTRE, element i spanning [i, i+1]*dx.
        # This used to be built from NODE positions and then used on element
        # arrays, which put the specimen one element to the right of nominal.
        # That shortcut is no longer available: the MATERIAL PROPERTIES hang off
        # these indices now, not just the labels, and the node version's last
        # output-bar index runs one past the end of every element array.
        # simulate_tension.py has always done it this way.
        centres = (np.arange(self.N_x) + 0.5) * dx
        x_spec0 = self.L_inputbar                       # input bar | specimen
        x_spec1 = x_spec0 + self.L_specimen             # specimen | output bar
        self.inputBarIndices = np.flatnonzero(centres < x_spec0)
        self.specimenIndices = np.flatnonzero((centres >= x_spec0)
                                              & (centres < x_spec1))
        self.outputBarIndices = np.flatnonzero(centres >= x_spec1)

        # The bar elements bounding the specimen, where the unilateral (no
        # tension) condition is enforced each step.
        self.iface_in = int(self.inputBarIndices[-1])
        self.iface_out = int(self.outputBarIndices[0])

        # A loose specimen has a contact face on BOTH sides -- neither bar can
        # pull it. With no specimen the bars touch on ONE plane, and iface_in
        # and iface_out are the two elements meeting there; clipping both would
        # stop the output bar carrying the tensile wave its own free end sends
        # back, 1 mm inside the bar and for no physical reason. Clip the input
        # side only: that single condition IS the contact.
        self.contact_faces = ((self.iface_in,) if self.no_specimen
                              else (self.iface_in, self.iface_out))

        # per-element material properties; input bar everywhere, then overwrite
        self.E_elem = np.full(self.N_x, self.E_bar)
        self.A_elem = np.full(self.N_x, self.A_bar)
        self.rho_elem = np.full(self.N_x, self.rho_bar)
        for idx, (Ee, Ae, re) in (
                (self.specimenIndices, (self.specimen_E,
                                        self.specimen_cross_section_area,
                                        self.rho_specimen)),
                (self.outputBarIndices, (self.E_outbar, self.A_outbar,
                                         self.rho_outbar))):
            self.E_elem[idx], self.A_elem[idx], self.rho_elem[idx] = Ee, Ae, re

        # local wave speed and impedance rho*c == sqrt(E*rho), per element
        self.c_elem = np.sqrt(self.E_elem / self.rho_elem)
        self.Zc_elem = np.sqrt(self.E_elem * self.rho_elem)

        # accumulated plastic strain, one entry per specimen element
        self.eps_plastic = np.zeros(len(self.specimenIndices), float)

        # Nodal masses: half of each adjacent element's mass, so a node on a
        # material junction gets the correct average and the free ends get half.
        # A uniform mass per node was fine while the whole chain was one alloy;
        # it would put aluminium inertia on the polycarbonate nodes now.
        m_elem = self.rho_elem * self.A_elem * self.dx0
        self.m[:-1] += 0.5 * m_elem
        self.m[1:] += 0.5 * m_elem


if __name__ == "__main__":
    simulator = SimulateDirectImpact()
