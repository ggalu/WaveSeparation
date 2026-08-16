# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-24 16:35:10
#
# 1D direct-impact Hopkinson bar: a moving input bar strikes a specimen resting
# against a stationary output bar. Explicit leapfrog integration of a lumped
# mass-spring chain.
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
        bar, spec, num = cfg['bar'], cfg['specimen'], cfg['numerics']

        # specimen properties
        self.specimen_dia = spec['diameter']
        self.specimen_cross_section_area = 0.25 * np.pi * self.specimen_dia**2
        self.specimen_E = spec['E']
        self.JC_A, self.JC_B, self.JC_n = spec['JC_A'], spec['JC_B'], spec['JC_n']
        self.elastic_only = spec['elastic_only']

        # bar properties and geometry
        self.L_inputbar = bar['L_input']
        self.L_outputbar = bar['L_output']
        self.L_specimen = spec['length']
        self.E_bar = bar['E']
        self.rho = bar['rho']
        self.rho_bar = self.rho     # name the dump contract shares with SimulateSHTB
        self.diameter_bar = bar['diameter']
        self.A_bar = 0.25 * np.pi * self.diameter_bar**2
        self.initial_velocity = bar['initial_velocity']

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

            # purely elastic stress update of everything
            stress = self.E_bar * eps  # initially, everything is treated as bar material

            # now, overwrite stress for the specimen elements only
            if not self.elastic_only:
                stress[self.specimenIndices] = self.computeStressStrainSpecimen(dx[self.specimenIndices])

            # enforce no-tension condition on specimen interface elements of both bars
            stress[self.inputBarIndices[-1]] = min(stress[self.inputBarIndices[-1]], 0.0)
            stress[self.outputBarIndices[0]] = min(stress[self.outputBarIndices[0]], 0.0)

            # add artificial viscosity
            dv = self.v[1:] - self.v[0:-1]
            stress += self.damping * self.rho * self.c0 * dv

            # create element force array
            element_forces = stress * self.A_bar  # ... treat everything as bar material
            element_forces[self.specimenIndices] = (stress[self.specimenIndices]
                                                    * self.specimen_cross_section_area)

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
                # NOTE: this uses the BAR modulus, not specimen_E. Preserved as-is
                # so the specimen response is unchanged; see README.
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
        # apply initial velocity to left bar
        self.v[self.inputBarIndices] = self.initial_velocity

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
        self.c0 = np.sqrt(self.E_bar / self.rho)
        endTime = self.N_cycles * 2 * min(self.L_inputbar, self.L_outputbar) / self.c0
        self.dt = self.courant * self.dx0 / self.c0
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

        # indices of input bar, specimen, output bar.
        # NOTE: built as NODE indices but used to index ELEMENT arrays, which
        # shifts the specimen to elements 2001..2010, i.e. x in [2001, 2011]
        # rather than the nominal [2000, 2010]. Preserved as-is; drive_compression.py
        # derives the true interface planes from specimenIndices rather than
        # assuming them, so the pipeline stays consistent either way.
        specimenIndices = []
        inputBarIndices = []
        outputBarIndices = []
        for i in range(self.N_x + 1):
            pos = i * dx
            if pos <= self.L_inputbar:
                inputBarIndices.append(i)
            elif self.L_inputbar < pos <= self.L_inputbar + self.L_specimen:
                specimenIndices.append(i)
            else:
                outputBarIndices.append(i)

        self.inputBarIndices = np.asarray(inputBarIndices)
        self.specimenIndices = np.asarray(specimenIndices)
        self.outputBarIndices = np.asarray(outputBarIndices)

        # accumulated plastic strain, one entry per specimen element
        self.eps_plastic = np.zeros(len(self.specimenIndices), float)

        # uniform mass per node
        self.m[:] = self.rho * self.A_bar * self.dx0


if __name__ == "__main__":
    simulator = SimulateDirectImpact()
