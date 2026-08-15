# -*- coding: utf-8 -*-
"""
1D Split Hopkinson TENSION Bar (SHTB) with a tubular striker.

Companion to simulate_compression.py, which is a direct-impact COMPRESSION bar. Same lumped
mass-spring chain, same explicit leapfrog, same output contract -- so drive_compression.py's
sibling drive_tension.py produces .npy dumps the existing reduction scripts read
without modification.

--------------------------------------------------------------------------
How a striker makes a tensile pulse
--------------------------------------------------------------------------
This is the Nicholas-type arrangement. The input bar carries an anvil (a steel
disc) at its far end, and a HOLLOW striker tube rides on the bar between
specimen and anvil. The striker is launched AWAY from the specimen and strikes
the anvil from the specimen side. The impact drives the anvil further away, the
bar is dragged after it, and a TENSILE pulse runs from the anvil to the
specimen.

    striker: POM tube, 16.1/40.0 mm dia, 800 mm, velocity -v0
              |==================|
    [#####]===+==================+=======[spec]=======[============]
     anvil     (rides on the bar)      input bar        output bar
     steel                              7075 Al          7075 Al
     40 dia                             16 dia           16 dia
     20 long                            3000 long        3000 long
    x=0    x=20                                                x=L
    <-- struck this way

Because the striker is a tube surrounding the bar, striker and bar occupy the
same range of x. They are therefore modelled as two mechanically independent
chains, coupled by a single unilateral contact at the anvil's rear face: the
striker can push the anvil, never pull it, and the two separate when the striker
rebounds. That contact is the only thing they share.

The anvil is rigidly connected to the input bar -- they share a node -- so it is
simply the leftmost 20 elements of the bar chain, with steel properties. Its
mass (0.197 kg, 12% of the input bar's) is what the striker actually has to
accelerate, and it is large enough to matter.

The pulse length is set by the striker: the compressive wave in it runs to its
free end and back in 2*L_striker/c_striker, which is how long it keeps pushing
the anvil. Note this uses the STRIKER's wave speed, not the bar's. POM is slow
(c = 1459 mm/ms against 5051 in the aluminium), so an 800 mm POM tube gives a
1097 us pulse -- roughly 3.5x longer than the same tube in aluminium would.

--------------------------------------------------------------------------
Differences from simulate_compression.py that matter downstream
--------------------------------------------------------------------------
* The specimen is BONDED to both bars (threaded, as a real tension specimen is),
  so the no-tension conditions at the two specimen interfaces are gone. The
  unilateral condition now lives at the striker/anvil contact instead.
* The specimen yields in TENSION, so the return mapping is mirrored.
* Strain and force are POSITIVE in tension. Pass loading='tension' to
  wave_separation.specimen_response when reducing this data.
* The input bar's gauge sees the incident and reflected waves genuinely
  overlapping, which the direct-impact model does not produce. That is the point
  of having this simulator.
* The chain is NOT uniform: steel anvil, aluminium bars, soft specimen. Element
  stiffness, area and density are therefore per-element arrays, and nodal masses
  are assembled as half of each adjacent element's mass. Getting that wrong at
  the steel/aluminium junction would give the wrong reflection there.

Units: mm, ms, kg  =>  kN, GPa, and mm/ms (numerically equal to m/s).
"""

import numpy as np

import config as _config
from recording import GaugeRecorder


class SimulateSHTB:
    """
    Parameters come from the [tension] case of config.toml -- see that file for
    what each one means. Pass a config dict to override, e.g. for a sweep:

        cfg = config.load('tension')
        cfg['striker']['length'] = 1500.0
        sim = SimulateSHTB(cfg)
    """

    def __init__(self, cfg=None):
        cfg = _config.load('tension') if cfg is None else cfg
        self.cfg = cfg
        bar, spec = cfg['bar'], cfg['specimen']
        strk, anv, num = cfg['striker'], cfg['anvil'], cfg['numerics']

        # specimen properties (tension)
        self.specimen_dia = spec['diameter']
        self.specimen_cross_section_area = 0.25 * np.pi * self.specimen_dia**2
        self.specimen_E = spec['E']
        self.rho_specimen = spec['rho']
        self.JC_A, self.JC_B, self.JC_n = spec['JC_A'], spec['JC_B'], spec['JC_n']
        # plastic strain at which the specimen fails; absent from the TOML
        # (which has no null) means failure is disabled
        self.failure_strain = spec.get('failure_strain')

        # --- materials -----------------------------------------------------
        self.E_bar, self.rho_bar = bar['E'], bar['rho']          # 7075-T6 Al
        self.E_pom, self.rho_pom = strk['E'], strk['rho']        # POM striker
        self.E_anvil, self.rho_anvil = anv['E'], anv['rho']      # steel anvil

        # --- geometry ------------------------------------------------------
        self.L_inputbar = bar['L_input']
        self.L_outputbar = bar['L_output']
        self.L_specimen = spec['length']
        self.L_anvil = anv['length']
        # Striker length sets the pulse length (2*L_striker/c_striker) and,
        # through it, how well the specimen reaches force equilibrium. The
        # length/equilibrium table that used to sit here was measured for an
        # impedance-matched ALUMINIUM striker on 2000 mm bars and no longer
        # applies: POM is 3.5x slower, so the same 800 mm gives a 1097 us pulse
        # instead of 314 us. Re-measure before quoting numbers.
        self.L_striker = strk['length']
        self.diameter_bar = bar['diameter']
        self.striker_id = strk['inner_diameter']
        self.striker_od = strk['outer_diameter']
        self.diameter_anvil = anv['diameter']
        self.A_bar = 0.25 * np.pi * self.diameter_bar**2
        self.A_striker = 0.25 * np.pi * (self.striker_od**2 - self.striker_id**2)
        self.A_anvil = 0.25 * np.pi * self.diameter_anvil**2

        # --- numerics ------------------------------------------------------
        self.courant = num['courant']
        self.damping = num['damping']
        self.N_cycles = num['ncyc']
        self.dx_target = num['dx']
        self.record_full_field = num['record_full_field']
        self.gauge_distances = list(cfg['gauges'])

        self.striker_velocity = strk['velocity']
        # both bars start at rest: only the striker moves
        self.v0_in = self.v0_out = 0.0
        self.loading = cfg['loading']

        self.initialize_spatial_mesh()
        self.initialize_time_discretization()
        self.initialize_history_arrays()
        self.apply_initial_conditions()

        self.integrate_time()

        self.extract_discrete_results()

    # ------------------------------------------------------------------ setup

    def initialize_spatial_mesh(self):
        # The anvil is rigidly attached to the input bar, so it is part of the
        # same chain: x = 0 is now the anvil's OUTER face, not the bar's end.
        self.L = (self.L_anvil + self.L_inputbar + self.L_specimen
                  + self.L_outputbar)
        dx = self.dx_target      # element length, from config.toml
        self.N_x = int(round(self.L / dx))
        self.x = np.linspace(0, self.L, self.N_x + 1)
        self.v = np.zeros_like(self.x)
        self.f = np.zeros_like(self.x)
        self.dx0 = self.x[1] - self.x[0]

        # region boundaries along x
        x_anvil1 = self.L_anvil                        # anvil | input bar
        x_spec0 = x_anvil1 + self.L_inputbar           # input bar | specimen
        x_spec1 = x_spec0 + self.L_specimen            # specimen | output bar

        # Regions are assigned by ELEMENT CENTRE. simulate_compression.py builds these as
        # node indices and then uses them on element arrays, which puts its
        # specimen one element to the right of nominal; that shortcut is not
        # available here because the material properties, not just the labels,
        # hang off these indices. Element i spans [i, i+1]*dx.
        centres = (np.arange(self.N_x) + 0.5) * dx
        self.anvilIndices = np.flatnonzero(centres < x_anvil1)
        self.inputBarIndices = np.flatnonzero((centres >= x_anvil1)
                                              & (centres < x_spec0))
        self.specimenIndices = np.flatnonzero((centres >= x_spec0)
                                              & (centres < x_spec1))
        self.outputBarIndices = np.flatnonzero(centres >= x_spec1)

        # per-element material properties; bar everywhere, then overwrite
        self.E_elem = np.full(self.N_x, self.E_bar)
        self.A_elem = np.full(self.N_x, self.A_bar)
        self.rho_elem = np.full(self.N_x, self.rho_bar)
        for idx, (Ee, Ae, re) in (
                (self.anvilIndices, (self.E_anvil, self.A_anvil, self.rho_anvil)),
                (self.specimenIndices, (self.specimen_E,
                                        self.specimen_cross_section_area,
                                        self.rho_specimen))):
            self.E_elem[idx], self.A_elem[idx], self.rho_elem[idx] = Ee, Ae, re

        # artificial viscosity acts through the local impedance rho*c, and
        # rho*c == sqrt(E*rho). Per element, since the chain is not uniform.
        self.Zc_elem = np.sqrt(self.E_elem * self.rho_elem)

        # Nodal masses: half of each adjacent element's mass, so a node on a
        # material junction gets the correct average and the free ends get half.
        # (simulate_compression.py can give every node a full element mass because its chain
        # is uniform; here that would put the wrong inertia on the steel node.)
        m_elem = self.rho_elem * self.A_elem * self.dx0
        self.m = np.zeros_like(self.x)
        self.m[:-1] += 0.5 * m_elem
        self.m[1:] += 0.5 * m_elem

        # Striker: its own chain, a tube riding on the bar, so it is spatially
        # coincident with the first L_striker mm of the input bar and couples to
        # the rest of the model only through the anvil contact. It starts flush
        # against the anvil's rear face.
        self.N_s = int(round(self.L_striker / dx))
        self.xs = np.linspace(x_anvil1, x_anvil1 + self.L_striker, self.N_s + 1)
        self.vs = np.zeros_like(self.xs)
        self.fs = np.zeros_like(self.xs)
        ms_elem = self.rho_pom * self.A_striker * self.dx0
        self.ms = np.zeros_like(self.xs)
        self.ms[:-1] += 0.5 * ms_elem
        self.ms[1:] += 0.5 * ms_elem
        self.Zc_striker = np.sqrt(self.E_pom * self.rho_pom)

        # the bar-chain node the striker hits: the anvil's rear (specimen-side)
        # face, which is also the anvil/input-bar junction
        self.anvil_face_node = int(round(x_anvil1 / dx))

        self.eps_plastic = np.zeros(len(self.specimenIndices), float)
        self.failed = np.zeros(len(self.specimenIndices), bool)

        # contact stiffness: the striker's own element stiffness, so the
        # closed contact simply continues the chain and adds no stiffer mode
        # (and therefore does not reduce the stable timestep). POM is soft, so
        # this is far below the steel stiffness already at that node.
        self.k_contact = self.E_pom * self.A_striker / self.dx0

    def initialize_time_discretization(self):
        # c0 is the BAR wave speed -- it is what drive_tension.py writes to
        # meta.npz and what the reduction uses. The striker and anvil have their
        # own, and the fastest of the three sets the stable timestep.
        self.c0 = np.sqrt(self.E_bar / self.rho_bar)
        self.c_striker = np.sqrt(self.E_pom / self.rho_pom)
        self.c_anvil = np.sqrt(self.E_anvil / self.rho_anvil)
        self.pulse_duration = 2 * self.L_striker / self.c_striker
        c_max = max(self.c0, self.c_striker, self.c_anvil)
        endTime = self.N_cycles * 2 * min(self.L_inputbar, self.L_outputbar) / self.c0
        self.dt = self.courant * self.dx0 / c_max
        self.num_timesteps = int(endTime / self.dt)
        self.T = np.arange(self.num_timesteps) * self.dt

    def initialize_history_arrays(self):
        # Only the gauge rows, the two interface elements and the specimen means
        # are kept; see recording.py. The full field is opt-in via config.toml.
        self.rec = GaugeRecorder(
            self.specimenIndices, self.dx0, self.L, self.gauge_distances,
            self.num_timesteps, self.N_x,
            record_full_field=self.record_full_field)

    def apply_initial_conditions(self):
        # striker launched toward the anvil, i.e. in -x, away from the specimen
        self.vs[:] = -self.striker_velocity

    # -------------------------------------------------------------- integrate

    def integrate_time(self):
        print(f"SHTB: POM striker {self.L_striker:.0f} mm (c={self.c_striker:.0f} "
              f"mm/ms) -> pulse {self.pulse_duration*1e3:.1f} us; "
              f"{self.num_timesteps} steps of {self.dt*1e3:.4f} us")

        for i in range(self.num_timesteps):

            # --- leapfrog: half-step velocities, then positions
            self.v += 0.5 * self.f * self.dt / self.m
            self.vs += 0.5 * self.fs * self.dt / self.ms
            self.x += self.dt * self.v
            self.xs += self.dt * self.vs

            # --- bar internal forces (anvil + bars + specimen, one chain)
            dx = self.x[1:] - self.x[0:-1]
            eps = (dx - self.dx0) / self.dx0
            stress = self.E_elem * eps
            stress[self.specimenIndices] = self.computeStressStrainSpecimen(
                dx[self.specimenIndices])
            # NOTE: no unilateral condition at the specimen interfaces -- a
            # tension specimen is threaded into both bars and carries tension.
            dv = self.v[1:] - self.v[0:-1]
            stress += self.damping * self.Zc_elem * dv

            element_forces = stress * self.A_elem
            self.f[:] = 0.0
            self.f[:-1] = element_forces
            self.f[1:] -= element_forces

            # --- striker internal forces (POM)
            dxs = self.xs[1:] - self.xs[0:-1]
            eps_s = (dxs - self.dx0) / self.dx0
            stress_s = self.E_pom * eps_s
            dvs = self.vs[1:] - self.vs[0:-1]
            stress_s += self.damping * self.Zc_striker * dvs
            fs_elem = stress_s * self.A_striker
            self.fs[:] = 0.0
            self.fs[:-1] = fs_elem
            self.fs[1:] -= fs_elem

            # --- unilateral striker/anvil contact (compression only)
            # Both start flush at the anvil's rear face. The striker's leading
            # node pushes that face in -x; penetration is positive once it has
            # overtaken it.
            j = self.anvil_face_node
            penetration = self.x[j] - self.xs[0]
            if penetration > 0.0:
                Fc = self.k_contact * penetration
                self.f[j] -= Fc     # anvil driven away from the specimen
                self.fs[0] += Fc    # striker decelerated

            # --- leapfrog: second half-step
            self.v += 0.5 * self.f * self.dt / self.m
            self.vs += 0.5 * self.fs * self.dt / self.ms

            self.rec.record(i, eps, element_forces)

    def computeStressStrainSpecimen(self, l):
        """
        Elastic - plastic with Johnson-Cook style hardening, yielding in TENSION.
        l : array of current element lengths
        """
        eps = (l - self.dx0) / self.dx0 - self.eps_plastic
        yield_stress = self.JC_A + self.JC_B * abs(self.eps_plastic)**self.JC_n
        stress = self.specimen_E * eps

        over = stress > yield_stress
        if np.any(over):
            dsig = stress[over] - yield_stress[over]
            stress[over] = yield_stress[over]
            # return mapping uses the SPECIMEN modulus (simulate_compression.py uses the bar
            # modulus here, which is a bug preserved there for continuity)
            self.eps_plastic[over] += dsig / self.specimen_E

        if self.failure_strain is not None:
            self.failed |= self.eps_plastic > self.failure_strain
            stress[self.failed] = 0.0
        return stress

    # ----------------------------------------------------------------- output

    def extract_discrete_results(self):
        # the specimen means were accumulated during the run, not stored per
        # element, so there is nothing to average here any more
        self.epsS = self.rec.spec_strain
        self.forceS = self.rec.spec_force
        self.sigS = self.forceS / self.specimen_cross_section_area
        np.savetxt("specimen.dat", np.column_stack((self.T, self.sigS, self.epsS)),
                   header="time[ms]  mean specimen stress[GPa]  mean specimen strain[-]"
                          "  (TENSION POSITIVE)")
        print("... wrote specimen.dat")


if __name__ == "__main__":
    simulator = SimulateSHTB()
