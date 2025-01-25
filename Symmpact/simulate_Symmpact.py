# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-24 16:35:10
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-25 17:07:26

import numpy as np
import matplotlib.pyplot as plt
import viz_tools    #self-developed module that groups animation functions
from progressbar import progressbar

class SimulateSymmpact:
    def __init__(self):

        # specimen properties
        self.specimen_dia = 10.0
        self.specimen_cross_section_area = 0.25 * np.pi * self.specimen_dia**2
        self.specimen_E = 70.0
        self.JC_A = 0.08 # Johnson-Cook A
        self.JC_B = 0.08 # Johnson-Cook B
        self.JC_n = 0.3  # Johnson-Cook n

        # set up initial conditions : bar
        self.distA = 130.0 # distance of strain gauge on  input bar, measured from bar/specimen interface
        self.distB = 130.0 # distance of strain gauge on output bar, measured from bar/specimen interface
        self.L_inputbar  = 2000.0
        self.L_outputbar = 2000.0
        self.L_specimen  = 10.0
        self.E_bar = 70.0
        self.rho = 2.7e-6
        self.diameter_bar = 40.0
        self.A_bar = 0.25 * np.pi * self.diameter_bar**2
        self.damping = 0.01
        self.initial_velocity = 5.0
        self.N_cycles = 4 # simulate this many wave-pingpongs in a bar

        self.initialize_spatial_mesh()
        self.initialize_time_discretization()
        self.initialize_history_arrays()
        self.apply_initial_conditions()

        self.integrate_time()

        self.extract_discrete_results()
        #self.animate_results()
        

        # ... iterate over timesteps
        #self.

        # ... write out data

        # ... plot results

    def integrate_time(self):
        """
        Perform explicit time integration
        """
        print(f"simulation duration is {self.T[-1]}, number of time steps is {self.num_timesteps}")
        print("specimen cross section area:", self.specimen_cross_section_area)
        print("bar cross section area:", self.A_bar)
        print("rho * c0 * Up:", 0.5 * self.rho * self.c0 * self.initial_velocity)
        for i in progressbar(range(self.num_timesteps)):

            # 1st part of leapfrog: update velocities half-step using old accelerations
            self.v +=  0.5 * self.f * self.dt / self.nodal_mass

            # second part of leapfrog: update positions
            self.x += self.dt * self.v
            # compute force between springs
            # we have N_x + 1 nodes and N_x elements
            dx = self.x[1:] - self.x[0:-1]
            eps = (dx - self.dx0) / self.dx0

            # purely elastic stress update of everything
            stress = self.E_bar * eps # initially, everything is treated as bar materials

            # now, overwrite stress for the specimen elements only
            stress[self.specimenIndices] = self.computeStressStrainSpecimen(dx[self.specimenIndices])

            # enforce no-tension condition on specimen interface elements of both bars
            stress[self.inputBarIndices[-1]] = min(stress[self.inputBarIndices[-1]], 0.0)
            stress[self.outputBarIndices[0]] = min(stress[self.outputBarIndices[0]], 0.0)

            # add artificial viscosity
            dv = self.v[1:] - self.v[0:-1]
            stress += self.damping * self.rho * self.c0 * dv # artificial viscosity

            # create element force array
            element_forces = stress * self.A_bar # ... initially, treat everything as bar material
            element_forces[self.specimenIndices] = stress[self.specimenIndices] * self.specimen_cross_section_area # ... and overwrite specimen elements

            self.f[:] = 0.0
            self.f[:-1] = element_forces # ... distribute stress as nodal forces
            self.f[1:] -= element_forces

            # third part of leapfrog: update velocities with new accelerations
            self.v += 0.5 * self.f * self.dt / self.nodal_mass

            self.history_nodal_vel[:,i] = self.v # save velocity for later retrieval in history array
            self.history_elems_force[:,i] = element_forces
            self.history_elems_stress[:,i] = stress
            self.history_elems_strain[:,i] = eps

    def computeStressStrainSpecimen(self, l):
        """
        Simple linear elastic - perfect plastic material behaviour
        l : array of current element lengths
        """
        
        eps = (l - self.dx0) / self.dx0
        eps -= self.eps_plastic

        # compute state-dependent yield stress
        yield_stress = self.JC_A + self.JC_B * abs(self.eps_plastic)**self.JC_n
        stress = self.specimen_E * eps # predict stress

        #print("stress:", stress)

        for i in range(len(eps)): # compression
            if stress[i] < -yield_stress[i]:
                dsig = stress[i] + yield_stress[i] # stress is outside of yield surface by this amount
                #print(f"negative beyond :: stress={stress[i]}, yield stress={yield_stress[i]}, delta={dsig}")
                stress[i] = -yield_stress[i]
                #print("negative beyond yield stress by delta: ", dsig)
                deps = dsig / self.E_bar
                self.eps_plastic[i] += deps

        #    elif stress[i] > yield_stress: # tension
        #        dsig = stress[i] - yield_stress # stress is outside of yield surface by this amount
        #        #print("positive beyond yield stress by delta: ", dsig)
        #        deps = dsig / self.E_bar
        #        self.eps_plastic[i] += deps


        # can we vectorize the above loop?
        # first, treat the case where stress[i] < -yield_stress, i.e., compression
        #dsig = np.where(stress < -yield_stress, stress + yield_stress, 0.0)
        #deps = dsig / self.specimen_cross_section_area
        #self.eps_plastic += deps

        return stress


    def extract_discrete_results(self):
        self.epsA,   self.epsB   = self.history_elems_strain[self.idxA,:], self.history_elems_strain[self.idxB,:]
        self.sigA,   self.sigB   = self.history_elems_stress[self.idxA,:], self.history_elems_stress[self.idxB,:]
        self.forceA, self.forceB = self.history_elems_force[self.idxA,:], self.history_elems_force[self.idxB,:]

        # interpolate nodal velocities to strain gauge locations
        # N_i ---- Element_i ----- N_i+1
        velA = 0.5 * (self.history_nodal_vel[self.idxA,:] + self.history_nodal_vel[self.idxA + 1,:])
        velB = 0.5 * (self.history_nodal_vel[self.idxB,:] + self.history_nodal_vel[self.idxB + 1,:])

        # compute average specimen strain
        #print("specimen indices:", specimenIndices)
        self.epsS   = np.mean(self.history_elems_strain[self.specimenIndices,:], axis=0)
        self.sigS   = np.mean(self.history_elems_stress[self.specimenIndices,:], axis=0)
        self.forceS = np.mean(self.history_elems_force[self.specimenIndices,:], axis=0)

        # save stuff
        data = np.column_stack((self.T, self.epsA, velA, self.epsB, velB))
        np.savetxt("eps_vel.dat", data)
        print("... wrote strain and velocity signals to file eps_vel.dat")

        # write specimen stress and strain to file
        data = np.column_stack((self.T, self.sigS, self.epsS))
        np.savetxt("specimen.dat", data)
        print("... wrote specimen stress and strain to file specimen.dat")

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(1200*px, 800*px))
        gs = fig.add_gridspec(2, hspace=1)
        axs = gs.subplots(sharex=False, sharey=False)

        axs[0].plot(self.T, self.forceA, label="force@A")
        axs[0].plot(self.T, self.forceB, label="force@B")
        axs[0].plot(self.T, self.forceS, label="force@S")

        #axs[0].plot(self.T, self.sigA, label="sig@A")
        #axs[0].plot(self.T, self.sigB, label="sig@B")
        #axs[0].plot(self.T, self.sigS, label="sig@S")

        axs[0].legend()
        axs[0].set_xlabel("time")

        axs[1].plot(-self.epsS, -self.sigS, label = "specimen stress/strain")
        axs[1].legend()
        axs[1].set_xlabel("strain")
        
        plt.show()


    def animate_results(self):
        """
        """
        # force level of input bar
        force_max = 0.5 * self.rho * self.c0 * self.initial_velocity * self.A_bar

        anim1 = viz_tools.anim_1D(self.x[:-1], -self.history_elems_force, self.dt, 20, save = False , myxlim = (0, self.L) , myylim = (-1.1*force_max, 1.1*force_max))
        plt.show()
        
    def apply_initial_conditions(self):
        
        # apply initial velocity to left bar
        self.v[self.inputBarIndices] = self.initial_velocity

    def initialize_history_arrays(self):
        """
        Initialize the history variables used to save simulation results.
        """
        self.history_nodal_vel =        np.zeros((self.N_x+1, self.num_timesteps),float) # Global solution
        self.history_elems_force =  np.zeros((self.N_x,   self.num_timesteps),float) # This is the force on the cross-section  of each element
        self.history_elems_stress = np.zeros((self.N_x,   self.num_timesteps),float) # This is the stress  on each element
        self.history_elems_strain = np.zeros((self.N_x,   self.num_timesteps),float) # This is the strain  on each element

    def initialize_time_discretization(self):
        self.c0 = np.sqrt(self.E_bar/self.rho)
        print("bar c0:", self.c0)
        
        endTime = self.N_cycles * 2 * min(self.L_inputbar, self.L_outputbar) / self.c0 #
        self.dt = 0.5 * self.dx0 / self.c0  # timestep
        self.num_timesteps = int(endTime/self.dt) #Points number of the temporal mesh
        self.T = np.linspace(0,endTime,self.num_timesteps) #Temporal array

    def initialize_spatial_mesh(self):

        # strain gauge locations
        locA = self.L_inputbar * 0.5
        locB = self.L_inputbar + self.L_specimen + 0.5 * self.L_outputbar
        print("strain gauge position A", locA)
        print("strain gauge position B", locB)

        self.L = self.L_inputbar + self.L_specimen + self.L_outputbar # Total length of bar
        dx = 1.0 # discretisation (element) length
        self.N_x = int(self.L/dx) # number of elements
        self.x = np.linspace(0,self.L,self.N_x+1) # nodal position
        self.v = np.zeros_like(self.x) # nodal velocity
        self.f = np.zeros_like(self.x) # nodal forces
        self.dx0 = self.x[1] - self.x[0] # initial length of each spring

        # indices of input bar, specimen, ouput bar nodes
        specimenIndices = []
        inputBarIndices = []
        outputBarIndices = []
        for i in range(self.N_x + 1):
            pos = i * dx
            if pos <= self.L_inputbar:
                inputBarIndices.append(i)
            elif self.L_inputbar < pos <= self.L_inputbar + self.L_specimen:
                #print("specimen position:", pos)
                specimenIndices.append(i)
            else:
                outputBarIndices.append(i)

        self.inputBarIndices = np.asarray(inputBarIndices)
        self.specimenIndices = np.asarray(specimenIndices)
        self.outputBarIndices = np.asarray(outputBarIndices)

        self.eps_plastic = np.zeros(len(self.specimenIndices), float) # array which holds the rest length of specimen elements

        print("input bar indices:", self.inputBarIndices.min(), self.inputBarIndices.max())
        print("specimen indices:", self.specimenIndices.min(), self.specimenIndices.max())
        print("output bar indices:", self.outputBarIndices.min(), self.outputBarIndices.max())

        # indices of elements corresponding to strain gauges
        locA = self.L_inputbar - self.distA
        locB = self.L_inputbar + self.L_specimen + self.distB
        self.idxA = int(locA / self.dx0)
        self.idxB = int(locB / self.dx0)
        print(f"location of strain gauges: {locA} -- {locB}")
        print(f"indices  of strain gauges: {self.idxA} -- {self.idxB}")



        # mass per node
        mass_per_length = self.rho * self.A_bar # this is the mass per unit length (1 mm)
        self.nodal_mass = mass_per_length * self.dx0
        print("nodal mass:", self.nodal_mass)

    def initialize_variables(self):
        #Temporal mesh with CFL < 1 - j indices
        self.endTime = self.N_cycles * 2 * self.L_inputbar / self.c0 #
        self.dt = 0.5 * dx0 / self.c0  # timestep
        num_timesteps = int(self.endTime/self.dt) #Points number of the temporal mesh
        self.T = np.linspace(0,self.endTime,num_timesteps) #Temporal array

        self.history_nodal_vel = np.zeros((N_x+1,num_timesteps),float) #Global solution
        self.history_elems_force = np.zeros((N_x,num_timesteps),float) # This is the force on the cross-section of each element
        self.history_elems_stress = np.zeros((N_x,num_timesteps),float) # This is the stress on each element
        self.history_elems_strain = np.zeros((N_x,num_timesteps),float) # This is the strain on each element

        self.force = np.zeros_like(x) # force

        # apply initial velocity to left bar
        self.v[inputBarIndices] = initial_velocity

    


if __name__ == "__main__":
    simulator = SimulateSymmpact()
