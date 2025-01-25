# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-11-28 17:17:43
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-24 15:03:59

############## MODULES IMPORTATION ###############
import numpy as np
import matplotlib.pyplot as plt
import viz_tools    #self-developed module that groups animation functions
import sys
import pickle

def clamp(y, lowerlimit=0.0, upperlimit=1.0):
    y = np.where(y < lowerlimit, lowerlimit, y)
    y = np.where(y > upperlimit, upperlimit, y)
    return y

def smootherstep(edge0, edge1, x):
    #Scale, and clamp x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0));
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)

def pulse(T, duration, risetime):
    return smootherstep(0.0, risetime, T) - (smootherstep(duration, duration + risetime, T) + 1) + 1


def computeStressStrainSpecimen(l):
    """
    Simple linear elastic - perfect plastic material behaviour
    l : current element length
    """

    yield_stress = 0.001
    eps = (l - dx0) / dx0
    eps -= eps_plastic
    stress = E_bar * eps

    #print("stress:", stress)

    for i in range(eps.shape[0]):
        if stress[i] < -yield_stress:
            dsig = stress[i] + yield_stress # stress is outside of yield surface by this amount
            #print("negative beyond yield stress by delta: ", dsig)
            deps = dsig / E_bar
            eps_plastic[i] += deps

        elif stress[i] > yield_stress:
            dsig = stress[i] - yield_stress # stress is outside of yield surface by this amount
            #print("positive beyond yield stress by delta: ", dsig)
            deps = dsig / E_bar
            eps_plastic[i] += deps

    eps = (l - dx0) / dx0
    eps -= eps_plastic
    stress = E_bar * eps

    return stress


L_inputbar  = 2000.0
L_outputbar = 2000.0
L_specimen  = 20.0
initial_velocity = 5.0
N_cycles = 2 # simulate this many wave-pingpongs

# strain gauge locations
locA = L_inputbar * 0.5
locB = L_inputbar + L_specimen + 0.5 * L_outputbar
locS = L_inputbar + 0.5 * L_specimen
print("strain gauge position A", locA)
print("strain gauge position B", locB)
print("strain gauge position S", locS)

L = L_inputbar + L_specimen + L_outputbar # Total length of bar
dx = 1.0 # discretisation (element) length
N_x = int(L/dx) # number of elements
x = np.linspace(0,L,N_x+1) # nodal position
v = np.zeros_like(x) # nodal velocity
dx0 = x[1] - x[0] # initial length of each spring

# indices of specimen nodes
specimenIndices = []
inputBarIndices = []
for i in range(N_x + 1):
    pos = i * dx
    if pos <= L_inputbar:
        inputBarIndices.append(i)
    elif L_inputbar < pos <= L_inputbar + L_specimen:
        #print("specimen position:", pos)
        specimenIndices.append(i)
inputBarIndices = np.asarray(inputBarIndices)
specimenIndices = np.asarray(specimenIndices)

print("input bar indices:", inputBarIndices.min(), inputBarIndices.max())
print("specimen indices:", specimenIndices.min(), specimenIndices.max())
#sys.exit()


eps_plastic = np.zeros(len(specimenIndices), float) # array which holds the rest length of specimen elements




damping = 0.02 # artificial viscosity
bar_diameter = 40.0
rho = 1.2e-6
A_bar = 0.25 * np.pi * bar_diameter**2
mass_per_length = rho * A_bar # this is the mass per unit length (1 mm)
nodal_mass = mass_per_length * dx0
print("nodal mass:", nodal_mass)
E_bar = 2.4
E_specimen = 0.1
c0 = np.sqrt(E_bar/rho)
print("c0:", c0)

# check total mass

total_volume = L * 0.25 * np.pi * bar_diameter**2
total_mass  = rho * total_volume
nodal_mass = total_mass / N_x
print(f"total mass is {total_mass}")
print(f"nodal mass is {nodal_mass}")


#sys.exit()


#Temporal mesh with CFL < 1 - j indices
endTime = N_cycles * 2 * L_inputbar / c0 #
dt = 0.5 * dx0 / c0  # timestep
num_timesteps = int(endTime/dt) #Points number of the temporal mesh
T = np.linspace(0,endTime,num_timesteps) #Temporal array

history_nodal = np.zeros((N_x+1,num_timesteps),float) #Global solution
history_elems_force = np.zeros((N_x,num_timesteps),float) # This is the force on the cross-section of each element
history_elems_stress = np.zeros((N_x,num_timesteps),float) # This is the stress on each element
history_elems_strain = np.zeros((N_x,num_timesteps),float) # This is the strain on each element

f = np.zeros_like(x) # force

# apply initial velocity to left bar
v[inputBarIndices] = initial_velocity

print(f"simulation duration is {endTime}, number of time steps is {num_timesteps}")
for i in range(num_timesteps):

    # 1st part of leapfrog: update velocities half-step using old accelerations
    v +=  0.5 * f * dt / nodal_mass

    # second part of leapfrog: update positions
    x += dt * v
    # compute force between springs
    # we have N_x + 1 nodes and N_x elements
    dx = x[1:] - x[0:-1]
    eps = (dx - dx0) / dx0
    stress = E_bar * eps # everything is treated as bar materials
    
    stress[specimenIndices] = computeStressStrainSpecimen(dx[specimenIndices])
     
    dv = v[1:] - v[0:-1]
    stress += damping * rho * c0 * dv # artificial viscosity

    f[:] = 0.0
    f[:-1] = stress * A_bar
    f[1:] -= stress * A_bar

    # third part of leapfrog: update velocities with new accelerations
    v += 0.5 * f * dt / nodal_mass

    history_nodal[:,i] = v # save velocity for later retrieval in history array
    history_elems_force[:,i] = A_bar * stress
    history_elems_stress[:,i] = stress
    history_elems_strain[:,i] = eps


idxA = int(locA / dx0)
idxB = int(locB / dx0)
idxS = int(locS / dx0)
print(f"location of strain gauges: {idxA} -- {idxS} -- {idxB}")

epsA, epsS, epsB = history_elems_strain[idxA,:], history_elems_strain[idxS,:], history_elems_strain[idxB,:]
sigA, sigS, sigB = history_elems_stress[idxA,:], history_elems_stress[idxS,:], history_elems_stress[idxB,:]
forceA, forceS, forceB = history_elems_force[idxA,:], history_elems_force[idxS,:], history_elems_force[idxB,:]

# interpolate nodal velocities to strain gauge locations
# N_i ---- Element_i ----- N_i+1
velA = 0.5 * (history_nodal[idxA,:] + history_nodal[idxA + 1,:])
velB = 0.5 * (history_nodal[idxB,:] + history_nodal[idxB + 1,:])
velS = 0.5 * (history_nodal[idxS,:] + history_nodal[idxS + 1,:])


# compute average specimen strain
print("specimen indices:", specimenIndices)
epsS = np.mean(history_elems_strain[specimenIndices,:], axis=0)



fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)
#fig.suptitle('')

axs[0].set_ylabel("force / kN")
axs[0].plot(T, forceA, label="A")
axs[0].plot(T, forceS, label="S")
axs[0].plot(T, forceB, label="B")
axs[0].legend()

axs[1].set_ylabel("strain / [-]")
axs[1].plot(T, epsA, label="A")
axs[1].plot(T, epsS, label="S")
axs[1].plot(T, epsB, label="B")
axs[1].legend()

axs[2].set_ylabel("stress / [GPa]")
#axs[2].plot(T, sigA, label="A")
#axs[2].plot(T, sigS, label="S")
#axs[2].plot(T, sigB, label="B")
axs[2].plot(T, velA, label="vel A")
axs[2].legend()




#axs[1].plot(self.t, self.P, "--", label="P")
#axs[1].plot(self.t, self.v, "--", label="v")
#axs[1].legend()

#axs[2].plot(self.t, self.P_shifted, "--", label="P shifted")
#axs[2].plot(self.t, self.v_shifted, "--", label="v_shifted")
#axs[2].legend()


plt.show()


# write signals at A, B C to file
data = np.column_stack((T, epsA, velA, epsB, velB))
np.savetxt("eps_vel.dat", data)
print("... wrote strain and velocity signals to file eps_vel.dat")

# write specimen stress and strain to file
data = np.column_stack((T, sigS, epsS))
np.savetxt("specimen.dat", data)
print("... wrote specimen stress and strain to file specimen.dat")

# write strain gauge locations to file
pickle.dump( (locA, locB, locS), open( "strain_gauge_locations.p", "wb" ) )
print("... pickled strain locations ABS to file")


#sys.exit()

# force level of input bar
force_max = 0.5 * rho * c0 * initial_velocity * A_bar

#anim1 = viz_tools.anim_1D(x,history_nodal, dt, 20, save = False , myxlim = (0, L) , myylim = (-1.0,1.0))
#anim1 = viz_tools.anim_1D(x[:-1], -history_elems, dt, 20, save = False , myxlim = (0, L) , myylim = (-1.1*force_max, 1.1*force_max))
#plt.show()