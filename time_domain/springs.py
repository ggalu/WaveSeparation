# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-11-28 17:17:43
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-09 15:04:34

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

    yield_stress = 0.01
    eps = (l - dx0) / dx0
    eps -= eps_plastic
    stress = E_bar * eps

    for i in range(eps.shape[0]):
        if stress[i] < -yield_stress:
            dsig = stress[i] + yield_stress # stress is outside of yield surface by this amount
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


L_inputbar  = 3000.0
L_outputbar = 0.0
L_specimen  = 0.0

# strain gauge locations
locA = 1500
locB = 2000
locC = 1500 #L_inputbar + L_specimen + 200
print("strain gauge position A", locA)
print("strain gauge position B", locB)
print("strain gauge position C", locC)

L = L_inputbar + L_specimen + L_outputbar # Total length of bar
dx = 1.0 # discretisation (element) length
N_x = int(L/dx) # number of elements
x = np.linspace(0,L,N_x+1) # nodal position
v = np.zeros_like(x) # nodal velocity
dx0 = x[1] - x[0] # initial length of each spring

# indices of specimen nodes
specimenIndices = []
for i in range(N_x + 1):
    pos = i * dx
    if L_inputbar < pos <= L_inputbar + L_specimen:
        #print("specimen position:", pos)
        specimenIndices.append(i)
specimenIndices = np.asarray(specimenIndices)
eps_plastic = np.zeros(len(specimenIndices), float) # array which holds the rest length of specimen elements


bar_diameter = 16.0
rho = 2.7e-6
A_bar = 0.25 * np.pi * bar_diameter**2
mass_per_length = rho * A_bar # this is the mass per unit length (1 mm)
nodal_mass = mass_per_length * dx0
print("nodal mass:", nodal_mass)
E_bar = 70.0
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
endTime = 2 * L_inputbar / c0 #
dt = 0.5 * dx0 / c0  # timestep
num_timesteps = int(endTime/dt) #Points number of the temporal mesh
T = np.linspace(0,endTime,num_timesteps) #Temporal array

striker_velocity = 10.0
striker_length = 200.0
striker_c0 = 1400.0
pulse_duration = 2 * striker_length / striker_c0
rise_time = 0.1 * pulse_duration
source = 0.5 * striker_velocity * pulse(T, pulse_duration - 2*rise_time, rise_time)
print("pulse duration is:", pulse_duration)
#plt.plot(T, source)
#plt.show()
#sys.exit()


# history arrays: these are sample at the integratin points, i.e. halfway between the nodes
velocity_history = np.zeros((N_x,num_timesteps),float) #Global solution
force_history = np.zeros((N_x,num_timesteps),float) #Global solution
eps_history = np.zeros((N_x,num_timesteps),float) #Global solution

f = np.zeros_like(x)

print(f"simulation duration is {endTime}, number of time steps is {num_timesteps}")
for i in range(num_timesteps):

    # 1st part of leapfrog: update velocities half-step using old accelerations
    v +=  0.5 * f * dt / nodal_mass

    # apply BC
    v[0] = source[i]

    # second part of leapfrog: update positions
    x += dt * v
    # compute force between springs
    # we have N_x + 1 nodes and N_x elements
    dx = x[1:] - x[0:-1]
    eps = (dx - dx0) / dx0
    eps_history[:,i] = eps
    stress = E_bar * eps # everything is treated as bar materials
    #print("specimen Indices:", specimenIndices)
    if len(specimenIndices) > 0:
        stress[specimenIndices] = computeStressStrainSpecimen(dx[specimenIndices])
     
    dv = v[1:] - v[0:-1]
    #stress += 0.02 * rho * c0 * dv # artificial viscosity

    f[:] = 0.0
    f[:-1] = stress * A_bar
    f[1:] -= stress * A_bar

    # third part of leapfrog: update velocities with new accelerations
    v += 0.5 * f * dt / nodal_mass

    # need to interpolate nodal velocity field to stress (integration) point position
    velocity_history[:,i] = 0.5 * (v[:-1] + v[1:]) # save velocity for later retrieval in history array
    
    force_history[:,i] = A_bar * stress


#print("10000 step is time:", 10000 * dt)
#plt.plot(X, U[:,10000])
#plt.show()

idxA = int(locA / dx0)
idxB = int(locB / dx0)
idxC = int(locC / dx0)

eps_A = eps_history[idxA,:]
eps_B = eps_history[idxB,:]
eps_C = eps_history[idxC,:]

vel_B = velocity_history[idxB,:]

plt.plot(T, eps_A, "--", label="eps_A")
#plt.plot(T, vel_A, "--", label="vel_A")
#plt.plot(T, eps_B, "--", label="eps_B")
#plt.plot(T, eps_C, "--", label="eps_C")
plt.legend()
plt.show()

# write signals at A, B C to file
data = np.column_stack((T, eps_A, eps_B, eps_C))
np.savetxt("ABC.dat", data)

# write CFC signals: 
data = np.column_stack((T, vel_B, eps_A))
np.savetxt("CFC.dat", data)


# write strain gauge locations to file
pickle.dump( (locA, locB, locC), open( "strain_gauge_locations.p", "wb" ) )

#sys.exit()

# force level of input bar
force_max = 0.5 * rho * c0 * striker_velocity * A_bar

#anim1 = viz_tools.anim_1D(x,history_nodal, dt, 20, save = False , myxlim = (0, L) , myylim = (-1.0,1.0))
#anim1 = viz_tools.anim_1D(x[:-1], -force_history, dt, 20, save = False , myxlim = (0, L) , myylim = (-1.1*force_max, 1.1*force_max))
#plt.show()