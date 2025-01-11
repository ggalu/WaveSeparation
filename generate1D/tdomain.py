# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-11-28 17:17:43
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-02 10:04:54
""" 
This file was built to solve numerically a classical PDE, 1D wave equation. The equation corresponds to :
$\dfrac{\partial}{\partial x} \left( \dfrac{\partial c^2 U}{\partial x} \right) = \dfrac{\partial^2 U}{\partial t^2}$
 
where
 - U represent the signal
 - x represent the position
 - t represent the time
 - c represent the velocity of the wave (depends on space parameters)

The numerical scheme is based on finite difference method. This program is also providing several boundary conditions. More particularly the Neumann, Dirichlet and Mur boundary conditions.
Copyright - © SACHA BINDER - 2021
"""

############## MODULES IMPORTATION ###############
import numpy as np
import matplotlib.pyplot as plt
import viz_tools    #self-developed module that groups animation functions
import sys

def smoothclamp(x, mi, mx): return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

def clamp(y, lowerlimit=0.0, upperlimit=1.0):
    y = np.where(y < lowerlimit, lowerlimit, y)
    y = np.where(y > upperlimit, upperlimit, y)
    return y

def smootherstep(edge0, edge1, x):
    #Scale, and clamp x to 0..1 range
    x = clamp((x - edge0) / (edge1 - edge0));
    return -x * x * x * (x * (6.0 * x - 15.0) + 10.0) + 1



#Def of the initial condition    
def I(x):
    """
    Single space variable fonction that 
    represent the wave form at t = 0
    """
    return smootherstep(0.2, 0.4, x)




############## SET-UP THE PROBLEM ###############

#Def of velocity (spatial scalar field)
# def celer(x):
#     """
#     constant velocity
#     """
#     return 1

def celer(x):
    """
    Single space variable fonction that represent 
    the wave's velocity at a position x
    """
    return 1
    #if x <=1:
    #    return 1e
    #else:
    #    return 1
        
loop_exec = 1  # Processing loop execution flag

left_bound_cond = 2  #Boundary cond 1 : Dirichlet, 2 : Neumann, 3 Mur
right_bound_cond = 1  #Boundary cond 1 : Dirichlet, 2 : Neumann, 3 Mur

if left_bound_cond not in [1,2,3]:
    loop_exec = 0
    print("Please choose a correct left boundary condition")

if right_bound_cond not in [1,2,3]:
    loop_exec = 0
    print("Please choose a correct right boundary condition")



#Spatial mesh - i indices
L_x = 1.4 #Range of the domain according to x [m]
dx = 0.01 #Infinitesimal distance
N_x = int(L_x/dx) #Points number of the spatial mesh
X = np.linspace(0,L_x,N_x+1) #Spatial array



#Temporal mesh with CFL < 1 - j indices
L_t = 3.0 #Duration of simulation [s]
dt = 0.005*dx  #Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
N_t = int(L_t/dt) #Points number of the temporal mesh
T = np.linspace(0,L_t,N_t+1) #Temporal array



#Velocity array for calculation (finite elements)
c = np.zeros(N_x+1, float)
for i in range(0,N_x+1):
    c[i] = celer(X[i])




############## CALCULATION CONSTANTS ###############
c_1 = c[0]
c_2 = c[N_x]

C2 = (dt/dx)**2

CFL_1 = c_1*(dt/dx)
CFL_2 = c_2*(dt/dx)


#plt.plot(X[0:N_x+1], I(X[0:N_x+1]))
#plt.show()
#sys.exit()


############## PROCESSING LOOP ###############

if loop_exec:
    # $\forall i \in {0,...,N_x}$
    u_jm1 = np.zeros(N_x+1,float)   #Vector array u_i^{j-1}
    u_j = np.zeros(N_x+1,float)     #Vector array u_i^j
    u_jp1 = np.zeros(N_x+1,float)   #Vector array u_i^{j+1}
    
    q = np.zeros(N_x+1,float)
    q[0:N_x+1] = c[0:N_x+1]**2
    
    U = np.zeros((N_x+1,N_t+1),float) #Global solution
    
    #init cond - at t = 0
    u_j[0:N_x+1] = I(X[0:N_x+1])

    U[:,0] = u_j.copy()
    
    
    #init cond - at t = 1
    #without boundary cond
    u_jp1[1:N_x] =  u_j[1:N_x] + 0.5*C2*( 0.5*(q[1:N_x] + q[2:N_x+1])*(u_j[2:N_x+1] - u_j[1:N_x]) - 0.5*(q[0:N_x-1] + q[1:N_x])*(u_j[1:N_x] - u_j[0:N_x-1]))
    
    
    #left boundary conditions
    if left_bound_cond == 1:
        #Dirichlet bound cond
        u_jp1[0] = 0
        
    elif left_bound_cond == 2:
        #Nuemann bound cond
        #i = 0
        u_jp1[0] = u_j[0] + 0.5*C2*( 0.5*(q[0] + q[0+1])*(u_j[0+1] - u_j[0]) - 0.5*(q[0] + q[0+1])*(u_j[0] - u_j[0+1]))

    elif left_bound_cond == 3:
        #Mur bound cond
        #i = 0
        u_jp1[0] = u_j[1] + (CFL_1 -1)/(CFL_1 + 1)*( u_jp1[1] - u_j[0])

    
    
    #right boundary conditions
    if right_bound_cond == 1:
        #Dirichlet bound cond
        u_jp1[N_x] = 0
        
        
    elif right_bound_cond == 2:
        #Nuemann bound cond
        #i = N_x
        u_jp1[N_x] =  u_j[N_x] + 0.5*C2*( 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x-1] - u_j[N_x]) - 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x] - u_j[i-1]))
        
        
    elif right_bound_cond == 3:
        #Mur bound cond
        #i = N_x
        u_jp1[N_x] = u_j[N_x-1] + (CFL_2 -1)/(CFL_2 + 1)*(u_jp1[N_x-1] - u_j[N_x])
    
    u_jm1 = u_j.copy()  #go to the next step
    u_j = u_jp1.copy()  #go to the next step
    U[:,1] = u_j.copy()
    
    
    #Process loop (on time mesh)
    for j in range(1, N_t):
        #calculation at step j+1
        #without boundary cond
        u_jp1[1:N_x] = -u_jm1[1:N_x] + 2*u_j[1:N_x] + C2*( 0.5*(q[1:N_x] + q[2:N_x+1])*(u_j[2:N_x+1] - u_j[1:N_x]) - 0.5*(q[0:N_x-1] + q[1:N_x])*(u_j[1:N_x] - u_j[0:N_x-1]))
           
        
        #left bound conditions
        if left_bound_cond == 1:
            #Dirichlet bound cond
            u_jp1[0] = 0

        elif left_bound_cond == 2:
            #Nuemann bound cond
            #i = 0
            u_jp1[0] = -u_jm1[0] + 2*u_j[0] + C2*( 0.5*(q[0] + q[0+1])*(u_j[0+1] - u_j[0]) - 0.5*(q[0] + q[0+1])*(u_j[0] - u_j[0+1]))       
            
        elif left_bound_cond == 3:
            #Mur bound cond
            #i = 0
            u_jp1[0] = u_j[1] + (CFL_1 -1)/(CFL_1 + 1)*( u_jp1[1] - u_j[0])



        #right bound conditions
        if right_bound_cond == 1:
            #Dirichlet bound cond
            u_jp1[N_x] = 0
            
        elif right_bound_cond == 2:
            #Nuemann bound cond
            #i = N_x
            u_jp1[N_x] = -u_jm1[N_x] + 2*u_j[N_x] + C2*( 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x-1] - u_j[N_x]) - 0.5*(q[N_x-1] + q[N_x])*(u_j[N_x] - u_j[N_x-1]))
            
        elif right_bound_cond == 3:
            #Mur bound cond
            #i = N_x
            u_jp1[N_x] = u_j[N_x-1] + (CFL_2 -1)/(CFL_2 + 1)*(u_jp1[N_x-1] - u_j[N_x])

       
        
        u_jm1[:] = u_j.copy()   #go to the next step
        u_j[:] = u_jp1.copy()   #go to the next step
        U[:,j] = u_j.copy()


def shift(y, shift_time):
    """
    shift wave signal y by shift_time
    """
    offset = shift_time // dt
    y_shifted = np.roll(y, offset)
    return y_shifted


############## PLOT ###############
#U = np.zeros((N_x+1,N_t+1),float) #Global solution
print("length of bar:", N_x)
idxA = N_x // 3 + 20
idxB = N_x - 20
idxC = idxA + (idxB - idxA) // 2
locA = idxA * dx
locB = idxB * dx
locC = idxC * dx

print("strain gauge position 1 in %", locA, locA / L_x)
print("strain gauge position 2 in %", locB, locB / L_x)
print("strain gauge position C in %", locC, )

print("wave speed:", c_1)
delta = 2 * (locB - locA) / c_1
print("2 x travel time between the two strain gauges:", delta)

# time for reflection at loc1: forward trip + partial back-trip
# !!! forward travel time is not Nx, because bar is already populated with a wave!!!
# forward length of bar is 1.0


arrivalA = (locA - 0.4) / c_1 # time for 1st wave to arrive at A
t_refA = 2 * (L_x - locA) / c_1 # time for 1st wave to reflect back to A from right end
RA = arrivalA + t_refA
print("RA:", RA)


arrivalB = (locB - 0.4) / c_1 # time for 1st wave to arrive at B
t_refB = 2 * (L_x - locB) / c_1 # time for 1st wave to reflect back to B from right end
RB = arrivalB + t_refB
print("RB:", RB)

eps_A = U[idxA,:]
eps_B = U[idxB,:]
eps_C = U[idxC,:]

eps_A_asc0 = np.where(T < RA, eps_A, 0.0)
eps_B_dsc0 = np.zeros_like(T)


#
# SEGMENT 1
#
i = 1

# isolate waves at A and B.
eps_A_1 = np.where(np.logical_and(RA + (i - 2) * delta <= T, T < RA + (i - 1) * delta), eps_A, 0.0) #np.where(T <= RA, eps_A, 0.0)
eps_B_1 = np.where(np.logical_and(RB + (i - 1) * delta <= T, T < RB + i * delta), eps_B, 0.0)

# this is the ascending wave at A. It is known initially
eps_A_asc1 = np.where(np.logical_and(RA + (i - 2) * delta <= T, T < RA + (i - 1) * delta), eps_A, 0.0) #np.where(T <= RA, eps_A, 0.0)

# propagate the ascending wave from A to B
eps_B_asc1 = shift(eps_A_asc1, 0.500001 * delta)
# compute the descending wave at B
eps_B_dsc1 = eps_B_1 - eps_B_asc1



# plot isolated A waves
plt.plot(T, eps_A_asc0, label="eps_A_asc0")
#plt.plot(T, eps_A_1, label="eps_A_1")
plt.plot(T, eps_A_asc1, label="eps_A_asc1")

plt.plot(T, eps_B_1, label="eps_B_1")
plt.plot(T, eps_B_asc1, label="eps_B_asc1")
plt.plot(T, eps_B_dsc1, label="eps_B_dsc1")

#plt.plot(T, eps_A, "--", label="eps_A")
plt.plot(T, eps_B, "--", label="eps_B")
#plt.plot(T, eps_C, "--", label="eps_C")

# write single pulse at A to file
data = np.column_stack((T, eps_A_asc0))
np.savetxt("A0.dat", data)



plt.legend()
plt.show()
# write signals at A, B to file
data = np.column_stack((T, eps_A, eps_B, eps_C))
np.savetxt("ABC.dat", data)
sys.exit()


# plot isolated B waves
#plt.plot(T, eps_B_1, label="eps_B_1")
#plt.plot(T, eps_B_asc1, label="eps_B_asc1")
#plt.plot(T, eps_B, "--", label="eps_B")
#plt.plot(T, eps_B_dsc1, label="eps_B_dsc1")
#plt.plot(T, eps_B_dsc2, label="eps_B_dsc2")


#plt.plot(T, eps_B_asc2, label="eps_B_asc2")
#plt.plot(T, eps_B, label="B")

#plt.plot(T, eps_B_dsc1 + eps_B_dsc2, label="eps_B_dsc1")
#plt.plot(T, eps_B_dsc2, label="eps_B_dsc2")
plt.legend()



#plt.plot(T, U[loc1,:])
#plt.plot(T, U[loc2,:])
#plt.show()

#anim1 = viz_tools.anim_1D(X,U, dt, 20, save = False , myxlim = (0, 1.5) , myylim = (-0.5,1.5))
plt.show()