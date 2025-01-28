# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:09:10 2020

@author: gcg

convert velocity measurement on putput bar to force

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import interpolate
import os
import sys
import bottleneck as bn
from scipy.interpolate import make_smoothing_spline

path = "/home/gcg/Projekte/21_WaveSeparation/2024-01-27_Waveseparation/01"

A = np.pi*20*20
rho = 1.21e-6
c0 = 1430.0

data = np.genfromtxt(os.path.join(path, "linescan_analysis.dat"))
time = data[:,0]
u = data[:,1]
u_subpixel = data[:,2]

#plt.plot(u)
#plt.plot(u - u_subpixel, label="subpixel")
#plt.legend()
#plt.show()
#sys.exit()

lam = 8.0e-6
spl = make_smoothing_spline(time, u, lam=lam)
linescan_force = -spl(time, nu=1)*0.16

data = np.genfromtxt(os.path.join(path, "Symmpact_time_force.txt"))
force_time, force = data[:,0], data[:,1]
plt.plot(force_time, force, label="DMS force")
#plt.plot(time, u)
plt.plot(time, linescan_force, label="spline")
plt.xlim(10,14)
plt.show()
sys.exit()

#u = bn.move_mean(data[:,1], window=10, min_count=1 )
v = np.diff(u, prepend=1) * 10
#v = np.diff(u) * 100


#plt.plot(time, u, label="linescan displacement")
#plt.plot(time, v, label="linescan velocity")
#plt.show()
#sys.exit()


#force = v * c0 * rho * A

#np.savetxt(os.path.join(path, "linescan_force.dat"), np.column_stack((time, force)))



plt.plot(time, v, label="linescan force")

plt.legend()
plt.show()


