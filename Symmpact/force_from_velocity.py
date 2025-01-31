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

class CalibrateWaveSeparation():
    def __init__(self, path):

        self.A = np.pi * 20.0**2
        self.rho = 2.7e-6
        self.c0 = 5090.0
        self.Lbar = 2000 - 2*130

        self.additionalShift = -1.0e-2

        self.LoadDisplacementOverTime(path)
        self.LoadForceOverTime(path)
        self.Interpolate()

        
        self.CalibrateVelocity(calibration_factor=0.009444415150988306)

        
        #self.compute_F_G()
        
        self.plotForce()

    def compute_F_G(self):
        self.F = 0.5 * (self.eps + self.v / self.c0)
        self.G = 0.5 * (self.eps - self.v / self.c0)
        plt.plot(self.time, self.F, label="F")
        plt.plot(self.time, self.G, label="G")
        plt.legend()
        plt.show()

        # combine F and G
        self.P = self.rho * self.c0**2 * self.A * (self.F + self.G)
        self.vdash = self.c0 * (self.F - self.G)
        #plt.plot(self.P, label="P")
        plt.plot(self.time, self.vdash, label="v from F-G")
        plt.plot(self.time, self.v, "r--", label="v from linescan")

        plt.legend()
        plt.show()

    def CalibrateVelocity(self, calibration_factor=None):
        """
        linescan velocity self.v_px is given in units of pixels/time.
        We need to establish a claibration factor to render this in physical units.
        We derive the calibration factor from  \sigma = \rho \c_0 U_p, i.e., by
        requiring that c_0 \varepsilon = Up. This is true only within the first wave transit time.
        """

        if calibration_factor == None:

            # establish first wave transit time
            import tools
            rise_time_index = tools.find_TTL(self.force, direction="positive", level=2.5)
            self.rise_time = self.time[rise_time_index]
            self.tau = 2 * self.Lbar / self.c0
            print(f"rise time of 1st wave transit is {self.rise_time}, duration is {self.tau}")

                # require that the mean of c_0 \varepsilon equals Up
            startIndex = np.argmax(self.time > self.rise_time)
            stopIndex  = np.argmax(self.time > self.rise_time + self.tau)
    
            print(f"discrete interval: {self.time[startIndex]} -- {self.time[stopIndex]}")

            c0eps_mean = self.c0 * np.mean(self.eps[startIndex:stopIndex])
            Up_mean = np.mean(self.v_px[startIndex:stopIndex])
            print(f"mean of c0.eps is {c0eps_mean}, mean of Up is {Up_mean}")
            calibration_factor = c0eps_mean / Up_mean
            print("calibration factor is:", calibration_factor)

        self.v = self.v_px * calibration_factor


        #plt.plot(self.time[startIndex:stopIndex], self.v_px[startIndex:stopIndex], "r--", label="strain from force")
        #plt.plot(self.time[startIndex:stopIndex], self.c0*self.eps0[startIndex:stopIndex], label="spline")
        #plt.title("Check Calibration!")
        #plt.show()

        outData = np.column_stack((self.time, self.eps, self.v))
        filename = "eps_vel.dat" #os.path.join(self.path, "eps_vel.dat")
        np.savetxt(filename, outData, header="time, strain, velocity")
        print("wrote strain and velocity to file: ", filename)



    def Interpolate(self):
        """
        define common time axis between strain gauge and line scan datasets.
        Interpolate line scan data to strain gauge time basis.
        """

        start= max(self.time_force[0], self.time_line[0])
        stop = min(self.time_force[-1], self.time_line[-1])
        print(f"Common time range: {start} -- {stop}")

        # create a spline interpoland of the line scan data
        lam = 1.0e-6
        
        spl_linescan = make_smoothing_spline(self.time_line - self.additionalShift, self.u_px, lam=lam)

        # create a common time axis
        self.dt = self.time_force[1] - self.time_force[0]
        N = int( (stop - start) / self.dt)
        self.time = np.linspace(start, stop, N+1, endpoint=False)
        print(f"requested dt is {self.dt}, actual dt is {self.time[1] - self.time[0]}")

        # interpolate velocity data to new time axis
        self.v_px = -spl_linescan(self.time, nu=1)

        # compute strain from force data and interpolate to new time axis
        spl_force = make_smoothing_spline(self.time_force, self.force, lam=lam)
        self.force = spl_force(self.time)
        self.eps = self.force / (self.A * self.rho * self.c0**2)



    def plotForce(self):
        #plt.plot(self.time, self.eps0, label="strain from force")
        #plt.plot(self.time, self.v / self.c0, label="velocity / c0")
        #plt.plot(self.time, self.eps, label="strain")

        plt.plot(self.time, self.force, label="force from strain gauge")
        plt.plot(self.time, self.v * self.rho * self.c0 * self.A, label="force from velocity")
        #plt.xlim(8,11)
        #plt.axvline(self.rise_time)
        #plt.axvline(self.rise_time + self.tau)
        plt.legend()
        plt.show()


    def LoadDisplacementOverTime(self, path):
        data = np.genfromtxt(os.path.join(path, "linescan_analysis.dat"))
        self.time_line = data[:,0] #/ 200.0
        self.u_px = data[:,1] # pixels

    def compute_velocity(self):
        """
        compute the velocity 
        """

        lam = 1.0e-6
        spl = make_smoothing_spline(self.time_line, self.u_px, lam=lam)
        self.v_px = -spl(self.time_line, nu=1)

    def LoadForceOverTime(self, path):
        data = np.genfromtxt(os.path.join(path, "Symmpact_time_force.txt"))
        self.time_force, self.force = data[:,0], data[:,1]


if __name__ == "__main__":

    path = "/home/gcg/Projekte/21_WaveSeparation/2025-01-30_Waveseparation/02_PC"
    instance = CalibrateWaveSeparation(path)
    sys.exit()

    A = np.pi*20*20
    rho = 1.21e-6
    c0 = 1430.0

    
    

    # derive ascending and descending waves
    # F = 0.5 * (eps.c0 + v)
    # G = 0.5 * (eps.c0 - v)



