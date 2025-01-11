# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-09 08:20:48
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-09 15:13:01

"""
Apply the wave separation technique of 
(1) Casem, D. T.; Fourney, W.; Chang, P. Wave Separation in Viscoelastic Pressure Bars Using Single-Point Measurements of Strain and Velocity. Polymer Testing 2003, 22 (2), 155–164. https://doi.org/10.1016/S0142-9418(02)00064-8.
"""

import numpy as np
from scipy import signal
import pylab as plt
import os, sys, pickle
global dt

class BarMeasurement:
    """
    This class represents a combined force / velocity measurement.
    After instantiatig this class, it computes the pair of ascending / descendig waves F and G
    """
    def __init__(self, filename):
        self.rho = 2.7e-6
        self.A_bar = 0.25 * np.pi * 16**2
        self.c0 = 5091.750772173155
        self.distance_vel_eps = 500.0
        
        self.read_file(filename)
        self.shift_velocity_to_strain()
        self.assemble_P_v()

    def shift_velocity_to_strain(self):
        V = np.fft.rfft(self.vel)
        V *= np.exp(self.gamma * self.distance_vel_eps)
        V_shifted = np.fft.irfft(V)
        
        plt.plot(self.t, self.rho * self.c0**2 * self.eps * self.A_bar, label="force original")
        plt.plot(self.t, self.vel, label="vel original")
        plt.plot(self.t, V_shifted, label="vel shifted")
        plt.legend()
        plt.show()


    def assemble_P_v(self):
        pass

    def calculate_F_G_FrequencyDomain(self):
        """
        calculate the F(w) and G(w)
        """

        EPS = np.fft.rfft(self.eps)
        V = np.fft.rfft(self.vel)
        

        #sys.exit()

        self.Fw =    0.5 * (EPS    - 1.0j * (self.gamma/self.w) * V)
        self.Fw[0] = 0.5 * (EPS[0] - 1.0j * (1.0j/self.c0) * V[0]) # need to handle w=0 separately, otherwise division by zero
        
        self.Gw =    0.5 * (EPS    + 1.0j * (self.gamma/self.w)      * V)
        self.Gw[0] = 0.5 * (EPS[0] + 1.0j * (1.0j/self.c0) * V[0])  # need to handle w=0 separately, otherwise division by zero

        self.F = np.fft.irfft(self.Fw)
        self.G = np.fft.irfft(self.Gw)

        self.P = self.rho * self.c0**2 * (self.F + self.G) * self.A_bar
        self.v =            self.c0    * (self.F - self.G)

    def read_file(self, filename):
        """
        read a datafile suitable for CFC analysis.
        time, strain, velocity
        """
        zeroPad = False
        low_pass_filtering = False

        data = np.genfromtxt(filename)
        t = data[:,0]
        vel = data[:,1]
        eps = data[:,2]
        self.dt = t[1] - t[0]
        N = len(t)
        print("read file [%s], dt=%g, Nrec=%d " % (filename, self.dt, N))

        if len(t) % 2 != 0:
            t = t[:-1]
            vel = vel[:-1]
            eps = eps[:-1]

        self.t = t
        self.vel = vel
        self.eps = eps

        self.w = 2*np.pi * np.fft.rfftfreq(len(self.t), d=self.dt)
        self.alpha = 0.0
        self.gamma = self.alpha + 1.0j * self.w / self.c0
        print("frequencies:", self.w)


    



class solveCFC:
    def __init__(self, filename):

        self.rho = 2.7e-6
        self.A_bar = 0.25 * np.pi * 16**2
        self.c0 = 5091.750772173155

        self.read_file(filename)
        if True:
            self.calculate_F_G_FrequencyDomain()
            self.resolveAtDistanceFrequencyDomain(-1500.0)
        else:
            self.calculate_F_G_TimeDomain()
            self.resolveAtDistanceTimeDomain(-1500.0)
        
        self.plot()

    def resolveAtDistanceTimeDomain(self, d):

        tau = d / self.c0 
        print("shifting time tau:", tau)

        self.t_shifted_F = self.t - tau
        self.F_shifted = np.interp(self.t, self.t_shifted_F, self.F) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.t_shifted_G = self.t + tau
        self.G_shifted = np.interp(self.t, self.t_shifted_G, self.G) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        #plt.plot(self.t, self.F, "r-", label="F")
        #plt.plot(self.t, self.F_shifted, "r--", label="F shifted resampled")
        #plt.plot(self.t, self.G, "g-", label="G")
        #plt.plot(self.t, self.G_shifted, "g--", label="G shifted resampled")
        #plt.legend()
        #plt.show()

        self.P_shifted = self.rho * self.c0**2 * (self.F_shifted + self.G_shifted) * self.A_bar
        self.v_shifted =            -self.c0    * (self.F_shifted - self.G_shifted)

    def resolveAtDistanceFrequencyDomain(self, d):

        tau = d / self.c0 
        print("shifting time tau:", tau)

        # shift F (e -gamma d) and G (e gamma d) in frequency space
        Fws = self.Fw * np.exp(-self.gamma * d)
        Gws = self.Gw * np.exp( self.gamma * d)

        Pw_shifted    = -self.rho * (self.w**2 / (self.gamma**2)) * (Fws    + Gws)    * self.A_bar
        Pw_shifted[0] = self.rho * (self.c0**2)                   * (Fws[0] + Gws[0]) * self.A_bar
        self.P_shifted = np.fft.irfft(Pw_shifted)

        vw_shifted = 1.0j * (self.w / self.gamma) * (Fws - Gws)
        vw_shifted[0] = self.c0 * (Fws[0] - Gws[0])
        self.v_shifted = np.fft.irfft(vw_shifted)




    def calculate_F_G_FrequencyDomain(self):
        """
        calculate the F(w) and G(w)
        """

        EPS = np.fft.rfft(self.epsA)# * dt
        V = np.fft.rfft(self.velA)# * dt
        

        #sys.exit()

        self.Fw =    0.5 * (EPS    - 1.0j * (self.gamma/self.w) * V)
        self.Fw[0] = 0.5 * (EPS[0] - 1.0j * (1.0j/self.c0) * V[0]) # need to handle w=0 separately, otherwise division by zero
        
        self.Gw =    0.5 * (EPS    + 1.0j * (self.gamma/self.w)      * V)
        self.Gw[0] = 0.5 * (EPS[0] + 1.0j * (1.0j/self.c0) * V[0])  # need to handle w=0 separately, otherwise division by zero

        self.F = np.fft.irfft(self.Fw)
        self.G = np.fft.irfft(self.Gw)

        self.P = self.rho * self.c0**2 * (self.F + self.G) * self.A_bar
        self.v =            self.c0    * (self.F - self.G)


    def calculate_F_G_TimeDomain(self):
        """
        calculate the ascending wave F and the descending wave G
        """
        self.F = 0.5 * (self.epsA + self.velA / self.c0)
        self.G = 0.5 * (self.epsA - self.velA / self.c0)

        self.P = self.rho * self.c0**2 * (self.F + self.G) * self.A_bar
        self.v =            self.c0    * (self.F - self.G)

    def read_file(self, filename):
        """
        read a datafile suitable for CFC analysis.
        time, strain, velocity
        """
        zeroPad = False
        low_pass_filtering = False

        data = np.genfromtxt(filename)
        t = data[:,0]
        velA = data[:,1]
        epsA = data[:,2]
        self.dt = t[1] - t[0]
        N = len(t)
        print("read file [%s], dt=%g, Nrec=%d " % (filename, self.dt, N))

        if len(t) % 2 != 0:
            t = t[:-1]
            velA = velA[:-1]
            epsA = epsA[:-1]

        print("Nrec", len(t))

        if zeroPad == True:
            n = np.ceil(np.log(N)/np.log(2)) + 1; 
            m = int(2**(n))
            print("next power of 2 = %d, %d" %(n, m))

            diff= m - N + 1
            velA = np.pad(velA, (0, diff), 'constant', constant_values=0.0)
            epsA = np.pad(epsA, (0, diff), 'constant', constant_values=0.0)
            t = np.arange(m) * self.dt
            self.dt = t[1] - t[0]
            print("padded with zeros, new x-axis extends from %g to %g, dt=%g" % (t[0], t[-1], self.dt))
            print("new length of x:",len(t))
            print("new length of y:",len(velA))
            #assert len(y) == m

        if low_pass_filtering == True:
            plt.plot(t, velA, label="original")

            fs = 1. / self.dt
            fc = 50.0  # 
            omega = fc / (fs / 2) # Normalize the frequency
            b, a = signal.butter(1, omega, 'low')
            velA = signal.filtfilt(b, a, velA)
            epsA = signal.filtfilt(b, a, epsA)

            plt.plot(t, velA, label="filtered")
            plt.show()

        def mirror(seq):
            mirrored = seq[::-1]
            return np.concatenate((seq, mirrored))

        self.t = t
        self.velA = velA
        self.epsA = epsA

        self.w = 2*np.pi * np.fft.rfftfreq(len(self.t), d=self.dt)
        self.alpha = 0.0
        self.gamma = self.alpha + 1.0j * self.w / self.c0
        print("frequencies:", self.w)

    def plot(self):
        fig = plt.figure()
        gs = fig.add_gridspec(3, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        fig.suptitle('Sharing both axes')
        axs[0].plot(self.t, self.epsA, label="A")
        axs[0].plot(self.t, self.velA, label="B")
        axs[0].legend()
        
        axs[1].plot(self.t, self.P, "--", label="P")
        axs[1].plot(self.t, self.v, "--", label="v")
        axs[1].legend()

        axs[2].plot(self.t, self.P_shifted, "--", label="P shifted")
        axs[2].plot(self.t, self.v_shifted, "--", label="v_shifted")
        axs[2].legend()

        
        plt.show()

        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
    
A = BarMeasurement("CFC.dat")

#solver = solveCFC("CFC.dat")