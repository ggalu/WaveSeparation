# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-09 08:20:48
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-31 11:31:31

"""
Apply the wave separation technique of 
(1) Casem, D. T.; Fourney, W.; Chang, P. Wave Separation in Viscoelastic Pressure Bars Using Single-Point Measurements of Strain and Velocity. Polymer Testing 2003, 22 (2), 155–164. https://doi.org/10.1016/S0142-9418(02)00064-8.
"""

import numpy as np
from scipy import signal
import pylab as plt
import os, sys, pickle
import bottleneck as bn
import scipy.signal
from scipy.signal import savgol_filter
#global dt

class solveCFC:
    def __init__(self, filename):

        self.rho = 2.7e-6
        self.E_bar = 70.0
        self.A_bar = 0.25 * np.pi * 40**2
        self.c0 = np.sqrt(self.E_bar/self.rho)
        self.L0 = 10.0 # specimen length
        self.alpha = 1.0e-12 #-1.2e-5

        self.specimen_diameter = 10.0
        self.read_file(filename)
        #self.read_specimen()
        
        self.calculate_F_G_unshifted()
        if True:
            self.resolveAtDistanceFrequencyDomain(-110.0)
        else:
            self.resolveAtDistanceTimeDomain(-120)
        

        self.plot_AB()
        #self.plot_FG()
        #self.plot_strain()
        #self.plot_specimen()
        #self.plot()

    def resolveAtDistanceTimeDomain(self, dA):
        """
        dA: shift for A signals, typically in positive direction

        compute self.FA_shifted, self.FB_shifted
        compute self.GA_shifted, self.GB_shifted
        compute self.PA_shifted, self.vA_shifted
        compute self.PB_shifted, self.vB_shifted
        """

        # strain gauge A
        tauA = dA / self.c0 
        print("shifting time tau for A:", tauA)
        self.t_shift= self.t - tauA
        self.FA_shifted = np.interp(self.t, self.t_shift, self.FA) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.t_shift = self.t + tauA
        self.GA_shifted = np.interp(self.t, self.t_shift, self.GA) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.PA_shifted = self.rho * self.c0**2 * (self.FA_shifted + self.GA_shifted) * self.A_bar
        self.vA_shifted =            -self.c0    * (self.FA_shifted - self.GA_shifted)

        # shift velocity to the left of specimen
        tauA = 2*dA / self.c0 
        print("shifting time tau for LHS:", tauA)
        self.t_shift= self.t - tauA
        self.FA_shifted = np.interp(self.t, self.t_shift, self.FA) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.t_shift = self.t + tauA
        self.GA_shifted = np.interp(self.t, self.t_shift, self.GA) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        self.t_shift= self.t - tauA
        self.vB_shifted =  -self.c0    * (self.FA_shifted - self.GA_shifted)

    def resolveAtDistanceFrequencyDomain(self, dA):

        print("shifting distance d:", dA)

        def shift_P(Fw, Gw, d):
            """
            Shift the sum of F and G by d.
            Do this by propagating the sum of the Fourier transforms F(w) and G(w).
            Return the inverse transform, i.e., the time-domain signal of the force P(t)
            """

            bracket = Fw * np.exp(self.gamma * d) + Gw * np.exp(-self.gamma * d)
            prefactor = -self.rho * (self.w**2 / (self.gamma**2)) * self.A_bar

            return np.fft.irfft(prefactor * bracket)
        
        def shift_v(Fw, Gw, d):
            """
            Shift the difference of F and G by d.
            Do this by propagating the difference of the Fourier transforms F(w) and G(w).
            Return the inverse transform, i.e., the time-domain signal of the velocity v(t)
            """

            bracket = Fw * np.exp(-self.gamma * d) - Gw * np.exp(self.gamma * d)
            prefactor = 1.0j * self.w / self.gamma

            return np.fft.irfft(prefactor * bracket)


        # forces
        self.PA_shifted    = shift_P(self.FAw, self.GAw, dA)
        self.PA_shifted -= self.PA_shifted[0] # restore DC component

        if len(self.PA_shifted) != len(self.t):
            print("len is different", len(self.PA_shifted), len(self.t))
            sys.exit()

        # velocities
        self.vA_shifted = shift_v(self.FAw, self.GAw, dA)
        self.vA_shifted -= self.vA_shifted[0]

        self.vB_shifted = shift_v(self.FAw, self.GAw, 2*dA)
        self.vB_shifted -= self.vB_shifted[0]


    def calculate_F_G_unshifted(self):
        """
        Calculate the unshifted F(w) and G(w) by doing Fourier transforms of the time-domain signals F(t) and G(t).
        Unshifted means that they are calculated at the locations of the strain gauges.
        """

        self.FA = 0.5 * (self.epsA + self.velA / self.c0)
        self.GA = 0.5 * (self.epsA - self.velA / self.c0)
        self.FAw = np.fft.rfft(self.FA)
        self.GAw = np.fft.rfft(self.GA)

    def read_specimen(self):
        """
        read specimen strain and force from simulation
        """
        data = np.genfromtxt("specimen.dat")
        self.specimen_stress = data[:,1]
        self.specimen_strain = data[:,2]
        #elA = data[:,2]

    def read_file(self, filename):
        """
        read a datafile suitable for CFC analysis.
        time, strain, velocity
        """

        def mirror(seq):
            mirrored = seq[::-1]
            return np.concatenate((seq, mirrored))
        

        zeroPad = True
        low_pass_filtering = True

        data = np.genfromtxt(filename)

        if data.shape[0] % 2 == 1:
            data = data[:-1,:]

        #sys.exit()

        t = data[:,0]
        epsA = data[:,1]
        velA = data[:,2]
        self.dt = t[1] - t[0]
        N = len(t)
        print("read file [%s], dt=%g, Nrec=%d " % (filename, self.dt, N))
        print("Nrec", len(t))

        

        if zeroPad == True:
            #n = np.ceil(np.log(N)/np.log(2)) + 1; 
            #m = int(2**(n))
            #print("next power of 2 = %d, %d" %(n, m))

            #diff= m - N + 1
            diff = 1000
           
            velA = np.pad(velA, (diff, diff), 'constant', constant_values=velA[0])
            epsA = np.pad(epsA, (diff, diff), 'constant', constant_values=epsA[0])
            m = len(velA)
            t = np.arange(m) * self.dt - diff * self.dt
            print("padded with zeros, new x-axis extends from %g to %g, dt=%g" % (t[0], t[-1], self.dt))
            print("new length of x:",len(t))
            print("new length of y:",len(velA))
            #assert len(y) == m

        if low_pass_filtering == True:
            #plt.plot(t, epsA, label="original")
            fs = 1. / self.dt
            fc = 50.0  # 
            omega = fc / (fs / 2) # Normalize the frequency
            b, a = signal.butter(1, omega, 'low')
            #velA = signal.filtfilt(b, a, velA)
            #epsA = signal.filtfilt(b, a, epsA)

            epsA = savgol_filter(epsA, 100, 1)
            velA = savgol_filter(velA, 100, 1)

            #plt.plot(t, epsA, label="filtered")
            #plt.legend()
            #plt.show()

        

        self.t = t
        self.velA = velA
        self.epsA = -epsA # flip sign to compression

        self.PA = self.epsA * self.E_bar * self.A_bar # store the initial force signal

        self.w = 2*np.pi * np.fft.rfftfreq(len(self.t), d=self.dt)
        self.gamma = self.alpha + 1.0j * self.w / self.c0

        # check if there are any poles in gamma
        for i in range(len(self.gamma)):
            if abs(self.gamma[i]) < 1.0e-8:
                print(f"pole in gamma at index {i}, value is {self.gamma[i]}")

        #sys.exit()


        #self.gamma[0] = self.gamma[1]
        print("frequencies:", self.w)

    def plot_AB(self):
        """
        plot the strain gauge signals at A and B
        """

        #plt.plot(self.t, self.rho * self.c0**2 * self.epsA * self.A_bar, label="force A")


        
        #filter with a window length of 5 and a degree 2 polynomial
        #PA_smooth = savgol_filter(self.PA_shifted, 200, 1)

        
        plt.plot(self.t, self.PA, "g-", label="force A")
        plt.plot(self.t, self.PA_shifted, "r--", label="force A shifted")

        #plt.plot(self.t, self.velA, label="velA")
        #plt.plot(self.t, self.vA_shifted, label="velA shifted")
        #plt.plot(self.t, self.vB_shifted, label="velB shifted")
        #plt.plot(self.t, self.vA_shifted - self.vB_shifted, label="difference of shifted velocities")

        #plt.plot(self.t, -self.PA_shifted, "c--", label="force A shifted")
        #plt.plot(self.t, -(self.PA_shifted + self.PB_shifted)/2, "r-", label="force AB shifted sum")

        #specimen_area = 0.25 * np.pi * self.specimen_diameter**2
        #specimen_force = -self.specimen_stress * specimen_area
        #plt.plot(self.t, specimen_force, "g--", label="specimen")
        plt.xlim(1.5, 5)
        plt.ylim(0, -15)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_FG(self):
        """
        plot the strain gauge signal at A and the associated waves F and G
        """
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(1200*px, 800*px))
        gs = fig.add_gridspec(3, 2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        fig.suptitle('Sharing both axes')
        
        axs[0,0].plot(self.t, self.rho * self.c0**2 * self.epsA * self.A_bar, label="force A")
        axs[0,0].plot(self.t, self.velA, label="velA")
        axs[0,0].legend()
        
        axs[1,0].plot(self.t, self.FA, "--", label="F @ A")
        axs[1,0].plot(self.t, self.GA, "--", label="G @ A")
        axs[1,0].legend()

        axs[2,0].plot(self.t, self.PA_shifted, "--", label="force shifted")
        axs[2,0].plot(self.t, self.vA_shifted, "--", label="velocity shifted")
        axs[2,0].legend()
        plt.show()

    


    def plot_strain(self):

        

        # plot the relative motion of the bar-specimen interfaces
        vrel = self.vA_shifted - self.vB_shifted
        urel = np.cumsum(vrel) * self.dt
        strain = urel / self.L0
        
        plt.plot(self.t, strain, label="calculated strain from interface velocity")
        plt.plot(self.t, self.specimen_strain[:], "r--", label="simulated strain")
        plt.legend()
        plt.show()

    def plot_specimen(self):

        # plot the relative motion of the bar-specimen interfaces
        vrel = self.vA_shifted - self.vB_shifted
        urel = np.cumsum(vrel) * self.dt
        strain = urel / self.L0
        
        specimen_area = 0.25 * np.pi * self.specimen_diameter**2
        
        plt.plot(self.t, -self.specimen_stress * specimen_area, label="specimen force")
        #plt.plot(self.t, self.PA_shifted, label="force A shifted")
        #plt.plot(self.t, self.PB_shifted, label="force B shifted")

        force_avg = 0.5 * (self.PA_shifted + self.PB_shifted)

        # compute a smooting duration corresponding to the time it takes to traverse the specimen
        dt = self.t[1] - self.t[0]
        tau = 0.0035 # step time in rise of signal
        nsmooth = int(tau / dt)

        #force_avg = bn.move_mean(force_avg, nsmooth)

        plt.plot(self.t, -force_avg, label="avg force shifted")
        plt.xlim(-0.05, 2.0)
        plt.ylim(0, None)
        plt.legend()
        plt.show()

        constitutive_stress = self.JC_A + self.JC_B * abs(-strain)**self.JC_n

        plt.plot(-strain, -force_avg / specimen_area, label="averaged shifted AB signals")
        #plt.plot(-strain, -self.PA_shifted / specimen_area)
        #plt.plot(-strain, -self.PB_shifted / specimen_area)

        #plt.plot(-strain, constitutive_stress)
        plt.plot(-self.specimen_strain, -self.specimen_stress, label="average specimen stress/strain")
        plt.legend()
        plt.ylim(0, None)
        #plt.show()
        
        
    
#A = BarMeasurement("CFC.dat")

solver = solveCFC("eps_vel.dat")