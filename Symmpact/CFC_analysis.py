# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-09 08:20:48
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-25 14:28:30

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
#global dt

class solveCFC:
    def __init__(self, filename):

        self.rho = 2.7e-6
        self.E_bar = 70.0
        self.A_bar = 0.25 * np.pi * 40**2
        self.c0 = np.sqrt(self.E_bar/self.rho)
        self.L0 = 10.0 # specimen length
        self.alpha = 0.0

        self.specimen_diameter = 10.0
        self.JC_A = 0.08 # Johnson-Cook A
        self.JC_B = 0.08 # Johnson-Cook B
        self.JC_n = 0.3  # Johnson-Cook n
        

        self.read_file(filename)
        #self.read_specimen()
        
        if True:
            self.calculate_F_G_FrequencyDomain()
            self.resolveAtDistanceFrequencyDomain(130.0, -130.0)
        else:
            self.calculate_F_G_TimeDomain()
            self.resolveAtDistanceTimeDomain(130.0, -130.0)
        

        self.plot_AB()
        #self.plot_FG()
        #self.plot_strain()
        #self.plot_specimen()
        #self.plot()

    def resolveAtDistanceTimeDomain(self, dA, dB):
        """
        dA: shift for A signals, typically in positive direction
        dB: shift for B signals, typically in negative direction

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

        # strain gauge B
        tauB = dB / self.c0 
        print("shifting time tau for B:", tauB)
        self.t_shift = self.t - tauB
        self.FB_shifted = np.interp(self.t, self.t_shift, self.FB) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.t_shift = self.t + tauB
        self.GB_shifted = np.interp(self.t, self.t_shift, self.GB) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.PB_shifted = self.rho *  self.c0**2 * (self.FB_shifted + self.GB_shifted) * self.A_bar
        self.vB_shifted =            -self.c0    * (self.FB_shifted - self.GB_shifted)

    def resolveAtDistanceFrequencyDomain(self, dA, dB):

        print("shifting distance d:", dA, dB)

        # do strain gauge A
        # shift F (e -gamma d) and G (e gamma d) in frequency space
        FAws = self.FAw * np.exp( self.gamma * dA)
        GAws = self.GAw * np.exp(-self.gamma * dA)

        self.FA_shifted = np.fft.irfft(FAws)
        self.GA_shifted = np.fft.irfft(GAws)

        # after shifting, we can have lost the DC offset, initial value at t=0 is zero. recover this.
        #self.FA_shifted -= self.FA_shifted[0]
        #self.GA_shifted -= self.GA_shifted[0]



        Pw_shifted    = -self.rho * (self.w**2 / (self.gamma**2)) * (FAws    + GAws)    * self.A_bar
        Pw_shifted[0] =  self.rho * (self.c0**2)                  * (FAws[0] + GAws[0]) * self.A_bar
        self.PA_shifted = np.fft.irfft(Pw_shifted)
        
        print(f"offset in PA_shifted is:", self.PA_shifted[0])
        #self.PA_shifted -= self.PA_shifted[0]

        vw_shifted = 1.0j * (self.w / self.gamma) * (FAws - GAws)
        vw_shifted[0] = self.c0 * (FAws[0] - GAws[0])
        self.vA_shifted = -np.fft.irfft(vw_shifted)
        #self.vA_shifted 

        # do strain gauge B
        # shift F (e -gamma d) and G (e gamma d) in frequency space
        FBws = self.FBw * np.exp( self.gamma * dB)
        GBws = self.GBw * np.exp(-self.gamma * dB)

        self.FB_shifted = np.fft.irfft(FBws)
        self.GB_shifted = np.fft.irfft(GBws)

        Pw_shifted    = -self.rho * (self.w**2 / (self.gamma**2)) * (FBws    + GBws)    * self.A_bar
        Pw_shifted[0] =  self.rho * (self.c0**2)                  * (FBws[0] + GBws[0]) * self.A_bar
        self.PB_shifted = np.fft.irfft(Pw_shifted)

        vw_shifted = 1.0j * (self.w / self.gamma) * (FBws - GBws)
        vw_shifted[0] = self.c0 * (FBws[0] - GBws[0])
        self.vB_shifted = -np.fft.irfft(vw_shifted)

        #print("gamma:", self.gamma)

    def calculate_F_G_FrequencyDomain(self):
        """
        calculate the F(w) and G(w)
        """

        # strain gauge A
        EPSA = np.fft.rfft(self.epsA)# * dt
        VA = np.fft.rfft(self.velA)# * dt

        self.FAw =    0.5 * (EPSA    - 1.0j * (self.gamma/self.w) * VA)
        self.FAw[0] = 0.5 * (EPSA[0] - 1.0j * (1.0j/self.c0) * VA[0]) # need to handle w=0 separately, otherwise division by zero
        self.FA = np.fft.irfft(self.FAw)
        
        self.GAw =    0.5 * (EPSA    + 1.0j * (self.gamma/self.w)      * VA)
        self.GAw[0] = 0.5 * (EPSA[0] + 1.0j * (1.0j/self.c0) * VA[0])  # need to handle w=0 separately, otherwise division by zero
        self.GA = np.fft.irfft(self.GAw)

        


        # strain gauge B
        EPSB = np.fft.rfft(self.epsB)# * dt
        VB = np.fft.rfft(self.velB)# * dt

        self.FBw =    0.5 * (EPSB    - 1.0j * (self.gamma/self.w) * VB)
        self.FBw[0] = 0.5 * (EPSB[0] - 1.0j * (1.0j/self.c0) * VB[0]) # need to handle w=0 separately, otherwise division by zero
        self.FB = np.fft.irfft(self.FBw)
        
        self.GBw =    0.5 * (EPSB    + 1.0j * (self.gamma/self.w)      * VB)
        self.GBw[0] = 0.5 * (EPSB[0] + 1.0j * (1.0j/self.c0) * VB[0])  # need to handle w=0 separately, otherwise division by zero
        self.GB = np.fft.irfft(self.GBw)


        # calculate Time Domain F, G for debugging
        #self.FATD = 0.5 * (self.epsA + self.velA / self.c0)
        #self.GATD = 0.5 * (self.epsA - self.velA / self.c0)

        #plt.plot(self.t, self.FA, label="FA Fourier")
        #plt.plot(self.t, self.GA, label="GA Fourier")
        #plt.plot(self.t, self.FATD, label="FA TD")
        #plt.plot(self.t, self.GATD, label="GA TD")
        #plt.legend()
        #plt.show()
        #sys.exit()



    def calculate_F_G_TimeDomain(self):
        """
        calculate the ascending wave F and the descending wave G for strain gauges A and B
        """
        self.FA = 0.5 * (self.epsA + self.velA / self.c0)
        self.GA = 0.5 * (self.epsA - self.velA / self.c0)

        self.FB = 0.5 * (self.epsB + self.velB / self.c0)
        self.GB = 0.5 * (self.epsB - self.velB / self.c0)

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
        t = data[:,0]
        epsA = data[:,1]
        velA = data[:,2]
        epsB = data[:,3]
        velB = data[:,4]
        self.dt = t[1] - t[0]
        N = len(t)
        print("read file [%s], dt=%g, Nrec=%d " % (filename, self.dt, N))

        #read specimen strain and force from simulation
        data = np.genfromtxt("specimen.dat")
        specimen_stress = data[:,1]
        specimen_strain = data[:,2]
        assert len(specimen_stress) == N

        #epsA = mirror(epsA)
        #epsB = mirror(epsB)
        #velA = mirror(velA)
        #velB = mirror(velB)
        #specimen_stress = mirror(specimen_stress)
        #specimen_strain = mirror(specimen_strain)
        #N = len(epsA)
        #t = np.arange(N) * self.dt

        #if len(t) % 2 != 0:
        #    t = t[:-1]
        #    velA = velA[:-1]
        #    epsA = epsA[:-1]
        #    velB = velB[:-1]
        #    epsB = epsB[:-1]

        print("Nrec", len(t))

        

        if zeroPad == True:
            #n = np.ceil(np.log(N)/np.log(2)) + 1; 
            #m = int(2**(n))
            #print("next power of 2 = %d, %d" %(n, m))

            #diff= m - N + 1
            diff = 1000

            
            velA = np.pad(velA, (diff, diff), 'constant', constant_values=velA[0])
            epsA = np.pad(epsA, (diff, diff), 'constant', constant_values=epsA[0])
            velB = np.pad(velB, (diff, diff), 'constant', constant_values=velB[0])
            epsB = np.pad(epsB, (diff, diff), 'constant', constant_values=epsB[0])
            specimen_strain = np.pad(specimen_strain, (diff, diff), 'constant', constant_values=specimen_strain[0])
            specimen_stress = np.pad(specimen_stress, (diff, diff), 'constant', constant_values=specimen_stress[0])

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
            velA = signal.filtfilt(b, a, velA)
            epsA = signal.filtfilt(b, a, epsA)
            velB = signal.filtfilt(b, a, velB)
            epsB = signal.filtfilt(b, a, epsB)

            #plt.plot(t, epsA, label="filtered")
            #plt.legend()
            #plt.show()

        

        self.t = t
        self.velA = velA
        self.epsA = epsA
        self.velB = velB
        self.epsB = epsB
        self.specimen_strain = specimen_strain
        self.specimen_stress = specimen_stress

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

        plt.plot(self.t, -self.rho * self.c0**2 * self.epsA * self.A_bar, label="force A")
        plt.plot(self.t, -self.rho * self.c0**2 * self.epsB * self.A_bar, label="force B")
        plt.plot(self.t, -self.PA_shifted, "c--", label="force A shifted")
        plt.plot(self.t, -self.PB_shifted, "r--", label="force B shifted")

        specimen_area = 0.25 * np.pi * self.specimen_diameter**2
        specimen_force = -self.specimen_stress * specimen_area
        plt.plot(self.t, specimen_force, "g--", label="specimen")
        #plt.xlim(-0.20, 0.25)
        #plt.ylim(-1, None)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_FG(self):
        """
        plot the strain gauge signal at A and the associated waves F and G
        """
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig = plt.figure(figsize=(1200*px, 800*px))
        gs = fig.add_gridspec(4, 2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        fig.suptitle('Sharing both axes')
        
        axs[0,0].plot(self.t, self.rho * self.c0**2 * self.epsA * self.A_bar, label="force A")
        axs[0,0].plot(self.t, self.velA, label="velA")
        axs[0,0].legend()
        
        axs[1,0].plot(self.t, self.FA, "--", label="F @ A")
        axs[1,0].plot(self.t, self.GA, "--", label="G @ A")
        axs[1,0].legend()

        axs[2,0].plot(self.t, self.FA_shifted, "--", label="F shifted")
        axs[2,0].plot(self.t, self.GA_shifted, "--", label="G_shifted")
        axs[2,0].legend()

        axs[3,0].plot(self.t, self.PA_shifted, "--", label="force shifted")
        axs[3,0].plot(self.t, self.vA_shifted, "--", label="velocity shifted")
        axs[3,0].legend()

        # plot strain gauge B
        axs[0,1].plot(self.t, self.rho * self.c0**2 * self.epsB * self.A_bar, label="force B")
        axs[0,1].plot(self.t, self.velB, label="velB")
        axs[0,1].legend()
        
        axs[1,1].plot(self.t, self.FB, "--", label="F @ B")
        axs[1,1].plot(self.t, self.GB, "--", label="G @ B")
        axs[1,1].legend()

        axs[2,1].plot(self.t, self.FB_shifted, "--", label="F@B shifted")
        axs[2,1].plot(self.t, self.GB_shifted, "--", label="G@B_shifted")
        axs[2,1].legend()

        axs[3,1].plot(self.t, self.PB_shifted, "--", label="force B shifted")
        axs[3,1].plot(self.t, self.vB_shifted, "--", label="velocity B shifted")
        axs[3,1].legend()
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