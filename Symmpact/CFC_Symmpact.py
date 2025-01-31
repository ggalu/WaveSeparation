# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-09 08:20:48
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-31 14:49:22

"""
Apply the wave separation technique of 
(1) Casem, D. T.; Fourney, W.; Chang, P. Wave Separation in Viscoelastic Pressure Bars Using Single-Point Measurements of Strain and Velocity. Polymer Testing 2003, 22 (2), 155–164. https://doi.org/10.1016/S0142-9418(02)00064-8.

- read Symmpact Force signal on output bar
- read linescan displacement signal on output bar

"""

import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp
import os, sys, pickle
import bottleneck as bn
import scipy.signal
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
#global dt


path = "/home/gcg/Projekte/21_WaveSeparation/2025-01-30_Waveseparation/01_M150"

class solveCFC:
    def __init__(self, path):


        self.rho = 2.7e-6
        self.E_bar = 70.0
        self.A_bar = 0.25 * np.pi * 40**2
        self.L_bar = 1900.0
        self.c0 = np.sqrt(self.E_bar/self.rho)
        self.L0 = 10.0 # specimen length
        self.alpha = 1.0e-12 #-1.2e-5
        self.specimen_diameter = 10.0
        self.delay = 0.0 # time delay between linescan and strain gauge data
        self.shift = 120.0 # shifting distance from strain gauge to specimen
        self.calibration_factor=0.009444415150988306*1.05 # conversion factor between force and velocity
        self.significant_force_level = 1.0

        self.LoadDisplacementOverTime(path)
        self.LoadForceOverTime(path)
        self.Interpolate()
        self.CalibrateVelocity()
        self.calculate_F_G_unshifted()
        #self.resolveAtDistanceFrequencyDomain(-110.0)
        self.resolveAtDistanceTimeDomain()

        self.plotForce()

        self.plotShiftedForce()
        self.show_SpinBox()

        #self.plot_AB()
        #self.plot_FG()
        #self.plot_strain()
        #self.plot_specimen()
        #self.plot()


    def show_SpinBox(self):
        """
        This routine is called whenever the additional shift value between line scan and strain gauge time axes changes.
        """

        def valueChanged_delay(spinbox):

            self.delay = spinbox.value()
            print(self.delay)

            self.Interpolate()
            #self.CalibrateVelocity(calibration_factor=0.009444415150988306)
            self.calculate_F_G_unshifted()
            ##self.resolveAtDistanceFrequencyDomain(-110.0)
            self.resolveAtDistanceTimeDomain()
            self.shiftedForceLine.setData(self.time, self.PA_shifted)

        def valueChanged_shift(spinbox):

            self.shift = spinbox.value()
            print(self.shift)
            self.Interpolate()
            #self.CalibrateVelocity(calibration_factor=0.009444415150988306)
            self.calculate_F_G_unshifted()
            ##self.resolveAtDistanceFrequencyDomain(-110.0)
            self.resolveAtDistanceTimeDomain()
            self.shiftedForceLine.setData(self.time, self.PA_shifted)

        app = pg.mkQApp("SpinBox Example")
        win = QtWidgets.QMainWindow()
        win.setWindowTitle('pyqtgraph example: SpinBox')
        cw = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        cw.setLayout(layout)
        win.setCentralWidget(cw)
        win.show()
        spin_delay = pg.SpinBox(value=0.0, bounds=[None, None], finite=True)
        spin_delay.sigValueChanged.connect(valueChanged_delay)

        spin_shift = pg.SpinBox(value=120, int=True, minStep=1, step=1, bounds=[None, None], finite=True)
        spin_shift.sigValueChanged.connect(valueChanged_shift)

        layout.addWidget(QtWidgets.QLabel("delay:"))
        layout.addWidget(spin_delay)
        layout.addWidget(QtWidgets.QLabel("shifting distance:"))
        layout.addWidget(spin_shift)
        pg.exec()

    def CalibrateVelocity(self):
        """
        linescan velocity self.v_px is given in units of pixels/time.
        We need to establish a claibration factor to render this in physical units.
        We derive the calibration factor from  \sigma = \rho \c_0 U_p, i.e., by
        requiring that c_0 \varepsilon = Up. This is true only within the first wave transit time.
        """

        

            # establish first wave transit time
        import tools
        rise_time_index = tools.find_TTL(self.force, direction="positive", level=self.significant_force_level)
        self.rise_time = self.time[rise_time_index]
        self.tau = 2 * self.L_bar / self.c0
        print(f"rise time of 1st wave transit is {self.rise_time}, duration is {self.tau}")
        
        if self.calibration_factor == None:

                # require that the mean of c_0 \varepsilon equals Up
            startIndex = np.argmax(self.time > self.rise_time)
            stopIndex  = np.argmax(self.time > self.rise_time + self.tau)
    
            print(f"discrete interval: {self.time[startIndex]} -- {self.time[stopIndex]}")

            c0eps_mean = self.c0 * np.mean(self.eps[startIndex:stopIndex])
            Up_mean = np.mean(self.v_px[startIndex:stopIndex])
            print(f"mean of c0.eps is {c0eps_mean}, mean of Up is {Up_mean}")
            calibration_factor = c0eps_mean / Up_mean
            print("calibration factor is:", calibration_factor)

        self.v = self.v_px * self.calibration_factor

    def Interpolate(self):
        """
        define common time axis between strain gauge and line scan datasets.
        Interpolate line scan data to strain gauge time basis.
        
        Additional shift refers to a time shift between the line scan data and the force data.
        This shift needs to be dialled such that the resulting force curve is as smooth as possible at the wave transits.
        """

        print("*** delay shift:", self.delay)

        start= max(self.time_force[0], self.time_line[0])
        stop = min(self.time_force[-1], self.time_line[-1])
        print(f"Common time range: {start} -- {stop}")

        # create a spline interpoland of the line scan data
        lam = 1.0e-6
        spl_linescan = make_smoothing_spline(self.time_line - self.delay, self.u_px, lam=lam)

        # create a common time axis with an even number of data points -- required beacuse we do FFT later on
        self.dt = self.time_force[1] - self.time_force[0]
        N = int( (stop - start) / self.dt)
        if N % 2 != 0: N -= 1
        self.time = np.linspace(start, stop, N, endpoint=False)
        print(f"requested dt is {self.dt}, actual dt is {self.time[1] - self.time[0]}, length of time axis is {N}")

        # create frequency axis for this time axis
        self.w = 2*np.pi * np.fft.rfftfreq(len(self.time), d=self.dt)
        self.gamma = self.alpha + 1.0j * self.w / self.c0

        # interpolate velocity data to new time axis
        self.v_px = -spl_linescan(self.time, nu=1)
        self.v = self.v_px * self.calibration_factor

        # compute strain from force data and interpolate to new time axis
        #print("length and dtype of time_force, force", len()
        spl_force = make_smoothing_spline(self.time_force, self.force_original, lam=lam)
        self.force = spl_force(self.time)
        self.eps = self.force / (self.A_bar * self.rho * self.c0**2)

    def LoadDisplacementOverTime(self, path):
        data = np.genfromtxt(os.path.join(path, "linescan_analysis.dat"))
        self.time_line = data[:,0] #/ 200.0
        self.u_px = data[:,1] # pixels

    def LoadForceOverTime(self, path):
        data = np.genfromtxt(os.path.join(path, "Symmpact_time_force.txt"))
        self.time_force, self.force_original = data[:,0], data[:,1]

    def resolveAtDistanceTimeDomain(self):
        """
        dA: shift for A signals, typically in positive direction

        compute self.FA_shifted, self.FB_shifted
        compute self.GA_shifted, self.GB_shifted
        compute self.PA_shifted, self.vA_shifted
        compute self.PB_shifted, self.vB_shifted
        """

        # strain gauge A
        tauA = self.shift / self.c0 
        print("shifting time tau for A:", tauA)
        self.t_shift= self.time - tauA
        self.FA_shifted = np.interp(self.time, self.t_shift, self.FA) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.t_shift = self.time + tauA
        self.GA_shifted = np.interp(self.time, self.t_shift, self.GA) # shifted time axes for F, G are not neccessarily the same. resample these at the original self.t
        
        self.PA_shifted = self.rho * self.c0**2 * (self.FA_shifted + self.GA_shifted) * self.A_bar
        self.vA_shifted =            -self.c0    * (self.FA_shifted - self.GA_shifted)

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

        if len(self.PA_shifted) != len(self.time):
            print("len of time axis after shifting is different", len(self.PA_shifted), len(self.time))
            sys.exit(1)

        # velocities
        self.vA_shifted = shift_v(self.FAw, self.GAw, dA)
        self.vA_shifted -= self.vA_shifted[0]

        self.vB_shifted = shift_v(self.FAw, self.GAw, 2*dA)
        self.vB_shifted -= self.vB_shifted[0]


    def calculate_F_G_unshifted(self, nsmooth=200):
        """
        Calculate the unshifted time domain signals F(t) and G(t).
        Unshifted means that they are calculated at the locations of the strain gauge.
        Also perform filtering.
        """

        smooth_eps = savgol_filter(self.eps, nsmooth, 3)
        smooth_v   = savgol_filter(self.v, nsmooth, 1)

        self.FA = 0.5 * (smooth_eps + smooth_v / self.c0)
        self.GA = 0.5 * (smooth_eps - smooth_v / self.c0)
        self.FAw = np.fft.rfft(self.FA)
        self.GAw = np.fft.rfft(self.GA)

    def plotForce(self):
        #plt.title("1st wave transit: check that forces agree!")
        pg.setConfigOptions(antialias=True)
        plotWidget = pg.plot(title="Force-Velocity consitency check")
        plotWidget.addLegend()
        
        self.shiftedForceLine = plotWidget.plot(self.time, self.force, pen=pg.mkPen(1, width=2,), name="force from strain gauges")  ## setting pen=None disables line drawing
        
        plotWidget.plot(self.time, self.v * self.rho * self.c0 * self.A_bar, pen=pg.mkPen("g", width=2,), name="force from velocity")
        plotWidget.setLabel('left', 'force', units='kN')
        plotWidget.setLabel('bottom', 'force', units='kN')
        plotWidget.showGrid(x=True, y=True)
        plotWidget.setXRange(self.rise_time,self.rise_time + self.tau)
        plotWidget.setAutoVisible(y=1)


    def plotShiftedForce(self):
        pg.setConfigOptions(antialias=True)
        plotWidget = pg.plot(title="CFC Force shifted to specimen")
        plotWidget.addLegend()
        
        plotWidget.plot(self.time, self.force, pen=pg.mkPen(1, width=2,), name="force from strain gauges")  ## setting pen=None disables line drawing
        self.shiftedForceLine = plotWidget.plot(self.time, self.PA_shifted, pen=pg.mkPen("g", width=2,), name="CFC shifted force")
        plotWidget.setLabel('left', 'force', units='kN')
        plotWidget.setLabel('bottom', 'force', units='kN')
        plotWidget.showGrid(x=True, y=True)
        #plotWidget.setXRange(self.rise_time,self.rise_time + self.tau)
        plotWidget.setAutoVisible(y=1)
        

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
        

if __name__ == "__main__":
    
    solver = solveCFC(path)
    #pg.exec()
    
    

