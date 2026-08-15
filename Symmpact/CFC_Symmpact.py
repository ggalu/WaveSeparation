# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-09 08:20:48
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-04-07 21:41:30

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
from PyQt5.QtGui import QFont
import os, sys, pickle
import bottleneck as bn
import scipy.signal
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
import pylab as plt
#global dt


#path = "/home/gcg/Projekte/21_WaveSeparation/2025-01-30_Waveseparation/02_PC"; v0=5.0
#path = "/home/gcg/Projekte/21_WaveSeparation/2025-03-27_Waveseparation/1_plain_pulse_1bar/"; v0=3.55, TD -0.009
path = "/home/gcg/Projekte/21_WaveSeparation/2025-03-27_Waveseparation/2_blue_foam_1bar"; v0=3.4; delay = 0.0
#path = "/home/gcg/Projekte/21_WaveSeparation/2025-03-27_Waveseparation/4_PC_1bar"; v0=4.2
#path = "/home/gcg/Projekte/21_WaveSeparation/2025-03-27_Waveseparation/5_PC_2bar"; v0=6.1; delay=0.001
#path = "/home/gcg/Projekte/21_WaveSeparation/2025-03-27_Waveseparation/6_PC_3bar"; v0=7.85 # TD is -0.096 ms
#path = "/home/gcg/Coding/WaveSeparation/Symmpact"; v0 = 5.0
L0 = 27.0 # INITIAL LENGTH OF SPECIMEN
A0 = 0.25 * np.pi * 27**2 # INITIAL CROSS-SECTIOn AREA OF SPECIMEN

class solveCFC:
    def __init__(self, path):

        self.path = path
        self.rho = 2.7e-6
        self.E_bar = 70.0
        self.A_bar = 0.25 * np.pi * 40**2
        self.L_bar = 1900.0
        self.c0 = np.sqrt(self.E_bar/self.rho)
        self.L0 = L0 # specimen length
        self.A0 = A0 # specimen area
        self.alpha = 0.0 #1.2e-5 # This the attenuation factor. For Aluminium, this makes the force skewed in time
        self.smoothingFactor = 1.0e-4
        self.nsmooth = 100
        self.specimen_diameter = 10.0
        self.delay = delay # time delay between linescan and strain gauge data
        self.shift = 131.0 # shifting distance from strain gauge to specimen
        self.calibration_factor= None #0.00956204 # conversion factor between force and velocity
        self.significant_force_level = 0.1
        self.firstCall = True     
        self.v0 = v0
        self.velocity_fudge_factor = 1.0


        self.ROIstart, self.ROIstop = 0.0, 2.0

        self.LoadDisplacementOverTime(path)
        self.LoadForceOverTime(path)
        self.Interpolate()
        #self.CalibrateVelocity()
        self.calculate_F_G_unshifted()
        #self.resolveAtDistanceFrequencyDomain(120.0)
        self.resolveAtDistanceTimeDomain()
        self.computeStressStrain()
        

        #self.plotForce()
        #self.plotShiftedForce()
        
        
        self.createWidgets()

        #self.plot_AB()
        #self.plot_FG()
        #self.plot_strain()
        #self.plot_specimen()
        #self.plot()

    def upd_drag(self, line):
        pos = line.pos()[0]

        print("pos:", pos)

    
        


    def createWidgets(self):
        """
        This routine is called whenever the additional shift value between line scan and strain gauge time axes changes.
        """
        def updatePlot():
            """
            Run through all rreqquired steps to produce the shifted CFC anaylsis force.
            """
            self.Interpolate() # uses updated delay between line scan and strain gauge data
            self.calculate_F_G_unshifted()
            ##self.resolveAtDistanceFrequencyDomain(-110.0)
            self.resolveAtDistanceTimeDomain() # uses upaded shifting distance
            self.computeStressStrain()

            self.shiftedForceLine.setData(self.time, self.PA_shifted)
            self.lineScanVelocityLine.setData(self.time, self.v)
            
            # plot 3
            self.specimenStressStrainLine.setData(self.specimen_strain, self.specimen_stress)

            # plot 4
            self.velocityIn_Line.setData(self.time, self.velLeft)
            self.velocityOut_Line.setData(self.time, self.vA_shifted)

        def valueChanged_ROI():
            self.ROIstart, self.ROIstop = ROI.getRegion()
            #updatePlot()
            #print("ROI start, stop:", self.ROIstart, self.ROIstop)
            self.computeStressStrain()
            self.specimenStressStrainLine.setData(self.specimen_strain, self.specimen_stress)
            pass


        def valueChanged_delay(spinbox):
            self.delay = spinbox.value()
            print("NEW DELAY:", self.delay)
            updatePlot()

        def valueChanged_v0(spinbox):
            self.v0 = spinbox.value()
            print("NEW V0:", self.v0)
            updatePlot()

        def valueChanged_shift(spinbox):
            self.shift = spinbox.value()
            updatePlot()

        def valueChanged_calibrationFactor(spinbox):
            self.velocity_fudge_factor = spinbox.value()
            #print("updateing calib factor", self.calibration_factor)
            updatePlot()

        def clickedBtnSave():
            outData = np.column_stack((self.time, self.specimen_strain, self.specimen_stress, self.specimen_displacement, self.PA_shifted, self.velLeft, self.vA_shifted))
            filename = os.path.join(self.path, "CFC_stress_strain.txt")
            np.savetxt(filename, outData, header="time, strain, stress, displacement, force, vel_in, vel_out")
            print("wrote time, stress, strain, displacemnt, force to file: ", filename)

            msgBox = QtWidgets.QMessageBox()
            msgBox.warning(self.win, 'Information',"Saved Data")

        


        #if self.firstCall:
        #    self.CalibrateVelocity()
        #    self.Interpolate() # uses updated delay between line scan and strain gauge data
        #    self.calculate_F_G_unshifted()
        #    self.resolveAtDistanceTimeDomain()
        #    self.firstCall = False


        self.app = pg.mkQApp("SpinBox Example")
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('CFC Analysis')
        centralWidget = QtWidgets.QWidget()
        self.win.setCentralWidget(centralWidget)
        #win.resize(1920,800)
        self.win.showMaximized()

        layoutH0 = QtWidgets.QHBoxLayout()
        
    
        layoutVL = QtWidgets.QVBoxLayout()
        layoutH0.addLayout(layoutVL, 0)
    
        layoutVR = QtWidgets.QVBoxLayout()
        layoutH0.addLayout(layoutVR, 2)

        layout3rdColumn = QtWidgets.QVBoxLayout()
        layoutH0.addLayout(layout3rdColumn, 2)
        centralWidget.setLayout(layoutH0)

        # plotting widgets

        
        pg.setConfigOptions(antialias=True)

        plotWidget = pg.plot(title="Force-Velocity consitency check")
        plotWidget.setTitle("Force-Velocity consistency check", size="20pt")
        plotWidget.addLegend()
        plotWidget.plot(self.time, self.force / (self.rho * self.c0 * self.A_bar), pen=pg.mkPen(1, width=2,), name="velocity from strain gauges")  ## setting pen=None disables line drawing
        self.lineScanVelocityLine = plotWidget.plot(self.time, self.v, pen=pg.mkPen("g", width=2,), name="linescan velocity")
        plotWidget.setLabel('left', 'velocity', units='m/s')
        plotWidget.setLabel('bottom', 'time', units='ms')
        plotWidget.showGrid(x=True, y=True)
        #plotWidget.setXRange(self.rise_time,self.rise_time + self.tau)
        plotWidget.setAutoVisible(y=1)
        

        # add second plot 
        pg.setConfigOptions(antialias=True)
        plotWidget2 = pg.plot(title="CFC Force shifted to specimen")
        plotWidget2.addLegend()
        plotWidget2.plot(self.time, self.force, pen=pg.mkPen("r", width=2,), name="force from strain gauges")  ## setting pen=None disables line drawing
        self.shiftedForceLine = plotWidget2.plot(self.time, self.PA_shifted, pen=pg.mkPen("g", width=2,), name="CFC shifted force")
        plotWidget2.setLabel('left', 'force', units='kN')
        plotWidget2.setLabel('bottom', 'time', units='ms')
        plotWidget2.showGrid(x=True, y=True)
        plotWidget2.setAutoVisible(y=1)
        ROI = pg.LinearRegionItem([self.ROIstart, self.ROIstop])
        ROI.setZValue(10)
        plotWidget2.addItem(ROI)
        ROI.sigRegionChanged.connect(valueChanged_ROI)


        # add third plot
        plotWidget3 = pg.plot(title="specimen")
        plotWidget3.addLegend()
        plotWidget3.setTitle("specimen")
        #self.FA_Line = plotWidget3.plot(self.time, self.FA, pen=pg.mkPen("r", width=2,), name="FA")
        #self.GA_Line = plotWidget3.plot(self.time, self.GA, pen=pg.mkPen("g", width=2,), name="GA")
        #self.vrel_Line = plotWidget3.plot(self.time, self.vrel, pen=pg.mkPen("g", width=2,), name="vrel")
        #self.vrel_Line = plotWidget3.plot(self.time, self.specimen_strain, pen=pg.mkPen("g", width=2,), name="vrel")
        self.specimenStressStrainLine = plotWidget3.plot(self.specimen_strain, self.specimen_stress, pen=pg.mkPen("g", width=2,), name="interface displacement")

        self.LoadSimulatedSpecimen(path)
        if self.simulation_stress is not None:
            plotWidget3.plot(self.simulation_eps, self.simulation_stress*1e9, pen=pg.mkPen("r", width=2,), name="simulation")


        plotWidget3.setLabel('left', 'nominal stress', units='Pa')
        plotWidget3.setLabel('bottom', 'nominal compressive strain', units='-')
        plotWidget3.showGrid(x=True, y=True)
        plotWidget3.setAutoVisible(y=1)

        # add 4th plot -- velocity of interfaces
        plotWidget4 = pg.plot(title="interface velocity")
        plotWidget4.addLegend()
        plotWidget4.setTitle("interface velocity")
        v_line = pg.InfiniteLine(angle=90, movable=True)
        v_line.sigPositionChanged.connect(self.upd_drag)
        plotWidget4.addItem(v_line)
        #
        self.velocityIn_Line  = plotWidget4.plot(self.time, self.velLeft, pen=pg.mkPen("g", width=2,), name="velocity in")
        self.velocityOut_Line = plotWidget4.plot(self.time, self.vA_shifted, pen=pg.mkPen("r", width=2,), name="velocity out")
        plotWidget4.setLabel('left', 'velocity', units='m/s')
        plotWidget4.setLabel('bottom', 'time', units='ms')
        plotWidget4.showGrid(x=True, y=True)
        plotWidget4.setAutoVisible(y=1)

        
        spin_delay = pg.SpinBox(value=self.delay, bounds=[None, None], finite=True, suffix="ms", step=0.001)
        #spin_delay.setAlignment(QtCore.Qt.AlignTop)
        spin_delay.sigValueChanged.connect(valueChanged_delay)

        spin_shift = pg.SpinBox(value=self.shift, int=True, minStep=1, step=1, bounds=[None, None], finite=True, suffix='mm')
        spin_shift.sigValueChanged.connect(valueChanged_shift)

        spin_calibrationFactor = pg.SpinBox(value=self.velocity_fudge_factor, step=0.005,  bounds=[None, None], finite=True)
        spin_calibrationFactor.sigValueChanged.connect(valueChanged_calibrationFactor)

        spin_v0 = pg.SpinBox(value=self.v0, bounds=[None, None], finite=True , step=0.05)
        spin_v0.sigValueChanged.connect(valueChanged_v0)

        spin_L0 = pg.SpinBox(value=self.L0, bounds=[None, None], finite=True , step=0.1)
        


        layoutVR.addWidget(plotWidget2, stretch=3, alignment=QtCore.Qt.AlignTop)
        layoutVR.addWidget(plotWidget, stretch=3)

        layout3rdColumn.addWidget(plotWidget3, stretch=3)
        layout3rdColumn.addWidget(plotWidget4, stretch=3)

        layoutVL.addWidget(QtWidgets.QLabel("linescan to strain gauge delay:"))
        layoutVL.addWidget(spin_delay)

        label = QtWidgets.QLabel()
        label.setFrameStyle(QtWidgets.QFrame.HLine)
        label.setLineWidth(1)
        layoutVL.addWidget(label)

        layoutVL.addWidget(QtWidgets.QLabel("shifting distance:"))
        layoutVL.addWidget(spin_shift)
        layoutVL.addWidget(label)
        layoutVL.addWidget(QtWidgets.QLabel("velocity calibration factor:"))
        layoutVL.addWidget(spin_calibrationFactor)
        layoutVL.addWidget(label)
        
        layoutVL.addWidget(QtWidgets.QLabel("input bar velocity at t0:"))
        layoutVL.addWidget(spin_v0)
        layoutVL.addWidget(label)

        layoutVL.addWidget(QtWidgets.QLabel("specimen L0:"))
        layoutVL.addWidget(spin_L0)
        layoutVL.addWidget(label)
        
        btnSave = QtWidgets.QPushButton('save shifted force')
        btnSave.clicked.connect(clickedBtnSave)
        layoutVL.addWidget(btnSave)

        layoutVL.addStretch()
        
        pg.exec()

    def CalibrateVelocity(self):
        """
        linescan velocity self.v_px is given in units of pixels/time.
        We need to establish a claibration factor to render this in physical units.
        We derive the calibration factor from  \sigma = \rho \c_0 U_p, i.e., by
        requiring that c_0 \varepsilon = Up. This is true only within the first wave transit time.
        """

        print("\n------------- VELOCITY - FORCE CALIBRATION --------------")

        # establish first wave transit time
        import tools
        rise_time_index = tools.find_TTL(self.force, direction="positive", level=self.significant_force_level)
        self.rise_time = self.time[rise_time_index] + 0.1
        self.tau = 2 * self.L_bar / self.c0 - 0.1
        print(f"rise time of 1st wave transit is {self.rise_time}, duration is {self.tau}")
        
        #if self.calibration_factor == None:
        if True:

            # require that the mean of c_0 \varepsilon equals Up
            startIndex = np.argmax(self.time > self.rise_time)
            stopIndex  = np.argmax(self.time > self.rise_time + self.tau)
    
            print(f"discrete interval: {self.time[startIndex]} -- {self.time[stopIndex]}")

            c0eps_mean = self.c0 * np.mean(self.eps[startIndex:stopIndex]) * self.velocity_fudge_factor
            Up_mean = np.mean(self.v_px[startIndex:stopIndex])
            print(f"mean of c0.eps is {c0eps_mean}, mean of Up is {Up_mean}")
            self.calibration_factor = c0eps_mean / Up_mean
            print("calibration factor is:", self.calibration_factor)

        #self.v = self.v_px * self.calibration_factor # this needs to be done Interpolate()

        print("------------- END VELOCITY - FORCE CALIBRATION --------------\n")

    def Interpolate(self):
        """
        define common time axis between strain gauge and line scan datasets.
        Interpolate line scan data to strain gauge time basis.
        
        Additional shift refers to a time shift between the line scan data and the force data.
        This shift needs to be dialled such that the resulting force curve is as smooth as possible at the wave transits.
        """

        

        # shift linescan data according to delay
        print("*** delay shift:", self.delay)
        self.time_line_delayed = self.time_line + self.delay

        start= max(self.time_force[0], self.time_line_delayed[0])
        stop = min(self.time_force[-1], self.time_line_delayed[-1])
        print(f"Common time range: {start} -- {stop}")

        # create a spline interpoland of the line scan data
        #
        spl_linescan = make_smoothing_spline(self.time_line_delayed, self.u_px, lam=self.smoothingFactor)

        # create a common time axis with an even number of data points -- required beacuse we do FFT later on
        self.dt = self.time_force[1] - self.time_force[0]
        N = int( (stop - start) / self.dt)
        if N % 2 != 0: N -= 1
        self.time = np.linspace(start, stop, N, endpoint=False)
        print(f"requested dt is {self.dt}, actual dt is {self.time[1] - self.time[0]}, length of time axis is {N}")

        # now we need to assert that the common time axis self.time is contained in both line scan and force time axes
        assert(self.time[0] >= self.time_force[0])
        assert(self.time[0] >= self.time_line_delayed[0])
        assert(self.time[-1] <= self.time_force[-1])
        assert(self.time[-1] <= self.time_line_delayed[-1])


        # create frequency axis for this time axis
        self.w = 2*np.pi * np.fft.rfftfreq(len(self.time), d=self.dt)
        self.gamma = self.alpha + 1.0j * self.w / self.c0

        

        # compute strain from force data and interpolate to new time axis
        #print("length and dtype of time_force, force", len()
        spl_force = make_smoothing_spline(self.time_force, self.force_original, lam=self.smoothingFactor)
        #spl_force = make_smoothing_spline(self.time_force, self.force_original)
        self.force = spl_force(self.time)
        self.eps = self.force / (self.A_bar * self.rho * self.c0**2)

        # interpolate velocity data to new time axis. Thi svelcoity is in pixel units, 
        # the physical velocity is obtained subsequently after calling self.CalibrateVelocity()
        self.v_px = spl_linescan(self.time, nu=1)

        self.CalibrateVelocity()

        self.v = self.v_px * self.calibration_factor

        assert len(self.v) == len(self.force)

        #plt.plot(self.v_px)
        #plt.show()


    def computeStressStrain(self):
        
        # plot the relative motion of the bar-specimen interfaces
        
        
        self.vrel = self.v0 - 2 * self.vA_shifted
        self.velLeft = self.v0 - self.vA_shifted

        # the realtive displacement between the bar interfaces is only relevant during the time when the specimen is loaded.
        # this time is given by ROI start, stop

        self.vrel[self.time  < self.ROIstart] = 0.0
        self.vrel[self.time  > self.ROIstop] = 0.0

        self.specimen_displacement = np.cumsum(self.vrel) * self.dt
        self.specimen_strain = self.specimen_displacement  / self.L0
        
        self.specimen_stress = 1.0e9 * self.PA_shifted / self.A0 # this is in Pa
        self.specimen_stress[self.time  < self.ROIstart] = 0.0 # zero out forces outside ROI
        self.specimen_stress[self.time  > self.ROIstop] = 0.0
        

    def LoadDisplacementOverTime(self, path):
        data = np.genfromtxt(os.path.join(path, "linescan_analysis.dat"))
        self.time_line = data[:,0] #/ 200.0
        self.u_px = data[:,1] # pixels

        #plt.plot(self.time_line, self.u_px)
        #plt.show()


    def LoadForceOverTime(self, path):
        data = np.genfromtxt(os.path.join(path, "Symmpact_time_force.txt"))
        self.time_force, self.force_original = data[:,0], data[:,1]

    def LoadSimulatedSpecimen(self, path):
        filename = os.path.join(path, "specimen.dat")
        if os.path.exists(filename):
            data = np.genfromtxt(filename)
            self.simulation_eps, self.simulation_stress = -data[:,2], -data[:,1]
            print("... successfully loaded simulated data")
        else:
            self.simulation_eps = None
            self.simulation_stress = None



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
        self.vA_shifted =            self.c0    * (self.FA_shifted - self.GA_shifted)

    def resolveAtDistanceFrequencyDomain(self, dA):

        print("shifting distance d:", dA)

        def shift_P(Fw, Gw, d):
            """
            Shift the sum of F and G by d.
            Do this by propagating the sum of the Fourier transforms F(w) and G(w).
            Return the inverse transform, i.e., the time-domain signal of the force P(t)
            """

            bracket = Fw * np.exp(self.gamma * d) + Gw * np.exp(-self.gamma * d)
            prefactor = -self.rho * (self.w**2 / (self.gamma**2)) * self.A_bar # here, we divide by gamma. gamma[0] is 0 -- problem!
            prefactor[0] = 0.0 # deal with the divide-by-zero problem

            #print("self.w[0]", self.w[0])
            #print("self.gamma[0]", self.gamma[0])
            #sys.exit()

            return np.fft.irfft(prefactor * bracket)
        
        def shift_v(Fw, Gw, d):
            """
            Shift the difference of F and G by d.
            Do this by propagating the difference of the Fourier transforms F(w) and G(w).
            Return the inverse transform, i.e., the time-domain signal of the velocity v(t)
            """

            bracket = Fw * np.exp(-self.gamma * d) - Gw * np.exp(self.gamma * d)
            prefactor = 1.0j * self.w / self.gamma # here, we divide by gamma. gamma[0] is 0 -- problem!
            prefactor[0] = 0.0 # deal with the divide-by-zero problem

            return np.fft.irfft(prefactor * bracket)


        #FA and GA might need to be tapered

        from scipy.signal.windows import blackman
        N = len(self.FA)
        w = blackman(N)


        #plt.plot(self.FA*w)
        #plt.plot(self.GA*w)
        #plt.show()
        #sys.exit()

        FAw = np.fft.rfft(self.FA*w)
        GAw = np.fft.rfft(self.GA*w)

        # forces
        self.PA_shifted    = shift_P(FAw, GAw, dA)

        # remove linear bias from PA_shifted
        #self.PA_shifted = signal.detrend(self.PA_shifted)

        self.PA_shifted -= self.PA_shifted[0] # restore DC component

        if len(self.PA_shifted) != len(self.time):
            print("len of time axis after shifting is different", len(self.PA_shifted), len(self.time))
            sys.exit(1)

        # velocities
        self.vA_shifted = shift_v(FAw, GAw, dA)
        self.vA_shifted -= self.vA_shifted[0]

        #self.vB_shifted = shift_v(self.FAw, self.GAw, 2*dA)
        #self.vB_shifted -= self.vB_shifted[0]


    def calculate_F_G_unshifted(self):
        """
        Calculate the unshifted time domain signals F(t) and G(t).
        Unshifted means that they are calculated at the locations of the strain gauge.
        Also perform filtering.
        """


        print(f"in calculate_F_G_unshifted: length of eps={len(self.eps)}, length of v={len(self.v)}")
        smooth_eps = savgol_filter(self.eps, self.nsmooth, 1)
        smooth_v   = savgol_filter(self.v, self.nsmooth, 1)
        self.FA = 0.5 * (smooth_eps + smooth_v / self.c0)
        self.GA = 0.5 * (smooth_eps - smooth_v / self.c0)

        #self.FA = 0.5 * (self.eps + self.v / self.c0)
        #self.GA = 0.5 * (self.eps - self.v / self.c0)
        

    def plotInputSignals(self):
        #plt.title("1st wave transit: check that forces agree!")
        pg.setConfigOptions(antialias=True)
        plotWidget = pg.plot(title="Force-Velocity consitency check")
        plotWidget.addLegend()
        
        plotWidget.plot(self.time, self.force, pen=pg.mkPen(1, width=2,), name="force from strain gauges")  ## setting pen=None disables line drawing
        self.lineScanVelocityLine = plotWidget.plot(self.time, self.v * self.rho * self.c0 * self.A_bar, pen=pg.mkPen("g", width=2,), name="force from velocity")

        plotWidget.setLabel('left', 'force', units='kN')
        plotWidget.setLabel('bottom', 'force', units='kN')
        plotWidget.showGrid(x=True, y=True)
        plotWidget.setXRange(self.rise_time,self.rise_time + self.tau)
        plotWidget.setAutoVisible(y=1)


    def plotOutputSignals(self):
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
    
    

