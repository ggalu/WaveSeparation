# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-05-01 17:17:57
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-27 22:04:43

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import interpolate
import tools
import os
import sys
import glob

# ************************************************* 
#
# Note:
# This version is only for setup with strain gauge on out1,
# i.e., no strain gauge on striker
#
# *************************************************


sys.path.append("../") 


path = "/home/gcg/Coding/WaveSeparation/Symmpact/2024-01-27_Waveseparation/01"

#filename_in = sys.argv[1]
#print("input filename:", filename_in)



shiftThreshold = 0.5
validDuration = 2 * 1900 / 5090 # valid Duration of experiment -- wave transit time in bars. This is for Alu, c0 = 5090 mm / ms

class AnalyzeSHTBExperiment():
    
    
    def __init__(self, path):
        
        if not os.path.exists(path):
            print("supplied path %s does not exist, exiting.")
            sys.exit()
            
        self.path = path
        self.shiftThreshold = shiftThreshold
        self.readData_TranAx()

        flashIndex = tools.find_TTL(self.marker, 0.5)
        flashTime = self.time[flashIndex]
        print("flash time:", flashTime)

        self.convertVoltsToForceAndStrain()
        #self.cutData()
        self.writeNonShiftedData()
        
        #self.time -= self.tRiseF1 # !!! OFFSET time origin to force signal rise !!!
        #self.wtime -= self.tRiseF1 # !!! OFFSET time origin to force signal rise !!!
        #self.plotWindowedForces()
        #self.writeData("Symmpact_time_force.dat")
        
    
    def cutData(self):
        """
        cut out the relevant portion of the signal, i.e., during the first wave transit time
        """
        
        self.tRise = 0
        print("N=", self.Nrec)
        for i in range(0, self.Nrec):
            #print(self.force_out1[i])
            if self.force_out1[i] > self.shiftThreshold:
                self.tRise = self.time[i]
                break
        
        #assert self.tRise > 0, "unable to find rise above shift threshold = %f in output bar signal" % (self.shiftThreshold)
        
        indices = np.where(np.logical_and(self.time >= self.tRise - 0.1 , self.time < self.tRise + validDuration))
        self.time = self.time[indices]
        self.force_out1 = self.force_out1[indices]
        plt.plot(self.time, self.force_out1)
        plt.xlabel("time [ms]")
        plt.ylabel("force [kN]")
        plt.show()
    
    def writeNonShiftedData(self):
        """
        write out time and transmitted force, before applying any shifts.
        The intent is to use this date with the area camera, which is triggered to start at
        the exact same time as the force data acquisition         
        """
        
        outData = np.column_stack((self.time, self.force_out1))
        
        filename = os.path.join(self.path, "Symmpact_time_force.txt")
        np.savetxt(filename, outData, header="time, force")
        print("wrote unshifted time and transmitted_force to file: ", filename)
        
    def convertVoltsToForceAndStrain(self):
        """
        
        """
        
        # for ALU bars

        
        # --- amplifier settings ---
        gain = 100 # this is the gain factor for the metal foil strain gauges
        Ub = 10.0 # Versorgungsspannung in Volt (-5 V -- + 5 V)
        E = 70 # bar youngs modulus
        A = np.pi * 20**2 # bar cross section area
        
        # calculate theoretcial conversion factor based on symmetric half-bridge
        GF = 2.0 # metal foil manunfacturer's strain gauge factor -- only used for comparison purposes
        factor = -1.0 / (2.0 * E * A / (GF * Ub * gain))
        print("theoretical conversion factor based on manufacturer's gauge factor: ", factor )
        
        #self.volt_out1 *= factor # undo the application of a calbration factor directly in tranAx
        #plt.plot(self.time, self.volt_out1)
        #plt.show()
        self.force_out1 = self.volt_out1 /  0.0129835 # calibration 13. Oct 2021 Alu

        plt.plot(self.time, self.force_out1)
        plt.show()
    
    def readData_TranAx(self):
        """
        read TranAx file for Symmpact experiment.
        Remove fixed voltage baseline assuming that tra signal should be zero during the first 1% of record length.
        Assume the following layout:
        
            
        * TPC5 format with channels:
            1 -- strain gauge on input bar / striker
            2 -- strain gauge on output bar
            
        """
        import h5py
        from TPC5_IO import tpc5


        os.chdir(path)
        files = glob.glob("*.tpc5")
        print(files)
        if not (len(files) == 1):
            print("There are have multiple .tpc5 files in the working directory.")
            print("This program expects only a single .tpc5 file containing TranAx data.")
            sys.exit(1)

        print(files[0])
        
        f = h5py.File(files[0], "r")

        ''' Get Data scaled int voltage from channel 1 '''
        dataset1, m1  = tpc5.getVoltageData(f,1)
        dataset2, m2  = tpc5.getVoltageData(f,2)
        dataset3, m3  = tpc5.getVoltageData(f,3)
        dataset4, m4  = tpc5.getVoltageData(f,4)

        ch1 = tpc5.getChannelName(f, 1)
        ch2 = tpc5.getChannelName(f, 2)
        ch3 = tpc5.getChannelName(f, 3)
        ch4 = tpc5.getChannelName(f, 4)
        
        ''' Build Unit String '''
        unit = '[ ' + tpc5.getPhysicalUnit(f,1) + ' ]'
        
        ''' Get Time Meta Data TriggerSample number and Sampling Rate '''
        TriggerSample = tpc5.getTriggerSample(f,1,1)
        SamplingRate  = tpc5.getSampleRate(f,1,1)
        
        ''' Get Absolute recording time and split it up '''
        RecTimeString = tpc5.getStartTime(f,1,1)
        RecTimeList   = RecTimeString.split('T',1)
        RecDate       = RecTimeList[0]
        TimeListe     = RecTimeList[1].split('.',1)
        RecTime       = TimeListe[0] 

        f.close
        
        ''' Scale x Axis to ms '''
        TimeScale = 1000
        
        ''' Build up X-Axis Array '''
        startTime = -TriggerSample /SamplingRate * TimeScale
        endTime   = (len(dataset1)-TriggerSample)/SamplingRate * TimeScale
        self.time = np.arange(startTime, endTime, 1/SamplingRate * TimeScale)
        
        print("\nStart time is %g, end time is %g, Sampling rate is %g MHz" % (startTime, endTime, SamplingRate / 1.0e6))
        print("TriggerSample", TriggerSample, len(dataset1))
        
        self.marker = m1
        self.sampleRate = SamplingRate / TimeScale
        self.volt_in  = dataset1
        self.volt_out1 = dataset2
        self.volt_out2 = dataset3
        self.light_barrier = dataset4
        self.Nrec = len(self.time)
        self.sampleDt = self.time[1] - self.time[0]
        
        
        # zero first 1 percent of force signal
        #preTriggerRatio = (0.5*TriggerSample)/len(dataset1)
        preTriggerRatio = 0.05
        tools.zeroSignal(self.volt_in, preTriggerRatio)
        tools.zeroSignal(self.volt_out1, preTriggerRatio)
        tools.zeroSignal(self.volt_out2, preTriggerRatio)



    
if __name__ == "__main__":
    AnalyzeSHTBExperiment(path)
                        
