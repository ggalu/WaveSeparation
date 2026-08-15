''' Python Expample file for data read out from a TPC5 File
    Copyright: 2017 Elsys AG
    
    This example shows how to open a TPC5 file, read out the measurement data
    and the conversion values.
    The TestData.tpc5 contains 12 channels and 1 block per channel.
'''

import h5py
import numpy as np
from pylab import *

''' Import Helper Function for reading TPC5 Files '''
import tpc5


f = h5py.File("TestData.tpc5", "r")

''' Get Data scaled int voltage from channel 1 '''
dataset1 = tpc5.getVoltageData(f,1)
''' Get Data scaled in physical unit from channel 2'''
dataset2 = tpc5.getPhysicalData(f,2)


ch1 = tpc5.getChannelName(f, 1)
ch2 = tpc5.getChannelName(f, 2)

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

''' Make Plot '''
fig = plt.figure()
fig.suptitle('Imported Tpc5 File', fontsize=14, fontweight='bold')

''' Scale x Axis to ms '''
TimeScale = 1000

''' Build up X-Axis Array '''
startTime = -TriggerSample /SamplingRate * TimeScale
endTime   = (len(dataset1)-TriggerSample)/SamplingRate * TimeScale
t         = arange(startTime, endTime, 1/SamplingRate * TimeScale)


plot(t, dataset1, label=ch1)
plot(t, dataset2, label=ch2)

legend(framealpha=0.5)

if TimeScale == 1: 
    xlabel('time (s)')
elif TimeScale == 1000:
    xlabel('time (ms)')
elif TimeScale == 1000000:
    xlabel('time (us)')

ylabel(unit)

title('Recording Time: ' + RecDate + ' ' + RecTime)

grid(True)
savefig("TestDataPlot.png")
show()