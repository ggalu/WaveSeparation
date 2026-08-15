# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-11-30 13:58:31
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-11-30 14:28:42
""" read time series data, do frquency analysis
"""

import numpy as np
from scipy import signal
import pylab as plt
import os, sys
global dt
dt = 1


def read_file(filename):
    zeroPad = True
    low_pass_filtering = False
    
    data = np.genfromtxt(filename)
    x = data[:,0]
    y = data[:,1]
    global dt
    dt = x[1] - x[0]
    N = len(x)
    print("read file [%s], dt=%g, Nrec=%d " % (filename, dt, N))
    
    if zeroPad == True:
        n = np.ceil(np.log(N)/np.log(2));
        m = int(2**(n+0))
        print("next power of 2 = %d, %d" %(n, m))
        
        diff= m - N
        y = np.pad(y, (0, diff), 'constant', constant_values=0.0)
        start = x[0]
        x = np.linspace(start, start+(m-1)*dt, num=m)
        dt = x[1] - x[0]
        print("padded with zeros, new x-axis extends from %g to %g, dt=%g" % (x[0], x[-1], dt))
        
        assert len(y) == m
        
    if low_pass_filtering == True:
        fs = 1. / dt
        fc = 20.  # 
        omega = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(1, omega, 'low')
        y = signal.filtfilt(b, a, y)
        z = signal.filtfilt(b, a, z)
        w = signal.filtfilt(b, a, w)
        v = signal.filtfilt(b, a, v)
        
    return x, y
    

def do_fft(t, s):
    
    dt = t[1] - t[0]
    fa = 1.0/dt # scan frequency
    print('dt=%g s (Sample Time)' % dt)
    print('fa=%.2f Hz (Frequency)' % fa)
    
    Y = np.fft.rfft(s) * dt
    X = np.fft.rfftfreq(len(s), d=dt)
    return X, Y

def writeData(path, freqs, alpha, k):
    """
    write frequencies, attenuation and wavenumber to file
    """
    
    outData = np.column_stack((freqs, alpha, k))
    filename = os.path.join(path, "dispersion_analysis.dat")
    np.savetxt(filename, outData, header="frequeny, alpha, k", comments="#")
    print("wrote data to file: ", filename)
    

filename = "A0.dat"
t, f0 = read_file(filename)

X, F0 = do_fft(t, f0)

# --- reconstruct k from C
C = 1.0 # wavespeed
k = 2*np.pi * X / C
k[0] = 0
gamma = 1.0j*k

shift = 0.05
    
F0shifted = F0 * np.exp(-gamma*shift)
f0shifted = np.fft.irfft(F0shifted) / dt

plt.plot(t, f0, "b-", label="A, original")
plt.plot(t, f0shifted, "r-", label="shifted, original")
plt.show()

