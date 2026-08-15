# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-11-30 13:58:31
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-03 08:44:30
""" read time series data, do frquency analysis
"""

import numpy as np
from scipy import signal
import pylab as plt
import os, sys, pickle
global dt


def read_file(filename):
    zeroPad = True
    low_pass_filtering = False
    
    data = np.genfromtxt(filename)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    C = data[:,3]
    dt = x[1] - x[0]
    N = len(x)
    print("read file [%s], dt=%g, Nrec=%d " % (filename, dt, N))
    
    if len(x) % 2 != 0:
        x = x[:-1]
        y = y[:-1]
        z = z[:-1]
        C = C[:-1]
    
    
    print("Nrec", len(x))
    
    if zeroPad == True:
        n = np.ceil(np.log(N)/np.log(2)) + 1; 
        m = int(2**(n))
        print("next power of 2 = %d, %d" %(n, m))
        
        diff= m - N + 1
        y = np.pad(y, (0, diff), 'constant', constant_values=0.0)
        z = np.pad(z, (0, diff), 'constant', constant_values=0.0)
        C = np.pad(C, (0, diff), 'constant', constant_values=0.0)

        x = np.arange(m) * dt
        dt = x[1] - x[0]
        print("padded with zeros, new x-axis extends from %g to %g, dt=%g" % (x[0], x[-1], dt))
        
        print("new length of x:",len(x))
        print("new length of y:",len(y))

        #assert len(y) == m
        
    if low_pass_filtering == True:
        plt.plot(x, y, label="original")
        
        fs = 1. / dt
        fc = 50.0  # 
        omega = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(1, omega, 'low')
        y = signal.filtfilt(b, a, y)
        z = signal.filtfilt(b, a, z)

        plt.plot(x, y, label="filtered")
        plt.show()

    def mirror(seq):
        mirrored = seq[::-1]
        return np.concatenate((seq, mirrored))

    # apodize signals
    
    # make signal periodic
    #y = mirror(y)
    #z = mirror(z)
    #C = mirror(C)

    # reconstruct time axis
    #x = np.arange(y.size) * dt

    # cut signals above given time
    #tCut = 1.08
    #idx = (x > 1.08).nonzero()[0][0]
    #print("start inde: ", idx)
    ##[0]

    #x = x[:idx]
    #y = y[:idx]
    #z = z[:idx]
    #C = C[:idx]

    
        
    return x, y, z, C
    

def do_fft(t, s):
    
    #dt = t[1] - t[0]
    Y = np.fft.rfft(s)# * dt
    
    return Y

def writeData(path, freqs, alpha, k):
    """
    write frequencies, attenuation and wavenumber to file
    """
    
    outData = np.column_stack((freqs, alpha, k))
    filename = os.path.join(path, "dispersion_analysis.dat")
    np.savetxt(filename, outData, header="frequeny, alpha, k", comments="#")
    print("wrote data to file: ", filename)

def fourier_transform_1d(x, f):

    """
    Computes the continuous Fourier transform of function `func`, following the physicist's convention
    Grid x must be evenly spaced.

    Parameters
    ----------

    - func (callable): function of one argument to be Fourier transformed
    - x (numpy array) evenly spaced points to sample the function

    Returns
    --------
    - k (numpy array): evenly spaced x-axis on Fourier domain. Not sorted from low to high, unless `sort_results` is set to True
    - g (numpy array): Fourier transform values calculated at coordinate k
    """
    x0, dx = x[0], x[1] - x[0]

    def mirror(seq):
        mirrored = seq[::-1]
        return np.concatenate((seq, mirrored))
    
    f_mirrored = mirror(f)
    #plt.plot(f)
    #plt.plot(f_mirrored)
    #plt.show()
    #print("lengths", len(f), len(f_mirrored))
    
    F_mirrored = np.fft.rfft(f_mirrored) # DFT calculation
    w_mirrored = np.fft.rfftfreq(f_mirrored.size)*2*np.pi/dx # frequency normalization factor is 2*np.pi/dt
    

    F = np.fft.rfft(f) # DFT calculation
    w = np.fft.rfftfreq(f.size)*2*np.pi/dx

    #print("frequencies:", w.size, w_mirrored.size)

    #plt.loglog(w, abs(F), label="original")
    #plt.loglog(w_mirrored, abs(F_mirrored), label="mirrored")
    #plt.legend()
    #plt.show()
    
    return w_mirrored, F_mirrored


def inverse_fourier_transform_1d(func, k):
    """
    Computes the inverse Fourier transform of function `func`, following the physicist's convention
    Grid x must be evenly spaced.

    Parameters
    ----------

    - func (callable): function of one argument to be inverse Fourier transformed
    - k (numpy array) evenly spaced points in Fourier space to sample the function
    Returns
    --------
    - y (numpy array): evenly spaced x-axis. Not sorted from low to high, unless `sort_results` is set to True
    - h (numpy array): inverse Fourier transform values calculated at coordinate x
    """
    dk = k[1] - k[0]
    
    f = np.fft.rfft(func) * len(k) * dk /(2*np.pi)
    x = np.fft.rfftfreq(f.size)*2*np.pi/dk

    return x, f

filename = "ABC.dat"
t, epsA, epsB, epsC = read_file(filename)

# load strain gauge locations
a, b, c = pickle.load( open( "strain_gauge_locations.p", "rb" ) )
print("strain gauge position A", a)
print("strain gauge position B", b)
print("strain gauge position C", c)

if True:
    plt.plot(t, epsA, label="A")
    plt.plot(t, epsB, label="B")
    plt.plot(t, epsC, label="C")
    plt.legend()
    plt.show()


eta = 1.0e0
C = 5091.750772173155 # wavespeed
delta = b - a

dt = t[1] - t[0]
w = 2*np.pi * np.fft.rfftfreq(len(t), d=dt)
xi_dash = 1.0j * (w - 1.0j * eta) / C


#sys.exit()



EA = do_fft(t, epsA * np.exp(-eta * t))
#w, EA = fourier_transform_1d(epsA * np.exp(-eta * t), t)

EB = do_fft(t, epsB * np.exp(-eta * t))
#w, EB = fourier_transform_1d(epsB * np.exp(-eta * t), t)

xi_dash = 1.0j * (w - 1.0j * eta) / C
denominator = np.exp(-xi_dash * delta) - np.exp(xi_dash * delta)

Adash = (EB * np.exp(xi_dash * a) - EA * np.exp(xi_dash * b)) / denominator
Bdash = (EA * np.exp(-xi_dash * b) - EB * np.exp(-xi_dash * a)) / denominator

# shift o C
c = 1800
epsAscC = (np.fft.irfft(Adash * np.exp(-xi_dash * c))) * np.exp(eta * t)
epsDscC = (np.fft.irfft(Bdash * np.exp( xi_dash * c))) * np.exp(eta * t)

plt.plot(t, epsC, "g-", label="C measured")
#plt.plot(t, epsAscC, "b-", label="Asc")
#plt.plot(t, epsDscC, "r-", label="Dsc")
plt.plot(t, epsAscC + epsDscC, "r-", label="sum")


plt.legend()
plt.show()

sys.exit()


X, F0 = do_fft(t, epsA)

# --- reconstruct k from C


shift = 0.05
    
F0shifted = F0 * np.exp(-gamma*shift)
ashifted = np.fft.irfft(F0shifted) / dt

plt.plot(t, epsA, "b-", label="A, original")
plt.plot(t, epsB, "g-", label="B, original")
plt.plot(t, ashifted, "r-", label="shifted, original")
plt.show()

