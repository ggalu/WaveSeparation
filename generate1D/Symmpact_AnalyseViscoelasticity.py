""" read time series data, do frquency analysis
"""

import numpy as np
from scipy import signal
import pylab as plt
import os, sys
global dt
from smooth import smooth
dt = 1


def gen_test_data():
    """ Generate synthetic test data to check FFT """
    x = np.linspace(-3, 3, num=6000)
    global dt
    dt = x[1] - x[0]
    N = len(x)
    y = np.zeros(N)
    z = np.zeros(N)
    
    def kernel(x, x0, sigma=1.0e-2):
        #return np.exp(-sigma * (x-x0)**2)
        a = 1./np.sqrt(2*np.pi*sigma**2)
        return a*np.exp(-(x-x0)**2 /(2*sigma**2))
    
    start = 0
    stop = 1
    for i in range(N):
        y[i] = kernel(x[i], 0.0)
        #if x[i] > stop:
        #    y[i] = kernel(x[i], stop)
        #elif x[i] < start:
        #    y[i] = kernel(x[i], start)
        #else:
        #    y[i] = 1.0
    z = y
    return x, y, z
    

def read_file(path):
    zeroPad = True
    low_pass_filtering = True
    
    data = np.genfromtxt(os.path.join(path, "windowed_signals.dat"))
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    w = data[:,3]
    v = data[:,4]
    global dt
    dt = x[1] - x[0]
    N = len(x)
    print("read file in [%s], dt=%g, Nrec=%d " % (path, dt, N))
    
    if zeroPad == True:
        n = np.ceil(np.log(N)/np.log(2));
        m = int(2**(n+0))
        print("next power of 2 = %d, %d" %(n, m))
        
        diff= m - N
        y = np.pad(y, (0, diff), 'constant', constant_values=0.0)
        z = np.pad(z, (0, diff), 'constant', constant_values=0.0)
        w = np.pad(w, (0, diff), 'constant', constant_values=0.0)
        v = np.pad(v, (0, diff), 'constant', constant_values=0.0)
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
        
    return x, y, z, w, v
    

def do_fft(t, s):
    
    dt = t[1] - t[0]
    fa = 1.0/dt # scan frequency
    print('dt=%g ms (Sample Time)' % dt)
    print('fa=%.2f kHz (Frequency)' % fa)
    
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
    

path = r"C:\Users\gcg\Documents\Projekte\Symmpact\Daten\1Bar_PlainPulse_4.3velocity"
path = r"C:\Users\gcg\Documents\Projekte\Symmpact\Daten\3Bar_SikaBlockGreen01"
Nsmooth = 0
t, f0, f1, f2, favg = read_file(path)

f0 = 0.5*(f0 + f1)

#x, y, z = gen_test_data()
X, F0 = do_fft(t, f0)
X, F1 = do_fft(t, f1)
X, F2 = do_fft(t, f2)
X, Favg = do_fft(t, favg)
# ---- now we have the FT
F0mag = np.abs(F0)
F1mag = np.abs(F1)
F2mag = np.abs(F2)
N = len(X)
R = np.zeros(N)        
for i in range(N):
    if F1mag[i] > 0 and F2mag[i] > 0:
        R[i] = F1mag[i] / F2mag[i]
    else:
        R[i] = 1.0e-16
assert R.all() > 0
dx = 935.0 # distance of strain gauges, mm
alpha = np.log(R)

alpha /= dx # attenuation factor
#alpha = smooth(alpha, window_len=Nsmooth, window='flat')

phase_difference = np.unwrap(np.angle(F1) - np.angle(F2)) # ensures montonic increase
k = phase_difference / dx # wavenumber
C = 2*np.pi*X / (k+1.0e-16) # wave speed in frequency domain
C[0] = C[1]
#C = smooth(C, window_len=Nsmooth, window='flat')
c0 = C[0]
print("c0 = ", c0, 1.2e-6*c0**2)

# --- reconstruct k from C
k = 2*np.pi * X / C
k[0] = 0
gamma = alpha + 1.0j*k

index = None
for i in range(len(X)):
    if X[i] > 20.0:
        index = i
        break
print("frequency 20 kHz is at index %d" % (index))

alpha[index::] = 0
k[index::] = 0
gamma[index::] = 0

#writeData(path, X, alpha, k)




#
# --- plot stuff in frequency domain
#

if 0:
    plt.plot(X, k)
    #plt.plot(X, Ymag, label="amplitude of Y")
    #plt.plot(X, Zmag, label="amplitude of Z")
    plt.xlim((0,20))
    plt.legend()
    plt.show()
    
    
if 1:
    
    fig, ax1 = plt.subplots()
    ax1.plot(X, C, 'b-')
    ax1.set_xlabel('frequency / kHz')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('wave speed / m/s', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim((0,2000))

    ax2 = ax1.twinx()
    ax2.plot(X, 1000*alpha, 'r.')
    ax2.set_ylabel('attenuation coefficient / (1/m)', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim((-1,5))
    
    fig.tight_layout()
    plt.xlim((0,25))
    plt.show()

#
# --- perform wave shift
#
if 1:
    shift = 120.0
    
    F0shifted = F0 * np.exp(gamma*shift)
    f0shifted = np.fft.irfft(F0shifted) / dt
    
    F1shifted = F1 * np.exp(gamma*shift)
    f1shifted = np.fft.irfft(F1shifted) / dt
        
    F2shifted = F2 * np.exp(gamma*(dx+shift))
    f2shifted = np.fft.irfft(F2shifted) / dt
    
    Favgshifted = Favg * np.exp(gamma*(shift))
    favgshifted = np.fft.irfft(Favgshifted) / dt
    
    #plt.plot(t, f0, "b-", label="strain gauge 0, original")
    plt.plot(t, f1, "g-", label="strain gauge 1, original")
    plt.plot(t, f2, "r-", label="strain gauge 2, original")
    #plt.plot(t, favg, "r-", label="average, original")
    #plt.plot(t, f0shifted+1,  "b--", label="strain gauge 0, shifted to interface")
    plt.plot(t, f1shifted+1, "g--", label="strain gauge 1, shifted to interface")
    plt.plot(t, f2shifted+1, "r--", label="strain gauge 2, shifted to interface")
    #plt.plot(t, favgshifted+1, "r--", label="average, shifted to interface")
    plt.legend()
    plt.show()


