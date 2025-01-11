# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-19 09:41:44
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-09 17:41:18
"""
generate a synthetic linescan image
"""
import numpy as np
import pylab as plt
from scipy.optimize import minimize
from scipy.fft import rfft
from scipy.fft import rfftfreq
import tifffile
import sys
import bottleneck as bn
from scipy.signal.windows import kaiser
import scipy

def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res

def freq_from_crossings(signal, interp='linear'):
    """
    Estimate frequency by counting zero crossings

    Works well for long low-noise sines, square, triangle, etc.

    Pros: Fast, accurate (increasing with signal length).

    Cons: Doesn't work if there are multiple zero crossings per cycle,
    low-frequency baseline shift, noise, inharmonicity, etc.
    """
    signal = signal.astype(float)
    signal -= np.mean(signal)

    #plt.plot(signal)
    #plt.show()

    # Find all indices right before a rising-edge zero crossing
    indices = find((signal[1:] >= 0) & (signal[:-1] < 0))

    # linear interpolation of crossing locations
    crossings = [i - signal[i] / (signal[i+1] - signal[i]) for i in indices]

    # center of all crossings:
    offset = np.mean(crossings)

    #print("number of crossings", len(crossings))

    return np.mean(np.diff(crossings)), offset

def gauss_kern(size):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    x = np.arange(-size, size+1)
    g = np.exp(-x**2/float(size))
    return g / g.sum()

def generate_image(wavelength, nsmooth = 8):
    N = 4096
    indices = np.arange(N)
    
    offset = 0
    w = 2 * np.pi * (indices-offset) / wavelength
    line = 0.5 * (np.sin(w) + 1) - 0.5

    #noise_amplitude = 0.05
    #noise = noise_amplitude * (np.random.random(4096) - 0.5)
    #line += noise

    binary_line = np.where(line >= 0, 255, 0)
    if nsmooth > 0:
        g = gauss_kern(128)
        binary_line = scipy.signal.convolve(binary_line, g, mode='valid')
        #binary_line = bn.move_mean(binary_line, nsmooth, min_count=32) #.astype(np.uint8)
    #plt.plot(binary_line)
    #plt.show()

    return binary_line


wavelength0 = 128
line = generate_image(wavelength0, nsmooth=32)
plt.plot(line)
plt.show()

#sys.exit()
#period, offset = freq_from_crossings(line)


wavelength0 = 32
errors = []
strains = []
for i in range(100):
    wavelength = wavelength0 * (1 + 1.0e-4*i)
    line = generate_image(wavelength)
    period, offset = freq_from_crossings(line)
    strain_ideal  = (wavelength - wavelength0) / wavelength0
    strain_actual = (period - wavelength0) / wavelength0
    error = 100 * (strain_actual- strain_ideal)
    print(f"period is {period}, dtected strain is {strain_actual} imposed strain is {strain_ideal}, error is {error}%")

    errors.append(error)
    strains.append(strain_actual)

plt.plot(strains, errors)
plt.show()

sys.exit()

#filename = r"real_images/1.tif"
IMG = tifffile.imread(filename)
nPix = IMG.shape[1]


signal = time_reverse(IMG[0,:])
#signal = np.roll(IMG[0,:], 80) + 0.0
#signal = signal - np.mean(signal)
#signal = kaiser(signal, 100)
freq = initial_guess_FFT(signal, plot=True)
#sys.exit()
print("initial guess for freq:", freq, 1.0/freq)
filtered = bandpass(signal, [0.8*freq, 1.2*freq], 1.0)


for i in range(100):
    signal = np.roll(IMG[0,:], i)
    signal = time_reverse(signal)
    filtered = bandpass(signal, [0.8*freq, 1.2*freq], 1.0)
    freq2, offset = freq_from_crossings(filtered)
    print("freq2:", 1.0/freq2)

#sys.exit()


plt.plot(signal)
plt.plot(filtered)
plt.grid()
plt.show()

sys.exit()

# iterate over each image
offsets = []
wavelengths = []
for i in range(IMG.shape[0]):

    

    filtered = bandpass(signal, [0.8*freq, 1.2*freq], 1.0)
    freq, offset = freq_from_crossings(filtered)
    wavelength = 1.0/freq
    offsets.append(offset)
    wavelengths.append(wavelength)



fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Linescan analysis')
ax1.plot(offsets)

lambda0 = wavelengths[0] 
ax2.plot((wavelengths - lambda0) / lambda0 )
ax2.set_xlabel("strain")
plt.show()
