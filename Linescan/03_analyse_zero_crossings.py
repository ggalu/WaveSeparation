# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-19 09:41:44
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-10 22:43:38
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

def initial_guess_FFT(signal, plot=False):
    """
    Extract frequency and phase using FFT
    """

    # zero-pad signal
    #signal = np.pad(signal, (len(signal), len(signal)), 'constant')

    from scipy.fft import rfft, rfftfreq
    fourier = rfft(signal)
    mag = 2*np.abs(fourier)/len(signal)
    frequencies = rfftfreq(len(signal))
    #print("freq 0", frequencies[0], frequencies[-1])
    frequencies[0] = frequencies[1]
    mag[:10] = 0

    idx = np.argmax(mag)
    frequency = frequencies[idx]
    

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, sharex=False)
        ax1.plot(signal)
        ax2.plot(frequencies, mag)
        #ax2.loglog(periods)
        ax2.axvline(frequencies[idx])
        plt.show()

    #sys.exit()

    return frequency

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 1):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def bandpass2(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 1):

    # The band-pass filter will pass signals with frequencies between
    # low_end_cutoff and high_end_cutoff
    sampling_freq = 1.0
    low_end_cutoff = edges[0]
    lo_end_over_Nyquist = low_end_cutoff/(0.5*sampling_freq)
    high_end_cutoff = edges[1]
    hi_end_over_Nyquist = high_end_cutoff/(0.5*sampling_freq)

    bess_b,bess_a = scipy.signal.iirfilter(5,
            Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],
            btype="bandpass", ftype='bessel')
    filtered_data = scipy.signal.filtfilt(bess_b, bess_a, data)
    return filtered_data

def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res

def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    if int(x) != x:
        raise ValueError('x must be an integer sample index')
    else:
        x = int(x)
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def freq_from_crossings(signal, interp='linear'):
    """
    Estimate frequency by counting zero crossings

    Works well for long low-noise sines, square, triangle, etc.

    Pros: Fast, accurate (increasing with signal length).

    Cons: Doesn't work if there are multiple zero crossings per cycle,
    low-frequency baseline shift, noise, inharmonicity, etc.
    """
    signal = np.asarray(signal) + 0.0

    # Find all indices right before a rising-edge zero crossing
    indices = find((signal[1:] >= 0) & (signal[:-1] < 0))

    # linear interpolation of crossing locations
    crossings = [i - signal[i] / (signal[i+1] - signal[i]) for i in indices]

    # center of all crossings:
    offset = np.mean(crossings)

    #print("number of crossings", len(crossings))

    return np.mean(np.diff(crossings)), offset

def pad_signal(x, phase_offset=0):
    N = 8
    output = np.pad(x, (N * 4096, N * 4096))
    if phase_offset != 0:
        output = np.roll(output, phase_offset)
    return output


def freq_from_fft(signal):
    """
    Estimate frequency from peak of FFT

    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000004 Hz for 1000 Hz, for instance).  Due to parabolic
    interpolation being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with signal length

    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.
    """

    N = len(signal)

    # Compute Fourier transform of windowed signal
    #windowed = signal * scipy.signal.windows.kaiser(N, 10)
    f = rfft(signal)

    # Find the peak and interpolate to get a more accurate peak
    i_peak = np.argmax(abs(f))  # Just use this value for less-accurate result
    i_interp = parabolic(np.log(abs(f)), i_peak)[0]

    ## Convert to period
    return N / i_interp
    # Convert to equivalent frequency
    #return i_interp / N  # Hz

def unwrap_phases(phases, wavelength):
    """
    unwrap
    """

    phases = np.asarray(phases)
    # difference in phases is displacement. These suffer from jump discontinuities due to phase wrapping.
    deltas = phases[1:] - phases[:-1]

    print(f"PHASE_UNWRAPPING : wavelength is {wavelength}")
    # phase wrap jumps are indicated by delta values close to wavelength/2
    values = deltas[abs(deltas) > 0.4 * wavelength]
    signs = np.sign(values)
    direction = np.mean(signs)
    #print("values:", values)
    #print("direction:", direction)
    
    if direction > 0:
        print(f"PHASE_UNWRAPPING : travel direction is POSITIVE")
        indices = find((deltas[1:] <= 0) & (deltas[:-1] > 0))
    else:
        print(f"PHASE_UNWRAPPING : travel direction is NEGATIVE")
        indices = find((deltas[1:] >= 0) & (deltas[:-1] < 0))

    if True:
        plt.plot(deltas)
        for i in indices:
            plt.axvline(i, color="red")
        plt.title("displacement discontinuities due to phase wrapping")
        plt.ylabel("displacement")
        plt.show()

    # at each index, need to offset by the last phase
    phases_corr = 1.0 * phases
    for idx in indices:
        print(idx, deltas[idx])
        phases_corr[idx+1:] -= deltas[idx]

    # numpy solution
    phases_corr = np.unwrap(phases, discont=0.4*wavelength, period = 0.5*wavelength)

    velocity = phases_corr[1:] - phases_corr[:-1]

    #plt.plot(phases_corr)
    plt.plot(velocity)
    plt.show()


#filename = r"../patterns/Gimp-Pattern.tif"
filename = r"real_images/img.tif"
IMG = tifffile.imread(filename)
nPix = IMG.shape[1]


signal = pad_signal(IMG[10,:])
freq0 = initial_guess_FFT(signal, plot=False)
#sys.exit()
print("initial guess for freq:", freq0, 1.0/freq0)
filtered = bandpass(signal, [0.8*freq0, 1.2*freq0], 1.0)

period, offset = freq_from_crossings(filtered)
print(f"period from zero-crossings is {period}")

period = freq_from_fft(filtered)
print(f"period from interpolated FFT is {period}")
#
#plt.plot(signal)
#plt.plot(filtered)
#plt.grid()
#plt.show()
#
#sys.exit()

# TODO: compute phase from FFT instead from zero-crossings

# iterate over each image
offsets = []
wavelengths = []
periods_ipFFT = []
for i in range(IMG.shape[0]):
    print(f"line Nr. {i}")
    signal = pad_signal(IMG[i,:], phase_offset=0)
    filtered = bandpass(signal, [0.8*freq0, 1.2*freq0], 1.0)
    wavelength, offset = freq_from_crossings(filtered)
    offsets.append(offset)
    #wavelengths.append(wavelength)
    periods_ipFFT.append(freq_from_fft(filtered))

    freq_FFT = initial_guess_FFT(filtered, plot=False)
    wavelengths.append(1.0/freq_FFT)

lambda0 = wavelengths[0] 
#unwrap_phases(offsets, lambda0)
#sys.exit()


fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Linescan analysis')
ax1.plot(offsets - offsets[0])

lambda0 = wavelengths[0] 
ax2.plot((wavelengths - lambda0) / lambda0, label="FFT" )

lambda0 = periods_ipFFT[0] 
strains = (periods_ipFFT - periods_ipFFT[0])/periods_ipFFT[0]
ax2.plot(strains, label="ipol FFT")
ax2.legend()
ax2.set_xlabel("strain")
plt.show()
