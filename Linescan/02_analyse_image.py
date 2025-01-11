# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-19 09:41:44
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-06 18:02:21
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

def generate_image(p, nPix):

    offset = p[0]
    wavelength = p[1]
    exposure_gradient = p[2]
    black_level = p[3]
    white_level = p[4]

    #print("P", p)

    x = np.arange(nPix)
    w = 2 * np.pi * (x-offset) / wavelength
    image = (white_level - black_level)  * 0.5 * (np.sin(w) + 1) + black_level

    # non-uniform exposure
    mod = 1.0 - exposure_gradient * x / x.max()
    image *= mod
    
    return (image * 255).astype(np.uint8)

def metric(p, image0):
    """
    Compute the difference between the image geneated using the current parameter set p and the target image0.
    """
    nPix = len(image0)
    image1 = generate_image(p, nPix)
    error  = image1.astype(float) - image0.astype(float)

    noise_amplitude = 0.05
    noise = noise_amplitude * (np.random.random(nPix) - 0.5)
    error += noise

    return np.linalg.norm(error, ord=2)

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

    periods = 1.0 / frequencies

    #periods[0] = periods[1]
    #periods[:5] = 0
    mag[0] = 0

    idx = np.argmax(mag)
    period = periods[idx]
    phase = np.angle(fourier[idx]) * period / (2 * np.pi)
    print(f"maximum magnitude occurs at index {idx}, wavelength = {period}, phase = {phase}")

    #for i in range(100):
    #    print(periods[i])
    #sys.exit()
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, sharex=False)
        ax1.plot(signal)
        ax2.plot(periods, mag)
        #ax2.loglog(periods)
        ax2.axvline(periods[idx])
        #print("freq", frequencies)
        print("periods", periods)
        plt.show()

    #sys.exit()

    return period, phase

def intensity_distribution(signal, wavelength):
    """
    Compute the intensity distribution and smooth it with the wavelength
    """
    window_size = 2 * int(wavelength) + 1
    intensity_distribution = bn.move_mean(signal, window_size)
    #plt.plot(signal / intensity_distribution)
    #plt.plot(signal)
    #plt.show()
    #sys.exit()
    return intensity_distribution

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

def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res

def freq_from_HPS(sig, fs):
    """
    Estimate frequency using harmonic product spectrum (HPS)

    """
    windowed = sig * blackmanharris(len(sig))

    from pylab import subplot, plot, log, copy, show

    # harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 8
    subplot(maxharms, 1, 1)
    plot(log(c))
    for x in range(2, maxharms):
        a = copy(c[::x])  # Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = np.argmax(abs(c))
        true_i = parabolic(abs(c), i)[0]
        print('Pass %d: %f Hz' % (x, fs * true_i / len(windowed)))
        c *= a
        subplot(maxharms, 1, x)
        plot(log(c))
    show()

def freq_from_autocorr(signal, fs):
    """
    Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, doesn't find fundamental for inharmonic things like
    musical instruments, this implementation has trouble with finding the true
    peak
    """
    signal = np.asarray(signal) + 0.0

    # Calculate autocorrelation, and throw away the negative lags
    signal -= np.mean(signal)  # Remove DC offset
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first valley in the autocorrelation
    d = np.diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    i_peak = np.argmax(corr[start:]) + start
    i_interp = parabolic(corr, i_peak)[0]

    return i_interp


#filename = r"../patterns/Gimp-Pattern.tif"
filename = r"real_images/1.tif"
IMG = tifffile.imread(filename)
nPix = IMG.shape[1]





wavelength, offset = initial_guess_FFT(IMG[0,:], plot=False)
print("FFT wavelength #1:", wavelength)

signal = intensity_distribution(IMG[0,:], wavelength)
wavelength, offset = initial_guess_FFT(IMG[0,:])
print("FFT wavelength #2:", wavelength)

wavelength = freq_from_autocorr(IMG[:,0], 0)
print("wavelength:", wavelength)


exposure_gradient = 0.0
white_level = 0.6
black_level = 0.05
offset = 40

p0 = [offset, wavelength, exposure_gradient, black_level, white_level]

#print("INITIAL METRIC:", metric(p0, IMG[0,:]))
plt.plot(IMG[0,:], label="first image")
plt.plot(generate_image(p0, nPix), label="guess")
res = minimize(metric, p0, args=(IMG[0,:]), method='Nelder-Mead', options={'disp': True})
res = minimize(metric, res.x, args=(IMG[0,:]), method='Powell', options={'xtol': 1e-12, 'ftol': 1.0e-12, 'disp': True})
#plt.plot(generate_image(res.x, nPix), label="fit")
#print("FINAL  METRIC:", metric(res.x, IMG[0,:]))
plt.legend()
plt.show()

sys.exit()



# iterate over each image
offsets = []
wavelengths = []
for i in range(IMG.shape[0]):
    #res = minimize(metric, res.x, args=(IMG[i,:]), method='Powell', options={'xtol': 1e-12, 'ftol': 1.0e-12, 'disp': False})
    #print(f"solution {i}: offset {res.x[0]} | wavelength {res.x[1]} ")
    #offsets.append(res.x[0])
    #wavelengths.append(res.x[1])


    wavelengths.append(wavelength)



fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Linescan analysis')
ax1.plot(offsets)

lambda0 = wavelengths[0] 
ax2.plot((wavelengths - lambda0) / lambda0 )
ax2.set_xlabel("strain")
plt.show()
