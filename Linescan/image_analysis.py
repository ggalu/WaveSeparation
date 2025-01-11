# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-19 09:41:44
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-28 22:13:14
"""
generate a synthetic linescan image
"""
import numpy as np
import pylab as plt
from scipy.optimize import minimize
from scipy.fft import rfft
from scipy.fft import rfftfreq
import copy

def generate_image(p):

    offset = p[0]
    wavelength = p[1]
    exposure_gradient = p[2]
    black_level = p[3]
    white_level = p[4]

    x = np.arange(4096)
    w = 2 * np.pi * (x-offset) / wavelength
    image = (white_level - black_level)  * 0.5 * (np.sin(w) + 1) + black_level
    image = np.sin(w)

    weights = np.ones(4096, dtype=float)
    #weights[:left_border  - int(wavelength) ] = 0.0
    #weights[right_border + int(wavelength): ] = 0.0

    # non-uniform exposure
    #mod = 1.0 - exposure_gradient * x / x.max()
    #image *= mod
    
    return image, weights

def metric(p1, image0):
    """
    Compute the difference between the image geneated using the current parameter set p and the target image0.
    """
    image1, weights = generate_image(p1)
    error  = image1 - image0
    #error *= weights
    return np.linalg.norm(error, ord=2)

def LSA(y):
    """
    Perform local spectrum analysis of the 1D input image y
    """
    #t = np.linspace(0.0, 10 * 2*np.pi, 1000)
    #dt = t[1] - t[0]
    #period = 2.0
    #signal = np.sin(2 * np.pi * t / period)

    signal = y
    dt = 1.0

    frequencies = rfftfreq(len(signal), d=dt)
    fourier = rfft(signal)
    amplitude = 2*np.abs(fourier)/len(signal)

    plt.plot(signal)
    plt.show()

    periods = 1.0 / frequencies
    periods[0] = 0

    #plt.plot(frequencies, amplitude)
    plt.plot(periods, amplitude)
    plt.show()

def generate_image_sequence(p0):
    """
    Starting from an initial parameter set p, generate N subsequent images
    """
    N = 100 # number of images
    velocity = 0.5 # displacement [pixel/frame]
    max_strain = 0.005 
    strain = np.linspace(0.0, max_strain, N)

    noise_amplitude = 0.05

    IMG = np.zeros((N, 4096))
    for i in range(N):
        p = copy.copy(p0)
        p[0] += velocity * i
        p[1] *= 1.0 + strain[i] 
        noise = noise_amplitude * (np.random.random(4096) - 0.5)

        IMG[i] = generate_image(p)[0] + noise

    return IMG




# image parameters
offset = 1.23
wavelength = 23.76
exposure_gradient = 0.1
white_level = 0.75
black_level = 0.25

p0 = [offset, wavelength, exposure_gradient, black_level, white_level]


# generate the image
image0, weights = generate_image(p0)
noise_amplitude = 0.0
noise0 = noise_amplitude * (np.random.random(4096) - 0.5)
image0 += noise0

#plt.plot(image0)
#plt.show()

IMG = generate_image_sequence(p0)
#plt.imshow(IMG)
#plt.show()

# generate initial guess from which to start optimisation
noise_amplitude_parameters = 0.02
noise_parameters = noise_amplitude_parameters * (np.random.random(len(p0)) - 0.5)
p1 = p0 + p0 * noise_parameters

# iterate over each image
offsets = []
wavelengths = []
for i in range(IMG.shape[0]):
    res = minimize(metric, p1, args=(IMG[i]), method='Nelder-Mead', options={'gtol': 1e-12, 'ftol': 1.0e-12, 'disp': False})
    print(f"solution {i}: offset {res.x[0]} | wavelength {res.x[1]} ")
    offsets.append(res.x[0])
    wavelengths.append(res.x[1])
    p1 = res.x


plt.plot(offsets)
#plt.plot(wavelengths)
plt.show()

import sys
sys.exit()



# minimize
res = minimize(metric, p1, args=(image0), method='Nelder-Mead', options={'gtol': 1e-12, 'ftol': 1.0e-12, 'disp': True})
offset_error = 100 * (res.x[0] - offset) / offset
wavelength_error = 100 * (res.x[1] - wavelength) / wavelength
print(f"solution: offset {res.x[0]} ({offset_error}) | wavelength {res.x[1]} ({wavelength_error})")

image1, weights1 = generate_image(p1)
image2, weights2 = generate_image(res.x)

#print("metric is:", metric(p1, image0))



plt.plot(image0, label="target image")
plt.plot(weights, label="weights")
#plt.plot(image1, label="profile guess")
plt.plot(image2, label="fitted profile")
plt.legend()
#plt.plot(image1)
plt.show()