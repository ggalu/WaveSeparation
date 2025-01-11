# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2024-12-19 09:41:44
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2024-12-28 22:13:24
"""
generate a synthetic linescan image
"""
import numpy as np
import pylab as plt
import copy
import tifffile

def generate_image(p):

    offset = p[0]
    wavelength = p[1]
    exposure_gradient = p[2]
    black_level = p[3]
    white_level = p[4]

    x = np.arange(4096)
    w = 2 * np.pi * (x-offset) / wavelength
    image = (white_level - black_level)  * 0.5 * (np.sin(w) + 1) + black_level
    #image = np.sin(w)

    # non-uniform exposure
    mod = 1.0 - exposure_gradient * x / x.max()
    image *= mod

    #print("max, min", image.min(), image.max())
    
    #return image
    return (image * 255).astype(np.uint8)

def generate_image_sequence(p0):
    """
    Starting from an initial parameter set p, generate N subsequent images
    """
    N = 1000 # number of images
    velocity = 0.5 # displacement [pixel/frame]
    max_strain = 0.005 
    strain = np.linspace(0.0, max_strain, N)

    noise_amplitude = 0.15

    IMG = np.zeros((N, 4096))
    for i in range(N):
        p = copy.copy(p0)
        p[0] += velocity * i
        p[1] *= 1.0 + strain[i] 
        noise = noise_amplitude * (np.random.random(4096) - 0.5)

        IMG[i] = generate_image(p) + noise

    return IMG

# image parameters
offset = 1.23
wavelength = 32.7
exposure_gradient = 0.1
white_level = 0.75
black_level = 0.25

p0 = [offset, wavelength, exposure_gradient, black_level, white_level]


# generate a single image
image0 = generate_image(p0)
#noise_amplitude = 0.0
#noise0 = noise_amplitude * (np.random.random(4096) - 0.5)
#image0 += noise0

plt.plot(image0)
plt.show()

IMG = generate_image_sequence(p0).astype(np.uint8)
plt.imshow(IMG)
plt.show()

tifffile.imwrite("img.tif", IMG)

