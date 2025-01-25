# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-16 11:37:39
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-16 15:27:33
"""
generate a synthetic linescan image
"""
import numpy as np
import pylab as plt
import copy
import tifffile

# final image dimensions
width  = 100.0
height = 62.0
dpmm = 300 / 25.4 # dot per millimeter from dpi
nPixX = int(width * dpmm)
nPixY = int(height * dpmm)

# speckle size: we want 4 pixels
ratio = 4096 / nPixX
print(ratio)


# generate a vector of random 0 / ones

line = np.random.rand(nPixX)
line = np.where(line > 0.5, 1, 0)

print(line)

img = np.tile(line, (nPixY,1))


img = img.T
img = (img * 255).astype(np.uint8)

print(img.shape)

plt.imshow(img)
plt.show()

#import matplotlib.image
#matplotlib.image.imsave('random.bmp', img.T)


tifffile.imwrite("random.tif", img)

