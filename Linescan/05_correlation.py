# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-10 20:36:35
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-11 13:36:05
import numpy as np
import tifffile
from scipy.optimize import minimize
from scipy import interpolate
import pylab as plt

class LineScanAnalyzer:
    def __init__(self, filename):
        self.img = tifffile.imread(filename)
        self.nLines = self.img.shape[0]
        self.nPix = self.img.shape[1]
        self.x0 = np.arange(self.nPix)

        
        displacement = 2.0
        strain = 0.01
        p0 = [displacement, strain]

        
        otherline = self.img[99,:]
        self.metric(p0, otherline)

        res = minimize(self.metric, p0, args=(otherline,), method="Nelder-Mead")
        print("res:", res)

        # (1) load images
        # (2) define reference image as interpoland
        # (3) for each line, call a minimizer which minimizes the difference between the affine deformation of the reference image, and the line, thus providing the affine parameters

        self.iterate_over_lines()

    def iterate_over_lines(self):
        
        p0 = [1.0e-2, -1.0e-5]
        displacements = np.zeros(self.nLines)
        strains = np.zeros(self.nLines)
        #for i in range(self.nLines):
        for i in range(100):
            res = minimize(self.metric, p0, args=(self.img[i,:],), method="Nelder-Mead", options={"xatol":1.0e-6, "fatol": 1.0e-3})
            displacements[i] = res.x[0]
            strains[i] = res.x[1]
            p0 = res.x
            print(f"line {i} ... {res.success} ... {res.fun}")


        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('Linescan analysis')

        # estimate strain level
        F = 0.1 # kN
        A = np.pi * 20 * 20 # bar cross section, mm²
        E = 2.4 # Young's modulus, GPa
        reference_strain = F / (A * E)
        ax1.axhline(reference_strain, color="red")
        ax1.plot(strains)
        ax1.set_ylabel("strain")

        ax2.plot(displacements)
        ax2.set_ylabel("displacement")

        

        plt.show()



    def metric(self, p, line):
        # for a given modification parameter set p [displacement, strain],
        # compute the similarity between the modified reference line and the supplied line.

        #print("displacment:", p[0], "strain:", p[1])
        x_mod, f = self.create_interpoland(displacement=p[0], strain=p[1])
        
        # We can only compute the similarity within the common x-axis range between x0 and x_mod
        #print("xmod range:", x_mod[0])
        xmin = int(max(x_mod[0] , self.x0[0])) + 1
        xmax = int(min(x_mod[-1], self.x0[-1])) - 1

        #print(f"common range is {xmin} -- {xmax}")
        indices = np.arange(xmin, xmax)
        #print("indices", indices)
        f_mod = f(indices)

        metric = np.abs(f_mod - line[indices])
        #plt.plot(metric)
        #plt.show()
        #print("f_mod:", f_mod)
        #print(line)
        #print("metric: ", metric.sum())
        return metric.sum()

        

    def create_interpoland(self, displacement=0.0, strain=0.0):
        """
        Create an interpolation of the reference line, modified by a given translation and a strain.
        """

        x_mod = (1.0 + strain) * (self.x0 + displacement) # these are the distorted coordinates
        reference_line = self.img[0,:]
        f = interpolate.interp1d(x_mod, reference_line)

        #plt.plot(self.x0, reference_line)
        #plt.plot(x_mod, f(x_mod), label="mod")
        #plt.show()

        return x_mod, f



if __name__ == '__main__':
    instance = LineScanAnalyzer("real_images/img.tif")