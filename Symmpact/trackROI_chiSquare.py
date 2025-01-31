# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2025-01-27 17:50:34
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2025-01-31 14:33:47
import numpy as np
import copy
import pylab as plt
from scipy.optimize import minimize
from scipy.ndimage.interpolation import shift
import skimage

class TrackROI():
    def __init__(self, image_data, ROI_start, ROI_stop):
        
        """
        track the displacement of a region of interest (ROI) over several frames 
        """
        
        self.ROI_start = ROI_start
        self.ROI_stop = ROI_stop
        
        # load files
        self.allFrames = copy.deepcopy(image_data)
        self.numFrames = np.shape(image_data)[0]
        print("number of frames in image: ", self.numFrames)
        
        # load first frame and define ROI
        self.refFrame = self.yield_frame(0)
        self.numPixels = len(self.refFrame)
        
        # single out ROI by setting everything else in the first frame to zero
        self.ROI = (self.ROI_start, self.ROI_stop)
        
        # create arrays to save data calculated by this program
        self.displacements = np.zeros(self.numFrames)
        
        # iterate over all frames and find the shift of the reference image (template) to the current image.

        lastShift = 0.0
        for i in range(self.numFrames):
            
            currentFrame = self.yield_frame(i)
            
            #currentFrame2d = np.tile(currentFrame, (2,1))
            #print("shape of currentFrame", currentFrame.shape)


            #pixelShift, error, diffphase = skimage.registration.phase_cross_correlation(
            #    self.refFrame, currentFrame, reference_mask=self.ROI, disambiguate=False)
            s = self.chisqr_align(self.refFrame, currentFrame, self.ROI, init=lastShift,bound=50)

            lastShift = s

            self.displacements[i] = s
            print("shift:", s)


        plt.plot(self.displacements)
        #plt.plot(self.displacements + self.subPixelShifts, label="with subpix")
        plt.show()

    def equalize_array_size(self, array1,array2):
        '''
        reduce the size of one sample to make them equal size. 
        The sides of the biggest signal are truncated

        Args:
            array1 (1d array/list): signal for example the reference
            array2 (1d array/list): signal for example the target

        Returns:
            array1 (1d array/list): middle of the signal if truncated
            array2 (1d array/list): middle of the initial signal if there is a size difference between the array 1 and 2
            dif_length (int): size diffence between the two original arrays 
        '''
        len1, len2 = len(array1), len(array2)
        dif_length = len1-len2
        if dif_length<0:
            array2 = array2[int(np.floor(-dif_length/2)):len2-int(np.ceil(-dif_length/2))]
        elif dif_length>0:
            array1 = array1[int(np.floor(dif_length/2)):len1-int(np.ceil(dif_length/2))]
        return array1,array2, dif_length

    def chisqr_align(self, reference, target, roi=None, order=1, init=0.1, bound=1):
        '''
        Align a target signal to a reference signal within a region of interest (ROI)
        by minimizing the chi-squared between the two signals. Depending on the shape
        of your signals providing a highly constrained prior is necessary when using a
        gradient based optimization technique in order to avoid local solutions.

        Args:
            reference (1d array/list): signal that won't be shifted
            target (1d array/list): signal to be shifted to reference
            roi (tuple): region of interest to compute chi-squared
            order (int): order of spline interpolation for shifting target signal
            init (int):  initial guess to offset between the two signals
            bound (int): symmetric bounds for constraining the shift search around initial guess

        Returns:
            shift (float): offset between target and reference signal 

        Todo:
            * include uncertainties on spectra
            * update chi-squared metric for uncertainties
            * include loss function on chi-sqr

        https://github.com/pearsonkyle/Signal-Alignment

        '''
        reference, target, dif_length = self.equalize_array_size(reference,target)
        if roi==None: roi = [0,len(reference)-1] 
    
        # convert to int to avoid indexing issues
        ROI = slice(int(roi[0]), int(roi[1]), 1)

        # normalize ref within ROI
        reference = reference/np.mean(reference[ROI])

        # define objective function: returns the array to be minimized
        def fcn2min(x):
            shifted = shift(target,x,order=order)
            shifted = shifted/np.mean(shifted[ROI])
            return np.sum( ((reference - shifted)**2 )[ROI] )

        # set up bounds for pos/neg shifts
        minb = min( [(init-bound),(init+bound)] )
        maxb = max( [(init-bound),(init+bound)] )

        # minimize chi-squared between the two signals 
        #result = minimize(fcn2min,init,method='L-BFGS-B',bounds=[ (minb,maxb) ])
        result = minimize(fcn2min,init,method='Nelder-Mead',bounds=[ (minb,maxb) ], options={"xatol": 1.0e-4, "fatol" : 1.0e-4})

        return result.x[0] + int(np.floor(dif_length/2))

            
        
       
    def yield_frame(self, i):
        """ read frame number i """
        
        image = self.allFrames[i,:].astype(np.float32)
        return image


if __name__ == "__main__":
    filename = "/home/gcg/Projekte/21_WaveSeparation/2025-01-30_Waveseparation/02_PC/im.bmp"
    from matplotlib.pyplot import imread
    image_data = imread(filename)

    #image_data = skimage.filters.gaussian(image_data, sigma=(1.0, 0)) # scharfe Übergänge sichtbar
    image_data = skimage.filters.gaussian(image_data, sigma=(0, 2))

    x = np.arange(len(image_data))

    plt.imshow(image_data)
    plt.show()
    image_data = image_data.astype(np.float32) #[2300:2400,:].astype(np.float32)
    tracker = TrackROI(image_data, 485, 1740)

    filename_out = "/home/gcg/Projekte/21_WaveSeparation/2025-01-30_Waveseparation/02_PC/linescan_analysis.dat"
    np.savetxt(filename_out, np.column_stack((x, tracker.displacements)))
    print("wrote output file linescan_analysis.dat as [%s]" % (filename_out))